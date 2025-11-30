/*
 * SSA - Singular Spectrum Analysis (FFT-Accelerated)
 * 
 * Fast SSA using the Hankel matrix structure:
 *   - Matrix-vector products via FFT convolution: O(N log N)
 *   - Lanczos/power iteration for eigenvectors: O(k × iter × N log N)
 *   - Reconstruction via convolution: O(k × N log N)
 *   - Total: O(k × N log N) instead of O(N³)
 * 
 * Key insight: The trajectory (Hankel) matrix X has structure:
 *   X[i,j] = x[i+j]
 * 
 * This means X @ v is a convolution, computed efficiently via FFT.
 * 
 * Usage:
 *   SSA ssa;
 *   ssa_init(&ssa, prices, n_prices, window_length);
 *   ssa_decompose(&ssa, 20, 100);  // 20 components, 100 iterations
 *   
 *   double* trend = malloc(n_prices * sizeof(double));
 *   ssa_get_trend(&ssa, trend);
 *   
 *   ssa_free(&ssa);
 * 
 * Compile:
 *   gcc -O3 -o ssa_test ssa.c -lm -lfftw3
 *   (or link against your own FFT)
 * 
 * Author: Research Library
 * License: MIT
 */

#ifndef SSA_H
#define SSA_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Configuration
// ============================================================================

// Use your own FFT by defining SSA_CUSTOM_FFT and implementing:
//   void ssa_fft_forward(double* data, int n);   // In-place real-to-complex
//   void ssa_fft_inverse(double* data, int n);   // In-place complex-to-real
//   int  ssa_fft_alloc_size(int n);              // Required allocation size
//
// Otherwise, we use a simple built-in FFT (slower but self-contained)

#ifndef SSA_CUSTOM_FFT
#define SSA_BUILTIN_FFT 1
#endif

// Default convergence tolerance for power iteration
#ifndef SSA_CONVERGENCE_TOL
#define SSA_CONVERGENCE_TOL 1e-12
#endif

// ============================================================================
// Data Structures
// ============================================================================

typedef struct {
    // Input parameters
    int N;                  // Original series length
    int L;                  // Window (embedding) length
    int K;                  // K = N - L + 1
    
    // Precomputed FFTs for fast Hankel products
    double* fft_x;          // FFT of zero-padded x
    double* fft_x_rev;      // FFT of zero-padded reverse(x)
    int fft_len;            // Padded FFT length (power of 2)
    
    // Decomposition results
    double* U;              // Left singular vectors: L × n_components (column-major)
    double* V;              // Right singular vectors: K × n_components (column-major)
    double* sigma;          // Singular values: n_components
    double* eigenvalues;    // Squared singular values (for variance analysis)
    int n_components;       // Number of components computed
    
    // Workspace (reused across operations)
    double* work1;          // FFT workspace 1
    double* work2;          // FFT workspace 2
    double* work3;          // General workspace
    
    // State
    bool initialized;
    bool decomposed;
    double total_variance;  // Sum of all eigenvalues
} SSA;

// Component grouping for reconstruction
typedef struct {
    int* indices;           // Which components to include
    int count;              // Number of components in group
    char name[32];          // Optional name (e.g., "trend", "cycle1")
} SSAGroup;

// ============================================================================
// Core API
// ============================================================================

/**
 * Initialize SSA with input series.
 * 
 * @param ssa       SSA structure to initialize
 * @param x         Input time series
 * @param N         Length of series
 * @param L         Window length (typically N/4 to N/2)
 * @return          0 on success, -1 on error
 * 
 * Guideline for L:
 *   - L should capture the longest periodicity of interest
 *   - L ≈ N/2 gives best separability
 *   - L ≈ N/4 is more robust to non-stationarity
 */
int ssa_init(SSA* ssa, const double* x, int N, int L);

/**
 * Compute top k singular value decomposition.
 * 
 * @param ssa       Initialized SSA structure
 * @param k         Number of components to compute
 * @param max_iter  Maximum iterations per component (50-200 typical)
 * @return          0 on success, -1 on error
 * 
 * After this call:
 *   - ssa->U contains left singular vectors (L × k)
 *   - ssa->V contains right singular vectors (K × k)
 *   - ssa->sigma contains singular values
 *   - ssa->eigenvalues contains σ² (variance per component)
 */
int ssa_decompose(SSA* ssa, int k, int max_iter);

/**
 * Reconstruct time series from a group of components.
 * 
 * @param ssa       Decomposed SSA structure
 * @param group     Array of component indices to include
 * @param n_group   Number of components in group
 * @param output    Output buffer (length N)
 * @return          0 on success, -1 on error
 */
int ssa_reconstruct(const SSA* ssa, const int* group, int n_group, 
                    double* output);

/**
 * Free all allocated memory.
 */
void ssa_free(SSA* ssa);

// ============================================================================
// Convenience Functions
// ============================================================================

/**
 * Get trend component (first singular vector).
 * For most financial series, component 0 captures the trend.
 */
int ssa_get_trend(const SSA* ssa, double* output);

/**
 * Get noise component (last n components).
 * 
 * @param ssa           Decomposed SSA structure  
 * @param noise_start   First component to consider noise
 * @param output        Output buffer (length N)
 */
int ssa_get_noise(const SSA* ssa, int noise_start, double* output);

/**
 * Get oscillatory components (components start to end, inclusive).
 */
int ssa_get_oscillatory(const SSA* ssa, int start, int end, double* output);

/**
 * Get variance explained by component range.
 * 
 * @param ssa    Decomposed SSA structure
 * @param start  First component (inclusive)
 * @param end    Last component (inclusive), or -1 for all remaining
 * @return       Fraction of variance explained (0 to 1)
 */
double ssa_variance_explained(const SSA* ssa, int start, int end);

/**
 * Find optimal window length via heuristic.
 * Tests multiple L values and returns one that maximizes eigenvalue gaps.
 * 
 * @param x      Input series
 * @param N      Series length
 * @param L_min  Minimum window to test
 * @param L_max  Maximum window to test
 * @return       Recommended L
 */
int ssa_find_optimal_L(const double* x, int N, int L_min, int L_max);

/**
 * Identify paired eigenvalues (oscillatory components).
 * Returns indices of components that appear in pairs (similar eigenvalues).
 * 
 * @param ssa         Decomposed SSA structure
 * @param pairs       Output: array of pairs (2 × n_pairs)
 * @param max_pairs   Maximum pairs to find
 * @param tolerance   Relative tolerance for pairing (e.g., 0.05 for 5%)
 * @return            Number of pairs found
 */
int ssa_find_pairs(const SSA* ssa, int* pairs, int max_pairs, double tolerance);

// ============================================================================
// Advanced API
// ============================================================================

/**
 * Compute w-correlation matrix for component grouping.
 * High w-correlation indicates components should be grouped together.
 * 
 * @param ssa        Decomposed SSA structure
 * @param wcorr      Output: k × k correlation matrix (column-major)
 * @param k          Number of components to analyze
 */
int ssa_wcorrelation(const SSA* ssa, double* wcorr, int k);

/**
 * Online update: incorporate new data point.
 * Approximates updated decomposition without full recomputation.
 * 
 * @param ssa        Previously decomposed SSA
 * @param new_value  New data point to append
 * @return           0 on success, -1 if full recomputation needed
 */
int ssa_update_online(SSA* ssa, double new_value);

/**
 * SSA-based forecasting.
 * Extrapolates components forward using linear recurrence.
 * 
 * @param ssa        Decomposed SSA structure
 * @param group      Components to use for forecast
 * @param n_group    Number of components
 * @param horizon    Number of steps to forecast
 * @param forecast   Output buffer (length horizon)
 */
int ssa_forecast(const SSA* ssa, const int* group, int n_group,
                 int horizon, double* forecast);

// ============================================================================
// Implementation
// ============================================================================

#ifdef SSA_IMPLEMENTATION

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

// ----------------------------------------------------------------------------
// Built-in FFT (Cooley-Tukey radix-2)
// Replace with your optimized version for better performance
// ----------------------------------------------------------------------------

#ifdef SSA_BUILTIN_FFT

static int ssa_next_pow2(int n) {
    int p = 1;
    while (p < n) p <<= 1;
    return p;
}

// In-place FFT, interleaved real/imag format
// data[2*i] = real, data[2*i+1] = imag
static void ssa_fft_inplace(double* data, int n, int sign) {
    // Bit-reversal permutation
    int j = 0;
    for (int i = 0; i < n - 1; i++) {
        if (i < j) {
            double tr = data[2*j], ti = data[2*j+1];
            data[2*j] = data[2*i]; data[2*j+1] = data[2*i+1];
            data[2*i] = tr; data[2*i+1] = ti;
        }
        int k = n >> 1;
        while (k <= j) { j -= k; k >>= 1; }
        j += k;
    }
    
    // Cooley-Tukey
    for (int len = 2; len <= n; len <<= 1) {
        double ang = sign * 2.0 * M_PI / len;
        double wpr = cos(ang), wpi = sin(ang);
        
        for (int i = 0; i < n; i += len) {
            double wr = 1.0, wi = 0.0;
            for (int jj = 0; jj < len/2; jj++) {
                int a = i + jj;
                int b = i + jj + len/2;
                double tr = wr * data[2*b] - wi * data[2*b+1];
                double ti = wr * data[2*b+1] + wi * data[2*b];
                data[2*b] = data[2*a] - tr;
                data[2*b+1] = data[2*a+1] - ti;
                data[2*a] += tr;
                data[2*a+1] += ti;
                double wt = wr;
                wr = wr * wpr - wi * wpi;
                wi = wi * wpr + wt * wpi;
            }
        }
    }
    
    // Scale inverse
    if (sign == 1) {
        double scale = 1.0 / n;
        for (int i = 0; i < 2*n; i++) {
            data[i] *= scale;
        }
    }
}

static void ssa_fft_forward(double* data, int n) {
    ssa_fft_inplace(data, n, -1);
}

static void ssa_fft_inverse(double* data, int n) {
    ssa_fft_inplace(data, n, 1);
}

static int ssa_fft_alloc_size(int n) {
    return 2 * ssa_next_pow2(n);  // Complex interleaved
}

#endif // SSA_BUILTIN_FFT

// ----------------------------------------------------------------------------
// Helper Functions
// ----------------------------------------------------------------------------

static double ssa_dot(const double* a, const double* b, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

static double ssa_norm(const double* v, int n) {
    return sqrt(ssa_dot(v, v, n));
}

static void ssa_scale(double* v, int n, double s) {
    for (int i = 0; i < n; i++) {
        v[i] *= s;
    }
}

static double ssa_normalize(double* v, int n) {
    double norm = ssa_norm(v, n);
    if (norm > 1e-15) {
        ssa_scale(v, n, 1.0 / norm);
    }
    return norm;
}

static void ssa_axpy(double* y, const double* x, double a, int n) {
    for (int i = 0; i < n; i++) {
        y[i] += a * x[i];
    }
}

static int ssa_min(int a, int b) { return a < b ? a : b; }
static int ssa_max(int a, int b) { return a > b ? a : b; }

// Complex multiply: c = a * b (interleaved format)
static void ssa_complex_mul(const double* a, const double* b, double* c, int n) {
    for (int i = 0; i < n; i++) {
        double ar = a[2*i], ai = a[2*i+1];
        double br = b[2*i], bi = b[2*i+1];
        c[2*i]   = ar * br - ai * bi;
        c[2*i+1] = ar * bi + ai * br;
    }
}

// ----------------------------------------------------------------------------
// Hankel Matrix-Vector Products via FFT
// ----------------------------------------------------------------------------

// Compute y = X @ v where X is the L×K Hankel matrix
// X[i,j] = x[i+j], so y[i] = sum_j x[i+j] * v[j] = (x ★ reverse(v))[i + K - 1]
static void ssa_hankel_matvec(
    const SSA* ssa,
    const double* v,      // Input: length K
    double* y             // Output: length L
) {
    int K = ssa->K;
    int L = ssa->L;
    int n = ssa->fft_len;
    double* work = ssa->work1;
    
    // Pack reversed v into complex format
    memset(work, 0, 2 * n * sizeof(double));
    for (int i = 0; i < K; i++) {
        work[2*i] = v[K - 1 - i];  // Reverse
    }
    
    // FFT of reversed v
    ssa_fft_forward(work, n);
    
    // Pointwise multiply with precomputed FFT(x)
    ssa_complex_mul(ssa->fft_x, work, work, n);
    
    // Inverse FFT
    ssa_fft_inverse(work, n);
    
    // Extract y = conv[K-1 : K-1+L]
    for (int i = 0; i < L; i++) {
        y[i] = work[2*(K - 1 + i)];
    }
}

// Compute y = X^T @ u where X is the L×K Hankel matrix
// X^T[j,i] = x[i+j], so y[j] = sum_i x[i+j] * u[i] = (reverse(x) ★ u)[K - 1 - j]
static void ssa_hankel_matvec_T(
    const SSA* ssa,
    const double* u,      // Input: length L
    double* y             // Output: length K
) {
    int K = ssa->K;
    int L = ssa->L;
    int n = ssa->fft_len;
    double* work = ssa->work1;
    
    // Pack u into complex format
    memset(work, 0, 2 * n * sizeof(double));
    for (int i = 0; i < L; i++) {
        work[2*i] = u[i];
    }
    
    // FFT of u
    ssa_fft_forward(work, n);
    
    // Pointwise multiply with precomputed FFT(reverse(x))
    ssa_complex_mul(ssa->fft_x_rev, work, work, n);
    
    // Inverse FFT
    ssa_fft_inverse(work, n);
    
    // Extract y = conv[L-1 : L-1+K]
    for (int j = 0; j < K; j++) {
        y[j] = work[2*(L - 1 + j)];
    }
}

// ----------------------------------------------------------------------------
// Core Implementation
// ----------------------------------------------------------------------------

int ssa_init(SSA* ssa, const double* x, int N, int L) {
    if (!ssa || !x || N < 4 || L < 2 || L > N - 1) {
        return -1;
    }
    
    memset(ssa, 0, sizeof(SSA));
    
    ssa->N = N;
    ssa->L = L;
    ssa->K = N - L + 1;
    
    // FFT length: next power of 2 >= N (for convolution)
    int fft_n = ssa_next_pow2(N);
    ssa->fft_len = fft_n;
    
    // Allocate FFT arrays (complex interleaved)
    ssa->fft_x = calloc(2 * fft_n, sizeof(double));
    ssa->fft_x_rev = calloc(2 * fft_n, sizeof(double));
    ssa->work1 = calloc(2 * fft_n, sizeof(double));
    ssa->work2 = calloc(2 * fft_n, sizeof(double));
    ssa->work3 = calloc(ssa_max(L, ssa->K), sizeof(double));
    
    if (!ssa->fft_x || !ssa->fft_x_rev || !ssa->work1 || !ssa->work2 || !ssa->work3) {
        ssa_free(ssa);
        return -1;
    }
    
    // Precompute FFT of x (zero-padded)
    for (int i = 0; i < N; i++) {
        ssa->fft_x[2*i] = x[i];
    }
    ssa_fft_forward(ssa->fft_x, fft_n);
    
    // Precompute FFT of reverse(x)
    for (int i = 0; i < N; i++) {
        ssa->fft_x_rev[2*i] = x[N - 1 - i];
    }
    ssa_fft_forward(ssa->fft_x_rev, fft_n);
    
    ssa->initialized = true;
    return 0;
}

int ssa_decompose(SSA* ssa, int k, int max_iter) {
    if (!ssa || !ssa->initialized || k < 1) {
        return -1;
    }
    
    int L = ssa->L;
    int K = ssa->K;
    
    // Limit k to maximum possible
    k = ssa_min(k, ssa_min(L, K));
    
    // Allocate results
    ssa->U = calloc(L * k, sizeof(double));
    ssa->V = calloc(K * k, sizeof(double));
    ssa->sigma = calloc(k, sizeof(double));
    ssa->eigenvalues = calloc(k, sizeof(double));
    
    if (!ssa->U || !ssa->V || !ssa->sigma || !ssa->eigenvalues) {
        return -1;
    }
    
    ssa->n_components = k;
    
    // Workspace for iteration
    double* u = malloc(L * sizeof(double));
    double* u_prev = malloc(L * sizeof(double));
    double* v = malloc(K * sizeof(double));
    double* v_new = malloc(K * sizeof(double));
    
    if (!u || !u_prev || !v || !v_new) {
        free(u); free(u_prev); free(v); free(v_new);
        return -1;
    }
    
    // Seed random (deterministic for reproducibility)
    unsigned int seed = 42;
    
    ssa->total_variance = 0.0;
    
    for (int comp = 0; comp < k; comp++) {
        // Initialize v randomly and orthogonalize against previous v's
        for (int i = 0; i < K; i++) {
            seed = seed * 1103515245 + 12345;
            v[i] = (double)((seed >> 16) & 0x7fff) / 32768.0 - 0.5;
        }
        
        // Initial orthogonalization
        for (int j = 0; j < comp; j++) {
            double* v_j = &ssa->V[j * K];
            double dot = ssa_dot(v, v_j, K);
            ssa_axpy(v, v_j, -dot, K);
        }
        ssa_normalize(v, K);
        
        double sigma_prev = 0.0;
        
        // Power iteration for singular vector
        for (int iter = 0; iter < max_iter; iter++) {
            // u = X @ v
            ssa_hankel_matvec(ssa, v, u);
            
            // Orthogonalize u against previous u's
            for (int j = 0; j < comp; j++) {
                double* u_j = &ssa->U[j * L];
                double dot = ssa_dot(u, u_j, L);
                ssa_axpy(u, u_j, -dot, L);
            }
            
            // v_new = X^T @ u  
            ssa_hankel_matvec_T(ssa, u, v_new);
            
            // Orthogonalize v_new against previous v's
            for (int j = 0; j < comp; j++) {
                double* v_j = &ssa->V[j * K];
                double dot = ssa_dot(v_new, v_j, K);
                ssa_axpy(v_new, v_j, -dot, K);
            }
            
            // Normalize and get sigma estimate
            double v_norm = ssa_normalize(v_new, K);
            double sigma = sqrt(v_norm);
            
            // Check convergence via vector difference
            double diff = 0.0;
            for (int i = 0; i < K; i++) {
                double d = fabs(v[i]) - fabs(v_new[i]);  // Sign-invariant
                diff += d * d;
            }
            diff = sqrt(diff);
            
            // Copy
            memcpy(v, v_new, K * sizeof(double));
            sigma_prev = sigma;
            
            if (diff < SSA_CONVERGENCE_TOL && iter > 10) {
                break;
            }
        }
        
        // Final orthogonalization of v
        for (int j = 0; j < comp; j++) {
            double* v_j = &ssa->V[j * K];
            double dot = ssa_dot(v, v_j, K);
            ssa_axpy(v, v_j, -dot, K);
        }
        ssa_normalize(v, K);
        
        // Compute final u and sigma
        ssa_hankel_matvec(ssa, v, u);
        
        // Orthogonalize u
        for (int j = 0; j < comp; j++) {
            double* u_j = &ssa->U[j * L];
            double dot = ssa_dot(u, u_j, L);
            ssa_axpy(u, u_j, -dot, L);
        }
        
        double sigma = ssa_normalize(u, L);
        
        // Store results
        memcpy(&ssa->U[comp * L], u, L * sizeof(double));
        memcpy(&ssa->V[comp * K], v, K * sizeof(double));
        ssa->sigma[comp] = sigma;
        ssa->eigenvalues[comp] = sigma * sigma;
        ssa->total_variance += sigma * sigma;
    }
    
    // Sort by descending singular value (bubble sort, k is small)
    for (int i = 0; i < k - 1; i++) {
        for (int j = i + 1; j < k; j++) {
            if (ssa->sigma[j] > ssa->sigma[i]) {
                // Swap sigma
                double tmp = ssa->sigma[i];
                ssa->sigma[i] = ssa->sigma[j];
                ssa->sigma[j] = tmp;
                
                // Swap eigenvalue
                tmp = ssa->eigenvalues[i];
                ssa->eigenvalues[i] = ssa->eigenvalues[j];
                ssa->eigenvalues[j] = tmp;
                
                // Swap U columns
                for (int t = 0; t < L; t++) {
                    tmp = ssa->U[i * L + t];
                    ssa->U[i * L + t] = ssa->U[j * L + t];
                    ssa->U[j * L + t] = tmp;
                }
                
                // Swap V columns
                for (int t = 0; t < K; t++) {
                    tmp = ssa->V[i * K + t];
                    ssa->V[i * K + t] = ssa->V[j * K + t];
                    ssa->V[j * K + t] = tmp;
                }
            }
        }
    }
    
    // Fix sign convention: make first element of V positive
    // This ensures reproducible signs across runs
    for (int i = 0; i < k; i++) {
        // Compute sum of U to determine sign
        double sum_u = 0.0;
        for (int t = 0; t < L; t++) {
            sum_u += ssa->U[i * L + t];
        }
        
        // If sum is negative, flip both U and V
        if (sum_u < 0) {
            for (int t = 0; t < L; t++) {
                ssa->U[i * L + t] = -ssa->U[i * L + t];
            }
            for (int t = 0; t < K; t++) {
                ssa->V[i * K + t] = -ssa->V[i * K + t];
            }
        }
    }
    
    free(u);
    free(u_prev);
    free(v);
    free(v_new);
    
    ssa->decomposed = true;
    return 0;
}

int ssa_reconstruct(const SSA* ssa, const int* group, int n_group, 
                    double* output) {
    if (!ssa || !ssa->decomposed || !group || !output || n_group < 1) {
        return -1;
    }
    
    int N = ssa->N;
    int L = ssa->L;
    int K = ssa->K;
    int fft_n = ssa->fft_len;
    
    memset(output, 0, N * sizeof(double));
    
    double* work = ssa->work2;
    
    for (int g = 0; g < n_group; g++) {
        int idx = group[g];
        if (idx < 0 || idx >= ssa->n_components) {
            continue;
        }
        
        double sigma = ssa->sigma[idx];
        const double* u = &ssa->U[idx * L];
        const double* v = &ssa->V[idx * K];
        
        // Reconstruction via convolution:
        // The sum along antidiagonals of (sigma * u * v^T) is conv(u, v)
        
        // Pack u into complex format
        memset(work, 0, 2 * fft_n * sizeof(double));
        for (int i = 0; i < L; i++) {
            work[2*i] = u[i];
        }
        ssa_fft_forward(work, fft_n);
        
        // Pack v and FFT
        double* work_v = ssa->work1;
        memset(work_v, 0, 2 * fft_n * sizeof(double));
        for (int i = 0; i < K; i++) {
            work_v[2*i] = v[i];
        }
        ssa_fft_forward(work_v, fft_n);
        
        // Multiply
        ssa_complex_mul(work, work_v, work, fft_n);
        
        // Inverse FFT
        ssa_fft_inverse(work, fft_n);
        
        // Accumulate (conv result is in real parts)
        for (int t = 0; t < N; t++) {
            output[t] += sigma * work[2*t];
        }
    }
    
    // Hankel averaging: divide by count of elements per antidiagonal
    // For position t, count = min(t+1, L, K, N-t)
    for (int t = 0; t < N; t++) {
        int count = ssa_min(ssa_min(t + 1, L), ssa_min(K, N - t));
        if (count > 0) {
            output[t] /= count;
        }
    }
    
    return 0;
}

void ssa_free(SSA* ssa) {
    if (!ssa) return;
    
    free(ssa->fft_x);
    free(ssa->fft_x_rev);
    free(ssa->work1);
    free(ssa->work2);
    free(ssa->work3);
    free(ssa->U);
    free(ssa->V);
    free(ssa->sigma);
    free(ssa->eigenvalues);
    
    memset(ssa, 0, sizeof(SSA));
}

// ----------------------------------------------------------------------------
// Convenience Functions
// ----------------------------------------------------------------------------

int ssa_get_trend(const SSA* ssa, double* output) {
    int group[] = {0};
    return ssa_reconstruct(ssa, group, 1, output);
}

int ssa_get_noise(const SSA* ssa, int noise_start, double* output) {
    if (!ssa || !ssa->decomposed || noise_start < 0) {
        return -1;
    }
    
    int n_noise = ssa->n_components - noise_start;
    if (n_noise <= 0) {
        memset(output, 0, ssa->N * sizeof(double));
        return 0;
    }
    
    int* group = malloc(n_noise * sizeof(int));
    if (!group) return -1;
    
    for (int i = 0; i < n_noise; i++) {
        group[i] = noise_start + i;
    }
    
    int ret = ssa_reconstruct(ssa, group, n_noise, output);
    free(group);
    return ret;
}

int ssa_get_oscillatory(const SSA* ssa, int start, int end, double* output) {
    if (!ssa || !ssa->decomposed || start < 0 || end < start) {
        return -1;
    }
    
    end = ssa_min(end, ssa->n_components - 1);
    int n = end - start + 1;
    
    int* group = malloc(n * sizeof(int));
    if (!group) return -1;
    
    for (int i = 0; i < n; i++) {
        group[i] = start + i;
    }
    
    int ret = ssa_reconstruct(ssa, group, n, output);
    free(group);
    return ret;
}

double ssa_variance_explained(const SSA* ssa, int start, int end) {
    if (!ssa || !ssa->decomposed || start < 0 || ssa->total_variance <= 0) {
        return 0.0;
    }
    
    if (end < 0 || end >= ssa->n_components) {
        end = ssa->n_components - 1;
    }
    
    double sum = 0.0;
    for (int i = start; i <= end; i++) {
        sum += ssa->eigenvalues[i];
    }
    
    return sum / ssa->total_variance;
}

int ssa_find_pairs(const SSA* ssa, int* pairs, int max_pairs, double tolerance) {
    if (!ssa || !ssa->decomposed || !pairs || max_pairs < 1) {
        return 0;
    }
    
    int n_pairs = 0;
    int* used = calloc(ssa->n_components, sizeof(int));
    if (!used) return 0;
    
    for (int i = 0; i < ssa->n_components - 1 && n_pairs < max_pairs; i++) {
        if (used[i]) continue;
        
        double ev_i = ssa->eigenvalues[i];
        
        // Look for pair
        for (int j = i + 1; j < ssa->n_components && n_pairs < max_pairs; j++) {
            if (used[j]) continue;
            
            double ev_j = ssa->eigenvalues[j];
            double rel_diff = fabs(ev_i - ev_j) / (ev_i + 1e-15);
            
            if (rel_diff < tolerance) {
                pairs[2 * n_pairs] = i;
                pairs[2 * n_pairs + 1] = j;
                used[i] = used[j] = 1;
                n_pairs++;
                break;
            }
        }
    }
    
    free(used);
    return n_pairs;
}

int ssa_find_optimal_L(const double* x, int N, int L_min, int L_max) {
    if (!x || N < 10 || L_min < 2 || L_max > N - 2 || L_min > L_max) {
        return N / 4;  // Default
    }
    
    double best_score = -1e30;
    int best_L = (L_min + L_max) / 2;
    
    // Test a few values
    int step = ssa_max(1, (L_max - L_min) / 10);
    
    for (int L = L_min; L <= L_max; L += step) {
        SSA ssa;
        if (ssa_init(&ssa, x, N, L) != 0) continue;
        if (ssa_decompose(&ssa, 10, 50) != 0) {
            ssa_free(&ssa);
            continue;
        }
        
        // Score: sum of gaps between consecutive eigenvalues
        // Larger gaps = better separability
        double score = 0.0;
        for (int i = 0; i < ssa.n_components - 1; i++) {
            double gap = ssa.eigenvalues[i] - ssa.eigenvalues[i + 1];
            score += gap * gap;
        }
        
        if (score > best_score) {
            best_score = score;
            best_L = L;
        }
        
        ssa_free(&ssa);
    }
    
    return best_L;
}

// ----------------------------------------------------------------------------
// Advanced: W-Correlation
// ----------------------------------------------------------------------------

int ssa_wcorrelation(const SSA* ssa, double* wcorr, int k) {
    if (!ssa || !ssa->decomposed || !wcorr || k < 1) {
        return -1;
    }
    
    k = ssa_min(k, ssa->n_components);
    int N = ssa->N;
    
    // Compute weights for each position
    double* weights = malloc(N * sizeof(double));
    if (!weights) return -1;
    
    int L = ssa->L;
    int K = ssa->K;
    for (int t = 0; t < N; t++) {
        weights[t] = (double)ssa_min(ssa_min(t + 1, L), ssa_min(K, N - t));
    }
    
    // Reconstruct each component
    double* reconstructions = malloc(k * N * sizeof(double));
    if (!reconstructions) {
        free(weights);
        return -1;
    }
    
    for (int i = 0; i < k; i++) {
        int group[] = {i};
        ssa_reconstruct(ssa, group, 1, &reconstructions[i * N]);
    }
    
    // Compute weighted correlations
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < k; j++) {
            double* ri = &reconstructions[i * N];
            double* rj = &reconstructions[j * N];
            
            // Weighted inner product
            double sum_wij = 0.0, sum_wii = 0.0, sum_wjj = 0.0;
            for (int t = 0; t < N; t++) {
                double w = weights[t];
                sum_wij += w * ri[t] * rj[t];
                sum_wii += w * ri[t] * ri[t];
                sum_wjj += w * rj[t] * rj[t];
            }
            
            double denom = sqrt(sum_wii * sum_wjj);
            wcorr[i * k + j] = (denom > 1e-15) ? sum_wij / denom : 0.0;
        }
    }
    
    free(weights);
    free(reconstructions);
    return 0;
}

// ----------------------------------------------------------------------------
// Advanced: Forecasting via Linear Recurrence
// ----------------------------------------------------------------------------

int ssa_forecast(const SSA* ssa, const int* group, int n_group,
                 int horizon, double* forecast) {
    if (!ssa || !ssa->decomposed || !group || !forecast || 
        n_group < 1 || horizon < 1) {
        return -1;
    }
    
    int N = ssa->N;
    int L = ssa->L;
    
    // Compute linear recurrence coefficients
    // R = sum over group of (u_last / (1 - u_last^2)) * u[0:L-1]
    double* R = calloc(L - 1, sizeof(double));
    if (!R) return -1;
    
    double nu_sq = 0.0;  // sum of u_last^2
    
    for (int g = 0; g < n_group; g++) {
        int idx = group[g];
        if (idx < 0 || idx >= ssa->n_components) continue;
        
        const double* u = &ssa->U[idx * L];
        double u_last = u[L - 1];
        nu_sq += u_last * u_last;
        
        for (int i = 0; i < L - 1; i++) {
            R[i] += u_last * u[i];
        }
    }
    
    if (nu_sq >= 1.0 - 1e-10) {
        // Recurrence is unstable
        free(R);
        return -1;
    }
    
    double scale = 1.0 / (1.0 - nu_sq);
    for (int i = 0; i < L - 1; i++) {
        R[i] *= scale;
    }
    
    // Reconstruct the series from group
    double* reconstructed = malloc((N + horizon) * sizeof(double));
    if (!reconstructed) {
        free(R);
        return -1;
    }
    
    ssa_reconstruct(ssa, group, n_group, reconstructed);
    
    // Forecast using linear recurrence
    for (int h = 0; h < horizon; h++) {
        double val = 0.0;
        for (int i = 0; i < L - 1; i++) {
            val += R[i] * reconstructed[N + h - L + 1 + i];
        }
        reconstructed[N + h] = val;
        forecast[h] = val;
    }
    
    free(R);
    free(reconstructed);
    return 0;
}

#endif // SSA_IMPLEMENTATION

#ifdef __cplusplus
}
#endif

#endif // SSA_H
