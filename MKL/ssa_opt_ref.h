/*
 * ============================================================================
 * SSA-OPT Reference Implementation: Self-Contained Singular Spectrum Analysis
 * ============================================================================
 *
 * PURPOSE:
 *   This is a reference implementation for accuracy testing and education.
 *   It has ZERO external dependencies - everything is self-contained.
 *   Use ssa_opt_mkl.h for production performance.
 *
 * WHAT IS SSA?
 *   Singular Spectrum Analysis decomposes a time series into trend, periodic
 *   components, and noise by embedding it into a Hankel matrix and computing
 *   its SVD. This implementation avoids forming the explicit matrix.
 *
 * ALGORITHM:
 *   - O(N log N) Hankel matvec via FFT convolution (vs O(N²) naive)
 *   - Sequential power iteration for SVD computation
 *   - Built-in Cooley-Tukey radix-2 FFT
 *
 * USAGE:
 *   #define SSA_OPT_REF_IMPLEMENTATION
 *   #include "ssa_opt_ref.h"
 *
 *   SSA_Opt_Ref ssa = {0};
 *   ssa_opt_ref_init(&ssa, signal, N, L);
 *   ssa_opt_ref_decompose(&ssa, k, max_iter);
 *   ssa_opt_ref_reconstruct(&ssa, components, n_components, output);
 *   ssa_opt_ref_free(&ssa);
 *
 * BUILD:
 *   gcc -O2 -o test test.c -lm
 *
 * ============================================================================
 */

#ifndef SSA_OPT_REF_H
#define SSA_OPT_REF_H

#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef __cplusplus
extern "C"
{
#endif

// ============================================================================
// Configuration
// ============================================================================

#ifndef SSA_REF_CONVERGENCE_TOL
#define SSA_REF_CONVERGENCE_TOL 1e-12
#endif

#ifndef SSA_REF_ALIGN
#define SSA_REF_ALIGN 64
#endif

// ============================================================================
// Data Structures
// ============================================================================

/**
 * @brief SSA context for reference implementation.
 *
 * MEMORY LAYOUT:
 *   - All matrices are column-major (for compatibility with MKL version)
 *   - U is L × k: column i is the i-th left singular vector
 *   - V is K × k: column i is the i-th right singular vector
 *   - Complex arrays are interleaved: [re₀, im₀, re₁, im₁, ...]
 */
typedef struct
{
    // Dimensions
    int N;       // Original series length
    int L;       // Window length (embedding dimension)
    int K;       // K = N - L + 1
    int fft_len; // FFT length (next power of 2)

    // Precomputed data
    double *fft_x;          // FFT of input signal, interleaved complex
    double *inv_diag_count; // Precomputed 1/count for diagonal averaging

    // Workspace buffers
    double *ws_fft1; // FFT scratch buffer 1
    double *ws_fft2; // FFT scratch buffer 2
    double *ws_u;    // Vector scratch for iteration
    double *ws_v;    // Vector scratch for iteration

    // Results
    double *U;           // Left singular vectors, L × k, column-major
    double *V;           // Right singular vectors, K × k, column-major
    double *sigma;       // Singular values
    double *eigenvalues; // Squared singular values
    int n_components;    // Number of computed components

    // State
    bool initialized;
    bool decomposed;
    double total_variance;
    uint32_t rng_state; // Simple LCG state
} SSA_Opt_Ref;

// ============================================================================
// Public API
// ============================================================================

/**
 * @brief Initialize SSA context with input signal.
 * @param ssa   Pointer to zero-initialized SSA_Opt_Ref struct
 * @param x     Input time series, length N
 * @param N     Length of input signal
 * @param L     Window length (embedding dimension), typically N/3 to N/2
 * @return      0 on success, -1 on error
 */
int ssa_opt_ref_init(SSA_Opt_Ref *ssa, const double *x, int N, int L);

/**
 * @brief Compute SVD via sequential power iteration.
 * @param ssa       Initialized SSA context
 * @param k         Number of singular triplets to compute
 * @param max_iter  Maximum iterations per component (100-200 typical)
 * @return          0 on success, -1 on error
 */
int ssa_opt_ref_decompose(SSA_Opt_Ref *ssa, int k, int max_iter);

/**
 * @brief Reconstruct signal from selected components.
 * @param ssa      Decomposed SSA context
 * @param group    Array of component indices to include (0-based)
 * @param n_group  Number of components in group
 * @param output   Output buffer, length N
 * @return         0 on success, -1 on error
 */
int ssa_opt_ref_reconstruct(const SSA_Opt_Ref *ssa, const int *group, int n_group, double *output);

/**
 * @brief Free all memory associated with SSA context.
 */
void ssa_opt_ref_free(SSA_Opt_Ref *ssa);

// ============================================================================
// Analysis API
// ============================================================================

/**
 * @brief Compute W-correlation matrix between SSA components.
 */
int ssa_opt_ref_wcorr_matrix(const SSA_Opt_Ref *ssa, double *W);

/**
 * @brief Compute W-correlation between two specific components.
 */
double ssa_opt_ref_wcorr_pair(const SSA_Opt_Ref *ssa, int i, int j);

/**
 * @brief Get explained variance ratio for component range [start, end].
 */
double ssa_opt_ref_variance_explained(const SSA_Opt_Ref *ssa, int start, int end);

/**
 * @brief Reconstruct trend (component 0).
 */
int ssa_opt_ref_get_trend(const SSA_Opt_Ref *ssa, double *output);

/**
 * @brief Reconstruct noise (components from noise_start to end).
 */
int ssa_opt_ref_get_noise(const SSA_Opt_Ref *ssa, int noise_start, double *output);

// ============================================================================
// Implementation
// ============================================================================

#ifdef SSA_OPT_REF_IMPLEMENTATION

// ----------------------------------------------------------------------------
// Helpers
// ----------------------------------------------------------------------------

static inline int ssa_ref_next_pow2(int n)
{
    int p = 1;
    while (p < n)
        p <<= 1;
    return p;
}

static inline int ssa_ref_min(int a, int b) { return a < b ? a : b; }
static inline int ssa_ref_max(int a, int b) { return a > b ? a : b; }

static inline void *ssa_ref_alloc(size_t size)
{
    size_t aligned_size = (size + SSA_REF_ALIGN - 1) & ~(size_t)(SSA_REF_ALIGN - 1);
#if defined(_MSC_VER)
    return _aligned_malloc(aligned_size, SSA_REF_ALIGN);
#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
    return aligned_alloc(SSA_REF_ALIGN, aligned_size);
#else
    void *ptr = NULL;
    posix_memalign(&ptr, SSA_REF_ALIGN, aligned_size);
    return ptr;
#endif
}

static inline void ssa_ref_free_ptr(void *ptr)
{
#if defined(_MSC_VER)
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

// Simple LCG random number generator
static inline double ssa_ref_rand(uint32_t *state)
{
    *state = (*state) * 1103515245u + 12345u;
    return (double)((*state >> 16) & 0x7fff) / 32768.0 - 0.5;
}

// ----------------------------------------------------------------------------
// Vector Operations (all scalar, no dependencies)
// ----------------------------------------------------------------------------

static inline double ssa_ref_dot(const double *a, const double *b, int n)
{
    double sum = 0.0;
    for (int i = 0; i < n; i++)
        sum += a[i] * b[i];
    return sum;
}

static inline double ssa_ref_nrm2(const double *v, int n)
{
    return sqrt(ssa_ref_dot(v, v, n));
}

static inline void ssa_ref_scal(double *v, int n, double s)
{
    for (int i = 0; i < n; i++)
        v[i] *= s;
}

static inline void ssa_ref_axpy(double *y, const double *x, double a, int n)
{
    for (int i = 0; i < n; i++)
        y[i] += a * x[i];
}

static inline void ssa_ref_copy(const double *src, double *dst, int n)
{
    memcpy(dst, src, n * sizeof(double));
}

static inline double ssa_ref_normalize(double *v, int n)
{
    double norm = ssa_ref_nrm2(v, n);
    if (norm > 1e-15)
    {
        double inv = 1.0 / norm;
        ssa_ref_scal(v, n, inv);
    }
    return norm;
}

static inline void ssa_ref_zero(double *v, int n)
{
    memset(v, 0, n * sizeof(double));
}

// ----------------------------------------------------------------------------
// Complex Operations
// ----------------------------------------------------------------------------

// Complex multiply: c = a * b (interleaved format)
static void ssa_ref_complex_mul(const double *a, const double *b, double *c, int n)
{
    for (int i = 0; i < n; i++)
    {
        double ar = a[2 * i], ai = a[2 * i + 1];
        double br = b[2 * i], bi = b[2 * i + 1];
        c[2 * i] = ar * br - ai * bi;
        c[2 * i + 1] = ar * bi + ai * br;
    }
}

// ----------------------------------------------------------------------------
// Built-in Cooley-Tukey FFT
// ----------------------------------------------------------------------------

static void ssa_ref_fft(double *data, int n, int sign)
{
    // Bit-reversal permutation
    int j = 0;
    for (int i = 0; i < n - 1; i++)
    {
        if (i < j)
        {
            double tr = data[2 * j], ti = data[2 * j + 1];
            data[2 * j] = data[2 * i];
            data[2 * j + 1] = data[2 * i + 1];
            data[2 * i] = tr;
            data[2 * i + 1] = ti;
        }
        int k = n >> 1;
        while (k <= j)
        {
            j -= k;
            k >>= 1;
        }
        j += k;
    }

    // Cooley-Tukey butterfly
    for (int len = 2; len <= n; len <<= 1)
    {
        double ang = sign * 2.0 * 3.14159265358979323846 / len;
        double wpr = cos(ang), wpi = sin(ang);

        for (int i = 0; i < n; i += len)
        {
            double wr = 1.0, wi = 0.0;

            for (int jj = 0; jj < len / 2; jj++)
            {
                int a = i + jj;
                int b = a + len / 2;

                double tr = wr * data[2 * b] - wi * data[2 * b + 1];
                double ti = wr * data[2 * b + 1] + wi * data[2 * b];

                data[2 * b] = data[2 * a] - tr;
                data[2 * b + 1] = data[2 * a + 1] - ti;
                data[2 * a] += tr;
                data[2 * a + 1] += ti;

                double wt = wr;
                wr = wr * wpr - wi * wpi;
                wi = wi * wpr + wt * wpi;
            }
        }
    }

    // Scale for inverse transform
    if (sign == 1)
    {
        double scale = 1.0 / n;
        for (int i = 0; i < 2 * n; i++)
            data[i] *= scale;
    }
}

// Real to complex FFT (zero-pads input, returns interleaved complex)
static void ssa_ref_fft_r2c(int fft_len, const double *input, int input_len, double *output)
{
    ssa_ref_zero(output, 2 * fft_len);
    for (int i = 0; i < input_len; i++)
    {
        output[2 * i] = input[i];
    }
    ssa_ref_fft(output, fft_len, -1);
}

// ----------------------------------------------------------------------------
// Hankel Matrix-Vector Products
// ----------------------------------------------------------------------------

// y = H @ v  where H[i,j] = x[i+j]
static void ssa_ref_hankel_matvec(SSA_Opt_Ref *ssa, const double *v, double *y)
{
    int K = ssa->K;
    int L = ssa->L;
    int n = ssa->fft_len;
    double *ws = ssa->ws_fft1;

    // Pack reversed v into complex format
    ssa_ref_zero(ws, 2 * n);
    for (int i = 0; i < K; i++)
    {
        ws[2 * i] = v[K - 1 - i];
    }

    // FFT of reversed v
    ssa_ref_fft(ws, n, -1);

    // Pointwise multiply with precomputed FFT(x)
    ssa_ref_complex_mul(ssa->fft_x, ws, ws, n);

    // Inverse FFT
    ssa_ref_fft(ws, n, 1);

    // Extract result: conv[K-1 : K-1+L]
    for (int i = 0; i < L; i++)
    {
        y[i] = ws[2 * (K - 1 + i)];
    }
}

// y = H^T @ u
static void ssa_ref_hankel_matvec_T(SSA_Opt_Ref *ssa, const double *u, double *y)
{
    int K = ssa->K;
    int L = ssa->L;
    int n = ssa->fft_len;
    double *ws = ssa->ws_fft1;

    // Pack reversed u into complex format
    ssa_ref_zero(ws, 2 * n);
    for (int i = 0; i < L; i++)
    {
        ws[2 * i] = u[L - 1 - i];
    }

    // FFT of reversed u
    ssa_ref_fft(ws, n, -1);

    // Pointwise multiply with precomputed FFT(x)
    ssa_ref_complex_mul(ssa->fft_x, ws, ws, n);

    // Inverse FFT
    ssa_ref_fft(ws, n, 1);

    // Extract result: conv[L-1 : L-1+K]
    for (int j = 0; j < K; j++)
    {
        y[j] = ws[2 * (L - 1 + j)];
    }
}

// ----------------------------------------------------------------------------
// Core API Implementation
// ----------------------------------------------------------------------------

int ssa_opt_ref_init(SSA_Opt_Ref *ssa, const double *x, int N, int L)
{
    if (!ssa || !x || N < 4 || L < 2 || L > N - 1)
    {
        return -1;
    }

    memset(ssa, 0, sizeof(SSA_Opt_Ref));

    ssa->N = N;
    ssa->L = L;
    ssa->K = N - L + 1;

    int conv_len = N + ssa->K - 1;
    int fft_n = ssa_ref_next_pow2(conv_len);
    ssa->fft_len = fft_n;

    // Initialize RNG
    ssa->rng_state = 42;

    // Allocate workspace
    size_t fft_size = 2 * fft_n * sizeof(double);
    size_t vec_size = ssa_ref_max(L, ssa->K) * sizeof(double);

    ssa->fft_x = (double *)ssa_ref_alloc(fft_size);
    ssa->ws_fft1 = (double *)ssa_ref_alloc(fft_size);
    ssa->ws_fft2 = (double *)ssa_ref_alloc(fft_size);
    ssa->ws_u = (double *)ssa_ref_alloc(vec_size);
    ssa->ws_v = (double *)ssa_ref_alloc(vec_size);

    if (!ssa->fft_x || !ssa->ws_fft1 || !ssa->ws_fft2 || !ssa->ws_u || !ssa->ws_v)
    {
        ssa_opt_ref_free(ssa);
        return -1;
    }

    // Precompute FFT(x)
    ssa_ref_fft_r2c(fft_n, x, N, ssa->fft_x);

    // Precompute inverse diagonal counts for reconstruction
    ssa->inv_diag_count = (double *)ssa_ref_alloc(N * sizeof(double));
    if (!ssa->inv_diag_count)
    {
        ssa_opt_ref_free(ssa);
        return -1;
    }

    for (int t = 0; t < N; t++)
    {
        int count = ssa_ref_min(ssa_ref_min(t + 1, L), ssa_ref_min(ssa->K, N - t));
        ssa->inv_diag_count[t] = (count > 0) ? 1.0 / count : 0.0;
    }

    ssa->initialized = true;
    return 0;
}

int ssa_opt_ref_decompose(SSA_Opt_Ref *ssa, int k, int max_iter)
{
    if (!ssa || !ssa->initialized || k < 1)
    {
        return -1;
    }

    int L = ssa->L;
    int K = ssa->K;

    k = ssa_ref_min(k, ssa_ref_min(L, K));

    // Allocate results
    ssa->U = (double *)ssa_ref_alloc(L * k * sizeof(double));
    ssa->V = (double *)ssa_ref_alloc(K * k * sizeof(double));
    ssa->sigma = (double *)ssa_ref_alloc(k * sizeof(double));
    ssa->eigenvalues = (double *)ssa_ref_alloc(k * sizeof(double));

    if (!ssa->U || !ssa->V || !ssa->sigma || !ssa->eigenvalues)
    {
        return -1;
    }

    ssa->n_components = k;

    double *u = ssa->ws_u;
    double *v = ssa->ws_v;
    double *v_new = (double *)ssa_ref_alloc(K * sizeof(double));

    if (!v_new)
    {
        return -1;
    }

    ssa->total_variance = 0.0;

    // Sequential power iteration for each component
    for (int comp = 0; comp < k; comp++)
    {
        // Random initialization
        for (int i = 0; i < K; i++)
        {
            v[i] = ssa_ref_rand(&ssa->rng_state);
        }

        // Orthogonalize against previous V's
        for (int j = 0; j < comp; j++)
        {
            double *v_j = &ssa->V[j * K];
            double dot = ssa_ref_dot(v, v_j, K);
            ssa_ref_axpy(v, v_j, -dot, K);
        }
        ssa_ref_normalize(v, K);

        // Power iteration
        for (int iter = 0; iter < max_iter; iter++)
        {
            // u = H @ v
            ssa_ref_hankel_matvec(ssa, v, u);

            // Orthogonalize u against previous U's
            for (int j = 0; j < comp; j++)
            {
                double *u_j = &ssa->U[j * L];
                double dot = ssa_ref_dot(u, u_j, L);
                ssa_ref_axpy(u, u_j, -dot, L);
            }

            // v_new = H^T @ u
            ssa_ref_hankel_matvec_T(ssa, u, v_new);

            // Orthogonalize v_new against previous V's
            for (int j = 0; j < comp; j++)
            {
                double *v_j = &ssa->V[j * K];
                double dot = ssa_ref_dot(v_new, v_j, K);
                ssa_ref_axpy(v_new, v_j, -dot, K);
            }

            ssa_ref_normalize(v_new, K);

            // Convergence check
            double diff = 0.0;
            for (int i = 0; i < K; i++)
            {
                double d = fabs(v[i]) - fabs(v_new[i]);
                diff += d * d;
            }

            ssa_ref_copy(v_new, v, K);

            if (sqrt(diff) < SSA_REF_CONVERGENCE_TOL && iter > 10)
                break;
        }

        // Final orthogonalization
        for (int j = 0; j < comp; j++)
        {
            double *v_j = &ssa->V[j * K];
            double dot = ssa_ref_dot(v, v_j, K);
            ssa_ref_axpy(v, v_j, -dot, K);
        }
        ssa_ref_normalize(v, K);

        // Compute final u and sigma
        ssa_ref_hankel_matvec(ssa, v, u);

        for (int j = 0; j < comp; j++)
        {
            double *u_j = &ssa->U[j * L];
            double dot = ssa_ref_dot(u, u_j, L);
            ssa_ref_axpy(u, u_j, -dot, L);
        }

        double sigma = ssa_ref_normalize(u, L);

        // Recompute v for SVD consistency: v = H^T @ u / sigma
        ssa_ref_hankel_matvec_T(ssa, u, v);

        for (int j = 0; j < comp; j++)
        {
            double *v_j = &ssa->V[j * K];
            double dot = ssa_ref_dot(v, v_j, K);
            ssa_ref_axpy(v, v_j, -dot, K);
        }

        if (sigma > 1e-15)
        {
            ssa_ref_scal(v, K, 1.0 / sigma);
        }

        // Store results
        ssa_ref_copy(u, &ssa->U[comp * L], L);
        ssa_ref_copy(v, &ssa->V[comp * K], K);
        ssa->sigma[comp] = sigma;
        ssa->eigenvalues[comp] = sigma * sigma;
        ssa->total_variance += sigma * sigma;
    }

    // Sort by descending singular value
    for (int i = 0; i < k - 1; i++)
    {
        for (int j = i + 1; j < k; j++)
        {
            if (ssa->sigma[j] > ssa->sigma[i])
            {
                // Swap scalars
                double tmp = ssa->sigma[i];
                ssa->sigma[i] = ssa->sigma[j];
                ssa->sigma[j] = tmp;

                tmp = ssa->eigenvalues[i];
                ssa->eigenvalues[i] = ssa->eigenvalues[j];
                ssa->eigenvalues[j] = tmp;

                // Swap vectors
                for (int t = 0; t < L; t++)
                {
                    tmp = ssa->U[i * L + t];
                    ssa->U[i * L + t] = ssa->U[j * L + t];
                    ssa->U[j * L + t] = tmp;
                }
                for (int t = 0; t < K; t++)
                {
                    tmp = ssa->V[i * K + t];
                    ssa->V[i * K + t] = ssa->V[j * K + t];
                    ssa->V[j * K + t] = tmp;
                }
            }
        }
    }

    // Fix sign convention: make sum(U) positive
    for (int i = 0; i < k; i++)
    {
        double sum = 0;
        for (int t = 0; t < L; t++)
            sum += ssa->U[i * L + t];
        if (sum < 0)
        {
            ssa_ref_scal(&ssa->U[i * L], L, -1.0);
            ssa_ref_scal(&ssa->V[i * K], K, -1.0);
        }
    }

    ssa_ref_free_ptr(v_new);

    ssa->decomposed = true;
    return 0;
}

int ssa_opt_ref_reconstruct(const SSA_Opt_Ref *ssa, const int *group, int n_group, double *output)
{
    if (!ssa || !ssa->decomposed || !group || !output || n_group < 1)
    {
        return -1;
    }

    int N = ssa->N;
    int L = ssa->L;
    int K = ssa->K;
    int fft_n = ssa->fft_len;

    ssa_ref_zero(output, N);

    // Use workspace buffers (cast away const for workspace access)
    SSA_Opt_Ref *ssa_mut = (SSA_Opt_Ref *)ssa;
    double *ws1 = ssa_mut->ws_fft1;
    double *ws2 = ssa_mut->ws_fft2;

    for (int g = 0; g < n_group; g++)
    {
        int idx = group[g];
        if (idx < 0 || idx >= ssa->n_components)
            continue;

        double sigma = ssa->sigma[idx];
        const double *u_vec = &ssa->U[idx * L];
        const double *v_vec = &ssa->V[idx * K];

        // FFT of σ·u
        ssa_ref_zero(ws1, 2 * fft_n);
        for (int i = 0; i < L; i++)
            ws1[2 * i] = sigma * u_vec[i];
        ssa_ref_fft(ws1, fft_n, -1);

        // FFT of v
        ssa_ref_zero(ws2, 2 * fft_n);
        for (int i = 0; i < K; i++)
            ws2[2 * i] = v_vec[i];
        ssa_ref_fft(ws2, fft_n, -1);

        // Pointwise multiply
        ssa_ref_complex_mul(ws1, ws2, ws1, fft_n);

        // Inverse FFT
        ssa_ref_fft(ws1, fft_n, 1);

        // Accumulate real parts
        for (int t = 0; t < N; t++)
        {
            output[t] += ws1[2 * t];
        }
    }

    // Diagonal averaging
    for (int t = 0; t < N; t++)
    {
        output[t] *= ssa->inv_diag_count[t];
    }

    return 0;
}

void ssa_opt_ref_free(SSA_Opt_Ref *ssa)
{
    if (!ssa)
        return;

    ssa_ref_free_ptr(ssa->fft_x);
    ssa_ref_free_ptr(ssa->ws_fft1);
    ssa_ref_free_ptr(ssa->ws_fft2);
    ssa_ref_free_ptr(ssa->ws_u);
    ssa_ref_free_ptr(ssa->ws_v);
    ssa_ref_free_ptr(ssa->inv_diag_count);
    ssa_ref_free_ptr(ssa->U);
    ssa_ref_free_ptr(ssa->V);
    ssa_ref_free_ptr(ssa->sigma);
    ssa_ref_free_ptr(ssa->eigenvalues);

    memset(ssa, 0, sizeof(SSA_Opt_Ref));
}

// ----------------------------------------------------------------------------
// Analysis Functions
// ----------------------------------------------------------------------------

int ssa_opt_ref_wcorr_matrix(const SSA_Opt_Ref *ssa, double *W)
{
    if (!ssa || !ssa->decomposed || !W)
        return -1;

    int N = ssa->N;
    int L = ssa->L;
    int K = ssa->K;
    int n = ssa->n_components;

    // Compute weights
    double *weights = (double *)ssa_ref_alloc(N * sizeof(double));
    if (!weights)
        return -1;

    for (int t = 0; t < N; t++)
    {
        weights[t] = (double)ssa_ref_min(ssa_ref_min(t + 1, L),
                                         ssa_ref_min(K, N - t));
    }

    // Reconstruct all components
    double *reconstructed = (double *)ssa_ref_alloc(N * n * sizeof(double));
    if (!reconstructed)
    {
        ssa_ref_free_ptr(weights);
        return -1;
    }

    for (int i = 0; i < n; i++)
    {
        int group[] = {i};
        ssa_opt_ref_reconstruct(ssa, group, 1, &reconstructed[i * N]);
    }

    // Compute weighted norms
    double *norms = (double *)ssa_ref_alloc(n * sizeof(double));
    if (!norms)
    {
        ssa_ref_free_ptr(weights);
        ssa_ref_free_ptr(reconstructed);
        return -1;
    }

    for (int i = 0; i < n; i++)
    {
        double *x_i = &reconstructed[i * N];
        double sum = 0.0;
        for (int t = 0; t < N; t++)
        {
            sum += weights[t] * x_i[t] * x_i[t];
        }
        norms[i] = sqrt(sum);
    }

    // Compute W-correlation matrix
    for (int i = 0; i < n; i++)
    {
        double *x_i = &reconstructed[i * N];

        for (int j = i; j < n; j++)
        {
            double *x_j = &reconstructed[j * N];

            double inner = 0.0;
            for (int t = 0; t < N; t++)
            {
                inner += weights[t] * x_i[t] * x_j[t];
            }

            double denom = norms[i] * norms[j];
            double corr = (denom > 1e-15) ? inner / denom : 0.0;

            W[i * n + j] = corr;
            W[j * n + i] = corr;
        }
    }

    ssa_ref_free_ptr(weights);
    ssa_ref_free_ptr(reconstructed);
    ssa_ref_free_ptr(norms);

    return 0;
}

double ssa_opt_ref_wcorr_pair(const SSA_Opt_Ref *ssa, int i, int j)
{
    if (!ssa || !ssa->decomposed ||
        i < 0 || i >= ssa->n_components ||
        j < 0 || j >= ssa->n_components)
        return 0.0;

    int N = ssa->N;
    int L = ssa->L;
    int K = ssa->K;

    double *x_i = (double *)ssa_ref_alloc(N * sizeof(double));
    double *x_j = (double *)ssa_ref_alloc(N * sizeof(double));

    if (!x_i || !x_j)
    {
        ssa_ref_free_ptr(x_i);
        ssa_ref_free_ptr(x_j);
        return 0.0;
    }

    int group_i[] = {i};
    int group_j[] = {j};
    ssa_opt_ref_reconstruct(ssa, group_i, 1, x_i);
    ssa_opt_ref_reconstruct(ssa, group_j, 1, x_j);

    double inner = 0.0, norm_i = 0.0, norm_j = 0.0;

    for (int t = 0; t < N; t++)
    {
        double w = (double)ssa_ref_min(ssa_ref_min(t + 1, L),
                                       ssa_ref_min(K, N - t));
        inner += w * x_i[t] * x_j[t];
        norm_i += w * x_i[t] * x_i[t];
        norm_j += w * x_j[t] * x_j[t];
    }

    ssa_ref_free_ptr(x_i);
    ssa_ref_free_ptr(x_j);

    double denom = sqrt(norm_i) * sqrt(norm_j);
    return (denom > 1e-15) ? inner / denom : 0.0;
}

double ssa_opt_ref_variance_explained(const SSA_Opt_Ref *ssa, int start, int end)
{
    if (!ssa || !ssa->decomposed || start < 0 || ssa->total_variance <= 0)
    {
        return 0.0;
    }
    if (end < 0 || end >= ssa->n_components)
        end = ssa->n_components - 1;

    double sum = 0;
    for (int i = start; i <= end; i++)
        sum += ssa->eigenvalues[i];
    return sum / ssa->total_variance;
}

int ssa_opt_ref_get_trend(const SSA_Opt_Ref *ssa, double *output)
{
    int group[] = {0};
    return ssa_opt_ref_reconstruct(ssa, group, 1, output);
}

int ssa_opt_ref_get_noise(const SSA_Opt_Ref *ssa, int noise_start, double *output)
{
    if (!ssa || !ssa->decomposed || noise_start < 0)
        return -1;

    int n_noise = ssa->n_components - noise_start;
    if (n_noise <= 0)
    {
        ssa_ref_zero(output, ssa->N);
        return 0;
    }

    int *group = (int *)malloc(n_noise * sizeof(int));
    if (!group)
        return -1;

    for (int i = 0; i < n_noise; i++)
        group[i] = noise_start + i;

    int ret = ssa_opt_ref_reconstruct(ssa, group, n_noise, output);
    free(group);
    return ret;
}

#endif // SSA_OPT_REF_IMPLEMENTATION

#ifdef __cplusplus
}
#endif

#endif // SSA_OPT_REF_H
