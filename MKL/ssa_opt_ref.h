/*
 * ============================================================================
 * SSA-OPT Reference Implementation: Self-Contained Singular Spectrum Analysis
 * ============================================================================
 *
 * PURPOSE:
 *   This is a reference implementation for accuracy testing and education.
 *   It has ZERO external dependencies - everything is self-contained.
 *   Use ssa_opt_r2c.h with MKL for production performance.
 *
 * WHAT IS SSA?
 *   Singular Spectrum Analysis decomposes a time series into trend, periodic
 *   components, and noise by embedding it into a Hankel matrix and computing
 *   its SVD. This implementation avoids forming the explicit matrix.
 *
 * ALGORITHM:
 *   - O(N log N) Hankel matvec via FFT convolution (vs O(N² naive)
 *   - Sequential power iteration for SVD computation
 *   - Randomized SVD option for faster k >> 1 decomposition
 *   - Frequency-domain accumulation for fast reconstruction
 *   - Direct W-correlation formula (no reconstruction needed)
 *   - Built-in Cooley-Tukey radix-2 FFT
 *
 * OPTIMIZATIONS (matching MKL version):
 *   - Frequency-domain accumulation: 1 IFFT instead of n_group IFFTs
 *   - Direct W-correlation: O(n × N log N) instead of O(n² × N log N)
 *   - Correct sign-flip convergence check
 *   - Block power iteration option
 *   - Randomized SVD option
 *
 * USAGE:
 *   #define SSA_OPT_REF_IMPLEMENTATION
 *   #include "ssa_opt_ref.h"
 *
 *   SSA_Opt_Ref ssa = {0};
 *   ssa_opt_ref_init(&ssa, signal, N, L);
 *   ssa_opt_ref_decompose(&ssa, k, max_iter);           // Sequential
 *   // OR: ssa_opt_ref_decompose_randomized(&ssa, k, 8); // Faster for k > 10
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

#ifndef SSA_REF_MAX_ITER
#define SSA_REF_MAX_ITER 200
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
    double *ws_fft1;    // FFT scratch buffer 1
    double *ws_fft2;    // FFT scratch buffer 2
    double *ws_fft3;    // FFT scratch buffer 3 (for freq-domain accumulation)
    double *ws_u;       // Vector scratch for iteration
    double *ws_v;       // Vector scratch for iteration
    double *ws_proj;    // Projection coefficients for orthogonalization

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
    uint64_t rng_state; // xorshift64 state
} SSA_Opt_Ref;

/**
 * @brief Statistics for automatic component selection.
 */
typedef struct
{
    int n;                   // Number of components analyzed
    double *singular_values; // Copy of singular values
    double *log_sv;          // log(σᵢ) for scree plot
    double *gaps;            // Gap ratios: σᵢ/σᵢ₊₁
    double *cumulative_var;  // Cumulative explained variance
    int suggested_signal;    // Suggested signal component count
    double gap_threshold;    // Gap ratio at cutoff
} SSA_Ref_ComponentStats;

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
 * @brief Compute SVD via randomized algorithm (faster for k > 10).
 * @param ssa          Initialized SSA context
 * @param k            Number of singular triplets to compute
 * @param oversampling Extra random vectors for accuracy (typically 5-10)
 * @return             0 on success, -1 on error
 */
int ssa_opt_ref_decompose_randomized(SSA_Opt_Ref *ssa, int k, int oversampling);

/**
 * @brief Reconstruct signal from selected components.
 *
 * Uses frequency-domain accumulation: single IFFT regardless of n_group.
 *
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
 * @brief Compute W-correlation matrix using direct formula.
 *
 * Optimized: O(n × N log N + n² × N) instead of O(n² × N log N).
 * Uses h_i = conv(u_i, v_i) instead of full reconstruction.
 */
int ssa_opt_ref_wcorr_matrix(const SSA_Opt_Ref *ssa, double *W);

/**
 * @brief Compute W-correlation between two specific components.
 */
double ssa_opt_ref_wcorr_pair(const SSA_Opt_Ref *ssa, int i, int j);

/**
 * @brief Compute component statistics for automatic selection.
 */
int ssa_opt_ref_component_stats(const SSA_Opt_Ref *ssa, SSA_Ref_ComponentStats *stats);

/**
 * @brief Free component stats.
 */
void ssa_opt_ref_free_stats(SSA_Ref_ComponentStats *stats);

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
// Forecasting API
// ============================================================================

/**
 * @brief Compute Linear Recurrence Formula coefficients.
 * @param ssa         Decomposed SSA context
 * @param group       Components to use for LRF
 * @param n_group     Number of components
 * @param R           Output: LRF coefficients, length L-1
 * @param verticality Output: Verticality coefficient ν² (must be < 1 for stability)
 * @return            0 on success, -1 on error
 */
int ssa_opt_ref_compute_lrf(const SSA_Opt_Ref *ssa, const int *group, int n_group,
                            double *R, double *verticality);

/**
 * @brief Forecast using LRF.
 * @param ssa         Decomposed SSA context
 * @param group       Components used for forecasting
 * @param n_group     Number of components
 * @param n_forecast  Number of steps to forecast
 * @param output      Output buffer, length N + n_forecast
 * @return            0 on success, -1 on error
 */
int ssa_opt_ref_forecast(const SSA_Opt_Ref *ssa, const int *group, int n_group,
                         int n_forecast, double *output);

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
    if (!ptr) return;
#if defined(_MSC_VER)
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

// xorshift64 - better quality than LCG, still fast
static inline double ssa_ref_rand(uint64_t *state)
{
    uint64_t x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    // Convert to double in [-0.5, 0.5]
    return (double)(x >> 11) * (1.0 / 9007199254740992.0) - 0.5;
}

// Box-Muller transform for Gaussian random numbers
static inline double ssa_ref_randn(uint64_t *state)
{
    double u1 = ssa_ref_rand(state) + 0.5;  // (0, 1)
    double u2 = ssa_ref_rand(state) + 0.5;
    if (u1 < 1e-10) u1 = 1e-10;
    return sqrt(-2.0 * log(u1)) * cos(2.0 * 3.14159265358979323846 * u2);
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
    if (norm > 1e-12)  // Use 1e-12 for numerical stability
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
// Matrix Operations (for randomized SVD)
// ----------------------------------------------------------------------------

// y = A @ x (A is m×n column-major)
static void ssa_ref_gemv(const double *A, const double *x, double *y,
                         int m, int n, double alpha, double beta)
{
    for (int i = 0; i < m; i++)
    {
        double sum = 0.0;
        for (int j = 0; j < n; j++)
            sum += A[j * m + i] * x[j];
        y[i] = alpha * sum + beta * y[i];
    }
}

// y = A^T @ x (A is m×n column-major)
static void ssa_ref_gemv_T(const double *A, const double *x, double *y,
                           int m, int n, double alpha, double beta)
{
    for (int j = 0; j < n; j++)
    {
        double sum = 0.0;
        for (int i = 0; i < m; i++)
            sum += A[j * m + i] * x[i];
        y[j] = alpha * sum + beta * y[j];
    }
}

// C = A @ B (A is m×k, B is k×n, C is m×n, all column-major)
static void ssa_ref_gemm(const double *A, const double *B, double *C,
                         int m, int n, int k)
{
    ssa_ref_zero(C, m * n);
    for (int j = 0; j < n; j++)
    {
        for (int p = 0; p < k; p++)
        {
            double b_pj = B[j * k + p];
            for (int i = 0; i < m; i++)
            {
                C[j * m + i] += A[p * m + i] * b_pj;
            }
        }
    }
}

// C = A^T @ B (A is m×k, B is m×n, C is k×n)
static void ssa_ref_gemm_T(const double *A, const double *B, double *C,
                           int m, int n, int k)
{
    for (int j = 0; j < n; j++)
    {
        for (int i = 0; i < k; i++)
        {
            double sum = 0.0;
            for (int p = 0; p < m; p++)
                sum += A[i * m + p] * B[j * m + p];
            C[j * k + i] = sum;
        }
    }
}

// Householder QR factorization: A = Q @ R
// On exit: A contains R in upper triangle, Q is returned separately
static void ssa_ref_qr(double *A, double *Q, int m, int n)
{
    double *tau = (double *)ssa_ref_alloc(n * sizeof(double));
    double *work = (double *)ssa_ref_alloc(m * sizeof(double));

    // Compute Householder reflectors
    for (int j = 0; j < n; j++)
    {
        // Compute norm of column j below diagonal
        double norm = 0.0;
        for (int i = j; i < m; i++)
            norm += A[j * m + i] * A[j * m + i];
        norm = sqrt(norm);

        if (norm < 1e-15)
        {
            tau[j] = 0.0;
            continue;
        }

        // Compute Householder vector
        double sign = (A[j * m + j] >= 0) ? 1.0 : -1.0;
        double alpha = -sign * norm;
        double beta = A[j * m + j] - alpha;

        if (fabs(beta) < 1e-15)
        {
            tau[j] = 0.0;
            A[j * m + j] = alpha;
            continue;
        }

        // Store reflector in lower part of A
        for (int i = j + 1; i < m; i++)
            A[j * m + i] /= beta;

        tau[j] = beta / alpha;
        A[j * m + j] = alpha;

        // Apply reflector to remaining columns
        for (int k = j + 1; k < n; k++)
        {
            double dot = A[k * m + j];
            for (int i = j + 1; i < m; i++)
                dot += A[j * m + i] * A[k * m + i];
            dot *= tau[j];

            A[k * m + j] -= dot;
            for (int i = j + 1; i < m; i++)
                A[k * m + i] -= dot * A[j * m + i];
        }
    }

    // Form Q by applying reflectors to identity
    for (int i = 0; i < m * n; i++)
        Q[i] = 0.0;
    for (int i = 0; i < n; i++)
        Q[i * m + i] = 1.0;

    for (int j = n - 1; j >= 0; j--)
    {
        if (fabs(tau[j]) < 1e-15) continue;

        for (int k = j; k < n; k++)
        {
            double dot = Q[k * m + j];
            for (int i = j + 1; i < m; i++)
                dot += A[j * m + i] * Q[k * m + i];
            dot *= tau[j];

            Q[k * m + j] -= dot;
            for (int i = j + 1; i < m; i++)
                Q[k * m + i] -= dot * A[j * m + i];
        }
    }

    ssa_ref_free_ptr(tau);
    ssa_ref_free_ptr(work);
}

// Simple SVD for small matrices via Jacobi iteration
// A is m×n with m >= n. On exit: U is m×n, S is n, Vt is n×n
static void ssa_ref_svd_small(double *A, double *U, double *S, double *Vt,
                              int m, int n)
{
    const int max_sweeps = 30;
    const double tol = 1e-12;

    // Initialize U = A, Vt = I
    ssa_ref_copy(A, U, m * n);
    ssa_ref_zero(Vt, n * n);
    for (int i = 0; i < n; i++)
        Vt[i * n + i] = 1.0;

    // Jacobi iteration
    for (int sweep = 0; sweep < max_sweeps; sweep++)
    {
        double max_off = 0.0;

        for (int i = 0; i < n - 1; i++)
        {
            for (int j = i + 1; j < n; j++)
            {
                // Compute 2x2 submatrix of U^T @ U
                double aii = 0, aij = 0, ajj = 0;
                for (int k = 0; k < m; k++)
                {
                    double ui = U[i * m + k];
                    double uj = U[j * m + k];
                    aii += ui * ui;
                    ajj += uj * uj;
                    aij += ui * uj;
                }

                if (fabs(aij) < tol * sqrt(aii * ajj)) continue;
                if (fabs(aij) > max_off) max_off = fabs(aij);

                // Compute Jacobi rotation
                double tau = (ajj - aii) / (2.0 * aij);
                double t = (tau >= 0 ? 1.0 : -1.0) / (fabs(tau) + sqrt(1 + tau * tau));
                double c = 1.0 / sqrt(1 + t * t);
                double s = t * c;

                // Apply rotation to U
                for (int k = 0; k < m; k++)
                {
                    double ui = U[i * m + k];
                    double uj = U[j * m + k];
                    U[i * m + k] = c * ui - s * uj;
                    U[j * m + k] = s * ui + c * uj;
                }

                // Apply rotation to Vt
                for (int k = 0; k < n; k++)
                {
                    double vi = Vt[i * n + k];
                    double vj = Vt[j * n + k];
                    Vt[i * n + k] = c * vi - s * vj;
                    Vt[j * n + k] = s * vi + c * vj;
                }
            }
        }

        if (max_off < tol) break;
    }

    // Extract singular values (column norms of U) and normalize U
    for (int j = 0; j < n; j++)
    {
        double norm = 0.0;
        for (int i = 0; i < m; i++)
            norm += U[j * m + i] * U[j * m + i];
        S[j] = sqrt(norm);

        if (S[j] > 1e-12)
        {
            for (int i = 0; i < m; i++)
                U[j * m + i] /= S[j];
        }
    }

    // Sort by descending singular value
    for (int i = 0; i < n - 1; i++)
    {
        int max_idx = i;
        for (int j = i + 1; j < n; j++)
            if (S[j] > S[max_idx]) max_idx = j;

        if (max_idx != i)
        {
            // Swap singular values
            double tmp = S[i]; S[i] = S[max_idx]; S[max_idx] = tmp;

            // Swap columns of U
            for (int k = 0; k < m; k++)
            {
                tmp = U[i * m + k]; U[i * m + k] = U[max_idx * m + k]; U[max_idx * m + k] = tmp;
            }

            // Swap rows of Vt
            for (int k = 0; k < n; k++)
            {
                tmp = Vt[i * n + k]; Vt[i * n + k] = Vt[max_idx * n + k]; Vt[max_idx * n + k] = tmp;
            }
        }
    }
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

// Complex add: c += a (interleaved format)
static void ssa_ref_complex_add(double *c, const double *a, int n)
{
    for (int i = 0; i < 2 * n; i++)
        c[i] += a[i];
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

    // Initialize RNG with better seed
    ssa->rng_state = 0x123456789ABCDEF0ULL;

    // Allocate workspace
    size_t fft_size = 2 * fft_n * sizeof(double);
    size_t vec_size = ssa_ref_max(L, ssa->K) * sizeof(double);

    ssa->fft_x = (double *)ssa_ref_alloc(fft_size);
    ssa->ws_fft1 = (double *)ssa_ref_alloc(fft_size);
    ssa->ws_fft2 = (double *)ssa_ref_alloc(fft_size);
    ssa->ws_fft3 = (double *)ssa_ref_alloc(fft_size);  // For freq-domain accumulation
    ssa->ws_u = (double *)ssa_ref_alloc(vec_size);
    ssa->ws_v = (double *)ssa_ref_alloc(vec_size);
    ssa->ws_proj = (double *)ssa_ref_alloc(vec_size);  // For orthogonalization

    if (!ssa->fft_x || !ssa->ws_fft1 || !ssa->ws_fft2 || !ssa->ws_fft3 ||
        !ssa->ws_u || !ssa->ws_v || !ssa->ws_proj)
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

            // ================================================================
            // CONVERGENCE CHECK: Handle sign flips correctly
            //
            // Eigenvectors can flip sign between iterations.
            // If v_new = -v, then ||v - v_new|| ≈ 2 but they're converged!
            // Solution: check both ||v - v_new|| and ||v + v_new||, take min.
            // ================================================================
            double diff_same = 0.0, diff_flip = 0.0;
            for (int i = 0; i < K; i++)
            {
                double d_same = v[i] - v_new[i];
                double d_flip = v[i] + v_new[i];
                diff_same += d_same * d_same;
                diff_flip += d_flip * d_flip;
            }
            double diff = (diff_same < diff_flip) ? diff_same : diff_flip;

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

        if (sigma > 1e-12)
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

// ----------------------------------------------------------------------------
// Randomized SVD Decomposition
// ----------------------------------------------------------------------------

int ssa_opt_ref_decompose_randomized(SSA_Opt_Ref *ssa, int k, int oversampling)
{
    if (!ssa || !ssa->initialized || k < 1)
        return -1;

    int L = ssa->L;
    int K = ssa->K;
    int p = (oversampling <= 0) ? 8 : oversampling;
    int kp = k + p;

    kp = ssa_ref_min(kp, ssa_ref_min(L, K));
    k = ssa_ref_min(k, kp);

    // Allocate results
    ssa->U = (double *)ssa_ref_alloc(L * k * sizeof(double));
    ssa->V = (double *)ssa_ref_alloc(K * k * sizeof(double));
    ssa->sigma = (double *)ssa_ref_alloc(k * sizeof(double));
    ssa->eigenvalues = (double *)ssa_ref_alloc(k * sizeof(double));

    if (!ssa->U || !ssa->V || !ssa->sigma || !ssa->eigenvalues)
        return -1;

    ssa->n_components = k;
    ssa->total_variance = 0.0;

    // Allocate workspace
    double *Omega = (double *)ssa_ref_alloc(K * kp * sizeof(double));
    double *Y = (double *)ssa_ref_alloc(L * kp * sizeof(double));
    double *Q = (double *)ssa_ref_alloc(L * kp * sizeof(double));
    double *B = (double *)ssa_ref_alloc(K * kp * sizeof(double));
    double *U_small = (double *)ssa_ref_alloc(K * kp * sizeof(double));
    double *S_small = (double *)ssa_ref_alloc(kp * sizeof(double));
    double *Vt_small = (double *)ssa_ref_alloc(kp * kp * sizeof(double));

    if (!Omega || !Y || !Q || !B || !U_small || !S_small || !Vt_small)
    {
        ssa_ref_free_ptr(Omega);
        ssa_ref_free_ptr(Y);
        ssa_ref_free_ptr(Q);
        ssa_ref_free_ptr(B);
        ssa_ref_free_ptr(U_small);
        ssa_ref_free_ptr(S_small);
        ssa_ref_free_ptr(Vt_small);
        return -1;
    }

    // Step 1: Generate random test matrix Omega (K × kp)
    for (int i = 0; i < K * kp; i++)
        Omega[i] = ssa_ref_randn(&ssa->rng_state);

    // Step 2: Y = H @ Omega (L × kp)
    for (int j = 0; j < kp; j++)
    {
        ssa_ref_hankel_matvec(ssa, &Omega[j * K], &Y[j * L]);
    }

    // Step 3: QR factorization Y = Q @ R
    ssa_ref_copy(Y, Q, L * kp);
    double *R_dummy = (double *)ssa_ref_alloc(kp * kp * sizeof(double));
    ssa_ref_qr(Q, Y, L, kp);  // Y now contains Q
    ssa_ref_copy(Y, Q, L * kp);
    ssa_ref_free_ptr(R_dummy);

    // Step 4: B = H^T @ Q (K × kp)
    for (int j = 0; j < kp; j++)
    {
        ssa_ref_hankel_matvec_T(ssa, &Q[j * L], &B[j * K]);
    }

    // Step 5: SVD of B
    ssa_ref_svd_small(B, U_small, S_small, Vt_small, K, kp);

    // Step 6: U = Q @ Vt^T (first k columns)
    // Vt is kp × kp, we want Q @ Vt^T which gives L × kp
    // Then take first k columns for U
    for (int j = 0; j < k; j++)
    {
        for (int i = 0; i < L; i++)
        {
            double sum = 0.0;
            for (int p = 0; p < kp; p++)
                sum += Q[p * L + i] * Vt_small[j * kp + p];  // Vt^T[p,j] = Vt[j,p]
            ssa->U[j * L + i] = sum;
        }
    }

    // V = U_small (first k columns)
    for (int j = 0; j < k; j++)
    {
        ssa_ref_copy(&U_small[j * K], &ssa->V[j * K], K);
    }

    // Store singular values
    for (int i = 0; i < k; i++)
    {
        ssa->sigma[i] = S_small[i];
        ssa->eigenvalues[i] = S_small[i] * S_small[i];
        ssa->total_variance += ssa->eigenvalues[i];
    }

    // Fix sign convention
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

    // Cleanup
    ssa_ref_free_ptr(Omega);
    ssa_ref_free_ptr(Y);
    ssa_ref_free_ptr(Q);
    ssa_ref_free_ptr(B);
    ssa_ref_free_ptr(U_small);
    ssa_ref_free_ptr(S_small);
    ssa_ref_free_ptr(Vt_small);

    ssa->decomposed = true;
    return 0;
}

// ----------------------------------------------------------------------------
// Reconstruction with Frequency-Domain Accumulation
// ----------------------------------------------------------------------------

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

    // Cast away const for workspace access
    SSA_Opt_Ref *ssa_mut = (SSA_Opt_Ref *)ssa;

    // =========================================================================
    // FREQUENCY-DOMAIN ACCUMULATION
    //
    // Instead of: for each component { FFT, FFT, multiply, IFFT, accumulate }
    // We do:      for each component { FFT, FFT, multiply, accumulate_complex }
    //             single IFFT at end
    //
    // Reduces n_group IFFTs to 1 IFFT!
    // =========================================================================

    double *freq_accum = ssa_mut->ws_fft3;
    ssa_ref_zero(freq_accum, 2 * fft_n);

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

        // Accumulate in frequency domain
        ssa_ref_complex_add(freq_accum, ws1, fft_n);
    }

    // Single IFFT at the end
    ssa_ref_fft(freq_accum, fft_n, 1);

    // Extract real parts and apply diagonal averaging
    for (int t = 0; t < N; t++)
    {
        output[t] = freq_accum[2 * t] * ssa->inv_diag_count[t];
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
    ssa_ref_free_ptr(ssa->ws_fft3);
    ssa_ref_free_ptr(ssa->ws_u);
    ssa_ref_free_ptr(ssa->ws_v);
    ssa_ref_free_ptr(ssa->ws_proj);
    ssa_ref_free_ptr(ssa->inv_diag_count);
    ssa_ref_free_ptr(ssa->U);
    ssa_ref_free_ptr(ssa->V);
    ssa_ref_free_ptr(ssa->sigma);
    ssa_ref_free_ptr(ssa->eigenvalues);

    memset(ssa, 0, sizeof(SSA_Opt_Ref));
}

// ----------------------------------------------------------------------------
// Optimized W-Correlation (Direct Formula)
// ----------------------------------------------------------------------------

int ssa_opt_ref_wcorr_matrix(const SSA_Opt_Ref *ssa, double *W)
{
    if (!ssa || !ssa->decomposed || !W)
        return -1;

    int N = ssa->N;
    int L = ssa->L;
    int K = ssa->K;
    int n = ssa->n_components;
    int fft_n = ssa->fft_len;

    // Cast away const for workspace
    SSA_Opt_Ref *ssa_mut = (SSA_Opt_Ref *)ssa;

    // =========================================================================
    // OPTIMIZED W-CORRELATION
    //
    // Key insight: X_i[t] = σ_i × h_i[t] / c[t] where h_i = conv(u_i, v_i)
    //
    // <X_i, X_j>_W = σ_i × σ_j × Σ_t h_i[t] × h_j[t] / c[t]
    //
    // Steps:
    //   1. Compute h_i = conv(u_i, v_i) for all i via FFT: O(n × N log N)
    //   2. Compute G matrix: G[i,t] = σ_i × h_i[t] / sqrt(c[t]) / ||X_i||_W
    //   3. W[i,j] = Σ_t G[i,t] × G[j,t] = G @ G^T: O(n² × N)
    //
    // Total: O(n × N log N + n² × N) vs O(n² × N log N) naive
    // =========================================================================

    // Compute inverse sqrt weights
    double *inv_sqrt_c = (double *)ssa_ref_alloc(N * sizeof(double));
    double *inv_c = (double *)ssa_ref_alloc(N * sizeof(double));
    if (!inv_sqrt_c || !inv_c)
    {
        ssa_ref_free_ptr(inv_sqrt_c);
        ssa_ref_free_ptr(inv_c);
        return -1;
    }

    for (int t = 0; t < N; t++)
    {
        int c_t = ssa_ref_min(ssa_ref_min(t + 1, L), ssa_ref_min(K, N - t));
        inv_c[t] = 1.0 / c_t;
        inv_sqrt_c[t] = sqrt(inv_c[t]);
    }

    // Compute h_i and build scaled G matrix
    double *G = (double *)ssa_ref_alloc(n * N * sizeof(double));
    double *norms = (double *)ssa_ref_alloc(n * sizeof(double));
    double *h_temp = (double *)ssa_ref_alloc(2 * fft_n * sizeof(double));
    double *u_fft = (double *)ssa_ref_alloc(2 * fft_n * sizeof(double));
    double *v_fft = (double *)ssa_ref_alloc(2 * fft_n * sizeof(double));

    if (!G || !norms || !h_temp || !u_fft || !v_fft)
    {
        ssa_ref_free_ptr(inv_sqrt_c);
        ssa_ref_free_ptr(inv_c);
        ssa_ref_free_ptr(G);
        ssa_ref_free_ptr(norms);
        ssa_ref_free_ptr(h_temp);
        ssa_ref_free_ptr(u_fft);
        ssa_ref_free_ptr(v_fft);
        return -1;
    }

    for (int i = 0; i < n; i++)
    {
        double sigma = ssa->sigma[i];
        const double *u_vec = &ssa->U[i * L];
        const double *v_vec = &ssa->V[i * K];

        // FFT(u_i)
        ssa_ref_fft_r2c(fft_n, u_vec, L, u_fft);

        // FFT(v_i)
        ssa_ref_fft_r2c(fft_n, v_vec, K, v_fft);

        // h_i = IFFT(FFT(u) × FFT(v))
        ssa_ref_complex_mul(u_fft, v_fft, h_temp, fft_n);
        ssa_ref_fft(h_temp, fft_n, 1);

        // Compute ||X_i||_W² = σ² × Σ_t h_i[t]² / c[t]
        double norm_sq = 0.0;
        for (int t = 0; t < N; t++)
        {
            double h_t = h_temp[2 * t];  // Real part
            norm_sq += h_t * h_t * inv_c[t];
        }
        norm_sq *= sigma * sigma;
        norms[i] = sqrt(norm_sq);

        // Store scaled h in G: G[i,t] = σ × h[t] / sqrt(c[t]) / ||X_i||_W
        double scale = (norms[i] > 1e-12) ? sigma / norms[i] : 0.0;
        double *g_row = &G[i * N];
        for (int t = 0; t < N; t++)
        {
            g_row[t] = scale * h_temp[2 * t] * inv_sqrt_c[t];
        }
    }

    // Compute W = G @ G^T
    for (int i = 0; i < n; i++)
    {
        double *g_i = &G[i * N];
        for (int j = i; j < n; j++)
        {
            double *g_j = &G[j * N];
            double dot = 0.0;
            for (int t = 0; t < N; t++)
                dot += g_i[t] * g_j[t];
            W[i * n + j] = dot;
            W[j * n + i] = dot;
        }
    }

    ssa_ref_free_ptr(inv_sqrt_c);
    ssa_ref_free_ptr(inv_c);
    ssa_ref_free_ptr(G);
    ssa_ref_free_ptr(norms);
    ssa_ref_free_ptr(h_temp);
    ssa_ref_free_ptr(u_fft);
    ssa_ref_free_ptr(v_fft);

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
    int fft_n = ssa->fft_len;

    // Allocate temp buffers
    double *h_i = (double *)ssa_ref_alloc(2 * fft_n * sizeof(double));
    double *h_j = (double *)ssa_ref_alloc(2 * fft_n * sizeof(double));
    double *u_fft = (double *)ssa_ref_alloc(2 * fft_n * sizeof(double));
    double *v_fft = (double *)ssa_ref_alloc(2 * fft_n * sizeof(double));

    if (!h_i || !h_j || !u_fft || !v_fft)
    {
        ssa_ref_free_ptr(h_i);
        ssa_ref_free_ptr(h_j);
        ssa_ref_free_ptr(u_fft);
        ssa_ref_free_ptr(v_fft);
        return 0.0;
    }

    // Compute h_i = conv(u_i, v_i)
    ssa_ref_fft_r2c(fft_n, &ssa->U[i * L], L, u_fft);
    ssa_ref_fft_r2c(fft_n, &ssa->V[i * K], K, v_fft);
    ssa_ref_complex_mul(u_fft, v_fft, h_i, fft_n);
    ssa_ref_fft(h_i, fft_n, 1);

    // Compute h_j = conv(u_j, v_j)
    ssa_ref_fft_r2c(fft_n, &ssa->U[j * L], L, u_fft);
    ssa_ref_fft_r2c(fft_n, &ssa->V[j * K], K, v_fft);
    ssa_ref_complex_mul(u_fft, v_fft, h_j, fft_n);
    ssa_ref_fft(h_j, fft_n, 1);

    // Compute W-correlation
    double sigma_i = ssa->sigma[i];
    double sigma_j = ssa->sigma[j];

    double inner = 0.0, norm_i_sq = 0.0, norm_j_sq = 0.0;
    for (int t = 0; t < N; t++)
    {
        int c_t = ssa_ref_min(ssa_ref_min(t + 1, L), ssa_ref_min(K, N - t));
        double inv_c = 1.0 / c_t;
        double hi = h_i[2 * t];
        double hj = h_j[2 * t];
        inner += hi * hj * inv_c;
        norm_i_sq += hi * hi * inv_c;
        norm_j_sq += hj * hj * inv_c;
    }

    inner *= sigma_i * sigma_j;
    norm_i_sq *= sigma_i * sigma_i;
    norm_j_sq *= sigma_j * sigma_j;

    ssa_ref_free_ptr(h_i);
    ssa_ref_free_ptr(h_j);
    ssa_ref_free_ptr(u_fft);
    ssa_ref_free_ptr(v_fft);

    double denom = sqrt(norm_i_sq) * sqrt(norm_j_sq);
    return (denom > 1e-12) ? inner / denom : 0.0;
}

// ----------------------------------------------------------------------------
// Component Statistics
// ----------------------------------------------------------------------------

int ssa_opt_ref_component_stats(const SSA_Opt_Ref *ssa, SSA_Ref_ComponentStats *stats)
{
    if (!ssa || !ssa->decomposed || !stats)
        return -1;

    int n = ssa->n_components;

    stats->n = n;
    stats->singular_values = (double *)ssa_ref_alloc(n * sizeof(double));
    stats->log_sv = (double *)ssa_ref_alloc(n * sizeof(double));
    stats->gaps = (double *)ssa_ref_alloc((n - 1) * sizeof(double));
    stats->cumulative_var = (double *)ssa_ref_alloc(n * sizeof(double));

    if (!stats->singular_values || !stats->log_sv || !stats->gaps || !stats->cumulative_var)
    {
        ssa_opt_ref_free_stats(stats);
        return -1;
    }

    // Copy singular values and compute log
    for (int i = 0; i < n; i++)
    {
        stats->singular_values[i] = ssa->sigma[i];
        stats->log_sv[i] = (ssa->sigma[i] > 1e-15) ? log(ssa->sigma[i]) : -35.0;
    }

    // Compute gap ratios
    double max_gap = 0.0;
    int max_gap_idx = 0;
    for (int i = 0; i < n - 1; i++)
    {
        stats->gaps[i] = (ssa->sigma[i + 1] > 1e-15) ?
                         ssa->sigma[i] / ssa->sigma[i + 1] : 1e10;
        if (stats->gaps[i] > max_gap && i < n / 2)
        {
            max_gap = stats->gaps[i];
            max_gap_idx = i;
        }
    }

    // Compute cumulative variance
    double cumsum = 0.0;
    for (int i = 0; i < n; i++)
    {
        cumsum += ssa->eigenvalues[i];
        stats->cumulative_var[i] = cumsum / ssa->total_variance;
    }

    stats->suggested_signal = max_gap_idx + 1;
    stats->gap_threshold = max_gap;

    return 0;
}

void ssa_opt_ref_free_stats(SSA_Ref_ComponentStats *stats)
{
    if (!stats) return;
    ssa_ref_free_ptr(stats->singular_values);
    ssa_ref_free_ptr(stats->log_sv);
    ssa_ref_free_ptr(stats->gaps);
    ssa_ref_free_ptr(stats->cumulative_var);
    memset(stats, 0, sizeof(SSA_Ref_ComponentStats));
}

// ----------------------------------------------------------------------------
// Utility Functions
// ----------------------------------------------------------------------------

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

// ----------------------------------------------------------------------------
// Forecasting
// ----------------------------------------------------------------------------

int ssa_opt_ref_compute_lrf(const SSA_Opt_Ref *ssa, const int *group, int n_group,
                            double *R, double *verticality)
{
    if (!ssa || !ssa->decomposed || !group || !R || !verticality || n_group < 1)
        return -1;

    int L = ssa->L;

    // Extract π = last row of U for selected components
    // Compute ν² = Σ πᵢ² (verticality coefficient)
    double nu_sq = 0.0;
    double *pi = (double *)ssa_ref_alloc(n_group * sizeof(double));
    if (!pi) return -1;

    for (int g = 0; g < n_group; g++)
    {
        int idx = group[g];
        if (idx < 0 || idx >= ssa->n_components)
        {
            pi[g] = 0.0;
            continue;
        }
        pi[g] = ssa->U[idx * L + (L - 1)];  // Last element of column idx
        nu_sq += pi[g] * pi[g];
    }

    *verticality = nu_sq;

    if (nu_sq >= 1.0 - 1e-10)
    {
        // LRF not stable
        ssa_ref_zero(R, L - 1);
        ssa_ref_free_ptr(pi);
        return -1;
    }

    double scale = 1.0 / (1.0 - nu_sq);

    // R[j] = scale × Σᵢ πᵢ × U[j, i] for j = 0..L-2
    ssa_ref_zero(R, L - 1);
    for (int g = 0; g < n_group; g++)
    {
        int idx = group[g];
        if (idx < 0 || idx >= ssa->n_components)
            continue;
        const double *u_col = &ssa->U[idx * L];
        double coeff = scale * pi[g];
        for (int j = 0; j < L - 1; j++)
            R[j] += coeff * u_col[j];
    }

    ssa_ref_free_ptr(pi);
    return 0;
}

int ssa_opt_ref_forecast(const SSA_Opt_Ref *ssa, const int *group, int n_group,
                         int n_forecast, double *output)
{
    if (!ssa || !ssa->decomposed || !group || !output || n_group < 1 || n_forecast < 1)
        return -1;

    int N = ssa->N;
    int L = ssa->L;

    // First reconstruct the signal
    if (ssa_opt_ref_reconstruct(ssa, group, n_group, output) != 0)
        return -1;

    // Compute LRF
    double *R = (double *)ssa_ref_alloc((L - 1) * sizeof(double));
    double verticality;

    if (!R) return -1;

    if (ssa_opt_ref_compute_lrf(ssa, group, n_group, R, &verticality) != 0)
    {
        ssa_ref_free_ptr(R);
        return -1;
    }

    // Apply LRF for forecasting
    // x̃[t] = Σⱼ R[j] × x̃[t - L + 1 + j] for t = N, N+1, ..., N+n_forecast-1
    for (int t = N; t < N + n_forecast; t++)
    {
        double sum = 0.0;
        for (int j = 0; j < L - 1; j++)
        {
            int idx = t - L + 1 + j;
            if (idx >= 0)
                sum += R[j] * output[idx];
        }
        output[t] = sum;
    }

    ssa_ref_free_ptr(R);
    return 0;
}

#endif // SSA_OPT_REF_IMPLEMENTATION

#ifdef __cplusplus
}
#endif

#endif // SSA_OPT_REF_H
