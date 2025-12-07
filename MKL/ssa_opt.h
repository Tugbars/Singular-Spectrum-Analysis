/*
 * ============================================================================
 * SSA-OPT: High-Performance Singular Spectrum Analysis (MKL R2C Version)
 * ============================================================================
 */

#ifndef SSA_OPT_R2C_H
#define SSA_OPT_R2C_H

#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <mkl.h>
#include <mkl_dfti.h>
#include <mkl_vml.h>
#include <mkl_vsl.h>
#include <mkl_lapacke.h>

// SIMD intrinsics for AVX2/AVX-512 optimizations
#ifdef _MSC_VER
#include <intrin.h>
#else
#include <immintrin.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef __cplusplus
extern "C"
{
#endif

#ifndef SSA_CONVERGENCE_TOL
#define SSA_CONVERGENCE_TOL 1e-12
#endif

#define SSA_ALIGN 64

#ifndef SSA_BATCH_SIZE
#define SSA_BATCH_SIZE 32
#endif

    typedef struct
    {
        // === Dimensions ===
        int N;       // Signal length (number of samples)
        int L;       // Window length (embedding dimension), typically N/4 to N/2
        int K;       // Number of lagged copies, K = N - L + 1
        int fft_len; // FFT length (next power of 2 >= N + K for convolution)
        int r2c_len; // Real-to-complex output length = fft_len/2 + 1 (Hermitian symmetry)

        // === MKL FFT Descriptors ===
        DFTI_DESCRIPTOR_HANDLE fft_r2c;       // Forward: real → complex (single vector)
        DFTI_DESCRIPTOR_HANDLE fft_c2r;       // Inverse: complex → real (single vector)
        DFTI_DESCRIPTOR_HANDLE fft_r2c_batch; // Forward batched: up to SSA_BATCH_SIZE vectors
        DFTI_DESCRIPTOR_HANDLE fft_c2r_batch; // Inverse batched: up to SSA_BATCH_SIZE vectors
        DFTI_DESCRIPTOR_HANDLE fft_c2r_wcorr; // Inverse for W-correlation (n_components at once)

        // === Thread-Local FFT Descriptor Pool (OpenMP optimization) ===
        // Pre-allocated per-thread descriptors to avoid expensive DftiCommitDescriptor
        // calls inside hot loops like ssa_opt_cache_ffts
#ifdef _OPENMP
        DFTI_DESCRIPTOR_HANDLE *thread_fft_pool; // Array of descriptors, one per thread
        int thread_pool_size;                    // Number of threads (omp_get_max_threads at init)
#endif

        // === Random Number Generator ===
        VSLStreamStatePtr rng; // MKL VSL stream for randomized SVD (Mersenne Twister)

        // === Core FFT Workspaces ===
        double *fft_x;      // Pre-computed FFT of signal x, length 2*r2c_len (complex)
        double *ws_real;    // Real workspace for FFT input, length fft_len
        double *ws_complex; // Complex workspace for FFT output, length 2*r2c_len
        double *ws_real2;   // Secondary real workspace for adjoint matvec
        double *ws_proj;    // Projection workspace for power iteration

        // === Batched FFT Workspaces ===
        double *ws_batch_real;    // Batched real input, fft_len * SSA_BATCH_SIZE
        double *ws_batch_complex; // Batched complex output, 2*r2c_len * SSA_BATCH_SIZE

        // === SVD Results ===
        double *U;           // Left singular vectors, L × n_components (column-major)
        double *V;           // Right singular vectors, K × n_components (column-major)
        double *sigma;       // Singular values σ₀ ≥ σ₁ ≥ ... ≥ σₖ₋₁
        double *eigenvalues; // Eigenvalues λᵢ = σᵢ² (of HᵀH)
        int n_components;    // Number of computed components

        // === Reconstruction Cache (frequency-domain accumulation) ===
        double *inv_diag_count; // 1/w[t] where w[t] = diagonal averaging weight, length N
        double *U_fft;          // Cached FFT of scaled U columns: FFT(σᵢ·Uᵢ), n_components × 2*r2c_len
        double *V_fft;          // Cached FFT of V columns: FFT(Vᵢ), n_components × 2*r2c_len
        bool fft_cached;        // True if U_fft/V_fft are valid

        // === W-Correlation Workspace (DSYRK optimization) ===
        double *wcorr_ws_complex; // Complex workspace for batched component FFTs
        double *wcorr_h;          // Reconstructed components before weighting, n × N
        double *wcorr_G;          // Weighted/normalized matrix for DSYRK, n × N
        double *wcorr_sqrt_inv_c; // Precomputed √(1/w[t]) for fast weighting

        // === Randomized SVD Workspace (malloc-free hot path) ===
        int prepared_kp;          // Max k+p this workspace supports (set by ssa_opt_prepare)
        int prepared_lwork;       // LAPACK workspace size for DGESDD
        double *decomp_Omega;     // Random Gaussian matrix, K × (k+p)
        double *decomp_Y;         // Random projection Y = H·Ω, L × (k+p)
        double *decomp_Q;         // Orthonormal basis from QR(Y), L × (k+p)
        double *decomp_B;         // Projected matrix B = Hᵀ·Q, K × (k+p)
        double *decomp_tau;       // Householder reflectors from QR, length k+p
        double *decomp_B_left;    // Left singular vectors of B (DGESDD output)
        double *decomp_B_right_T; // Right singular vectors of B transposed
        double *decomp_S;         // Singular values from small SVD, length k+p
        double *decomp_work;      // LAPACK DGESDD workspace, length prepared_lwork
        int *decomp_iwork;        // LAPACK DGESDD integer workspace, length 8*(k+p)
        bool prepared;            // True if randomized workspace is allocated

        // === State Flags ===
        bool initialized; // True after ssa_opt_init() succeeds
        bool decomposed;  // True after any decompose function succeeds

        // === Statistics ===
        double total_variance; // Sum of all eigenvalues (trace of HᵀH)

    } SSA_Opt;

    // === Component Statistics (for automatic grouping / rank selection) ===
    typedef struct
    {
        int n; // Number of components analyzed

        // === Singular Value Analysis ===
        double *singular_values; // σᵢ values, length n (descending order)
        double *log_sv;          // log(σᵢ), useful for scree plot visualization
        double *gaps;            // Relative gaps: (σᵢ - σᵢ₊₁) / σᵢ, length n-1
                                 // Large gap suggests signal/noise boundary

        // === Variance Analysis ===
        double *cumulative_var; // Cumulative variance explained: Σⱼ₌₀ⁱ λⱼ / Σλ
                                // cumulative_var[i] = fraction of total variance in components 0..i

        // === Automatic Rank Selection ===
        double *second_diff;  // Second difference of log(σ): Δ²log(σᵢ), length n-2
                              // Peak indicates "elbow" in scree plot
        int suggested_signal; // Suggested number of signal components (auto-detected)
        double gap_threshold; // Threshold used for gap detection (default 0.1 = 10% drop)

    } SSA_ComponentStats;

    // === Linear Recurrence Formula (for R-forecasting) ===
    typedef struct
    {
        double *R; // LRF coefficients [a₁, a₂, ..., aₗ₋₁], length L-1
                   // Forecasts via: x[n] = Σⱼ aⱼ · x[n-j]
                   // Derived from last row of eigenvectors (shifted structure)

        int L; // Window length (determines LRF order = L-1)

        double verticality; // ν² = Σᵢ πᵢ² where πᵢ = U[L-1, i] (last row of U)
                            // Measures forecast stability:
                            //   ν² ≈ 0: stable, coefficients well-defined
                            //   ν² → 1: unstable, LRF becomes singular
                            // Rule: if ν² > 0.9, use V-forecast instead

        bool valid; // True if LRF was computed successfully
                    // False if verticality too high or other failure

    } SSA_LRF;

    // === Multivariate SSA (M time series analyzed jointly) ===
    typedef struct
    {
        // === Dimensions ===
        int M;       // Number of time series (channels)
        int N;       // Length of each series (all must be equal)
        int L;       // Window length per series
        int K;       // Number of lagged copies, K = N - L + 1
        int fft_len; // FFT length for Hankel matvec
        int r2c_len; // Real-to-complex output length = fft_len/2 + 1

        // === MKL FFT Descriptors ===
        DFTI_DESCRIPTOR_HANDLE fft_r2c;       // Forward: real → complex
        DFTI_DESCRIPTOR_HANDLE fft_c2r;       // Inverse: complex → real
        DFTI_DESCRIPTOR_HANDLE fft_r2c_batch; // Forward batched
        DFTI_DESCRIPTOR_HANDLE fft_c2r_batch; // Inverse batched

        // === Random Number Generator ===
        VSLStreamStatePtr rng; // For randomized SVD

        // === FFT Workspaces ===
        double *fft_x;            // Pre-computed FFT of all M signals, M × 2*r2c_len
                                  // fft_x[m * 2*r2c_len ...] = FFT(series m)
        double *ws_real;          // Real workspace, length fft_len
        double *ws_complex;       // Complex workspace, length 2*r2c_len
        double *ws_batch_real;    // Batched real workspace
        double *ws_batch_complex; // Batched complex workspace

        // === SVD Results ===
        // Block trajectory matrix is (M·L) × K, stacking L-lagged windows from all series
        double *U;           // Left singular vectors, (M·L) × n_components
                             // U is block-structured: U[m*L : (m+1)*L, :] for series m
        double *V;           // Right singular vectors, K × n_components (shared across series)
        double *sigma;       // Singular values, length n_components
        double *eigenvalues; // Eigenvalues λᵢ = σᵢ²
        int n_components;    // Number of computed components

        // === Reconstruction ===
        double *inv_diag_count; // Diagonal averaging weights (same for all series)

        // === State ===
        bool initialized;      // True after mssa_opt_init() succeeds
        bool decomposed;       // True after mssa_opt_decompose() succeeds
        double total_variance; // Total variance across all M series

    } MSSA_Opt;

    // Public API declarations
    int ssa_opt_init(SSA_Opt *ssa, const double *x, int N, int L);
    void ssa_opt_free(SSA_Opt *ssa);
    int ssa_opt_decompose(SSA_Opt *ssa, int k, int max_iter);
    int ssa_opt_decompose_block(SSA_Opt *ssa, int k, int block_size, int max_iter);

    // Randomized SVD decomposition - REQUIRES ssa_opt_prepare() first!
    // Returns -1 if prepare() was not called or if k+oversampling > prepared_kp
    int ssa_opt_decompose_randomized(SSA_Opt *ssa, int k, int oversampling);

    // Pre-allocate workspace for malloc-free hot path (call once before streaming loop)
    int ssa_opt_prepare(SSA_Opt *ssa, int max_k, int oversampling);
    void ssa_opt_free_prepared(SSA_Opt *ssa);

    // Update signal data without reallocation (just memcpy + 1 FFT)
    int ssa_opt_update_signal(SSA_Opt *ssa, const double *new_x);

    int ssa_opt_extend(SSA_Opt *ssa, int additional_k, int max_iter);
    int ssa_opt_reconstruct(const SSA_Opt *ssa, const int *group, int n_group, double *output);
    int ssa_opt_cache_ffts(SSA_Opt *ssa);
    void ssa_opt_free_cached_ffts(SSA_Opt *ssa);
    int ssa_opt_wcorr_matrix(const SSA_Opt *ssa, double *W);
    int ssa_opt_wcorr_matrix_fast(const SSA_Opt *ssa, double *W);
    double ssa_opt_wcorr_pair(const SSA_Opt *ssa, int i, int j);
    int ssa_opt_component_stats(const SSA_Opt *ssa, SSA_ComponentStats *stats);
    void ssa_opt_component_stats_free(SSA_ComponentStats *stats);
    int ssa_opt_find_periodic_pairs(const SSA_Opt *ssa, int *pairs, int max_pairs, double sv_tol, double wcorr_thresh);
    int ssa_opt_compute_lrf(const SSA_Opt *ssa, const int *group, int n_group, SSA_LRF *lrf);
    void ssa_opt_lrf_free(SSA_LRF *lrf);
    int ssa_opt_forecast(const SSA_Opt *ssa, const int *group, int n_group, int n_forecast, double *output);
    int ssa_opt_forecast_full(const SSA_Opt *ssa, const int *group, int n_group, int n_forecast, double *output);
    int ssa_opt_forecast_with_lrf(const SSA_LRF *lrf, const double *base_signal, int base_len, int n_forecast, double *output);

    // Vector forecast (V-forecast) - alternative to recurrent forecast
    // Projects onto eigenvector subspace at each step instead of using LRR
    int ssa_opt_vforecast(const SSA_Opt *ssa, const int *group, int n_group, int n_forecast, double *output);
    int ssa_opt_vforecast_full(const SSA_Opt *ssa, const int *group, int n_group, int n_forecast, double *output);
    int ssa_opt_vforecast_fast(const SSA_Opt *ssa, const int *group, int n_group,
                               const double *base_signal, int base_len, int n_forecast, double *output);

    int mssa_opt_init(MSSA_Opt *mssa, const double *X, int M, int N, int L);
    int mssa_opt_decompose(MSSA_Opt *mssa, int k, int oversampling);
    int mssa_opt_reconstruct(const MSSA_Opt *mssa, int series_idx, const int *group, int n_group, double *output);
    int mssa_opt_reconstruct_all(const MSSA_Opt *mssa, const int *group, int n_group, double *output);
    int mssa_opt_series_contributions(const MSSA_Opt *mssa, double *contributions);
    double mssa_opt_variance_explained(const MSSA_Opt *mssa, int start, int end);
    void mssa_opt_free(MSSA_Opt *mssa);
    int ssa_opt_get_trend(const SSA_Opt *ssa, double *output);
    int ssa_opt_get_noise(const SSA_Opt *ssa, int noise_start, double *output);
    double ssa_opt_variance_explained(const SSA_Opt *ssa, int start, int end);

    // Getters for decomposition results
    int ssa_opt_get_singular_values(const SSA_Opt *ssa, double *output, int max_n);
    int ssa_opt_get_eigenvalues(const SSA_Opt *ssa, double *output, int max_n);
    double ssa_opt_get_total_variance(const SSA_Opt *ssa);

    // Cadzow iterations - iterative finite-rank signal approximation
    typedef struct
    {
        int iterations;    // Iterations performed
        double final_diff; // Final relative difference
        double converged;  // 1.0 if converged, 0.0 if hit max_iter
    } SSA_CadzowResult;

    int ssa_opt_cadzow(const double *x, int N, int L, int rank, int max_iter, double tol,
                       double *output, SSA_CadzowResult *result);
    int ssa_opt_cadzow_weighted(const double *x, int N, int L, int rank, int max_iter,
                                double tol, double alpha, double *output, SSA_CadzowResult *result);
    int ssa_opt_cadzow_inplace(SSA_Opt *ssa, int rank, int max_iter, double tol, SSA_CadzowResult *result);

    // ESPRIT - Estimation of Signal Parameters via Rotational Invariance
    // Extracts frequencies/periods from eigenvectors
    typedef struct
    {
        double *periods;     // Estimated periods (in samples), length = n_components
        double *frequencies; // Frequencies (cycles per sample), length = n_components
        double *moduli;      // |eigenvalue| - damping factor (1.0 = undamped sinusoid)
        double *rates;       // log(|eigenvalue|) - damping rate (0 = undamped)
        int n_components;    // Number of components analyzed
    } SSA_ParEstimate;

    // Estimate periods/frequencies from SSA eigenvectors using ESPRIT
    // group: component indices to analyze (NULL = use all decomposed components)
    // n_group: number of components in group (0 = use all)
    // Returns 0 on success, allocates result arrays (caller must free with ssa_opt_parestimate_free)
    int ssa_opt_parestimate(const SSA_Opt *ssa, const int *group, int n_group, SSA_ParEstimate *result);
    void ssa_opt_parestimate_free(SSA_ParEstimate *result);

    // Gap Filling - handle missing values in time series
    // Iteratively fills gaps using SSA reconstruction
    typedef struct
    {
        int iterations;    // Iterations performed
        double final_diff; // Final relative change in gap values
        int converged;     // 1 if converged, 0 if hit max_iter
        int n_gaps;        // Number of gap positions filled
    } SSA_GapFillResult;

    // Iterative gap filling using SSA
    // x: input signal with NaN for missing values (will be modified in place)
    // N: signal length
    // L: window length
    // rank: number of components for reconstruction
    // max_iter: maximum iterations (typical: 10-50)
    // tol: convergence tolerance (typical: 1e-6)
    // result: output statistics (can be NULL if not needed)
    // Returns 0 on success, filled signal in x
    int ssa_opt_gapfill(double *x, int N, int L, int rank, int max_iter, double tol,
                        SSA_GapFillResult *result);

    // Simple gap filling using forecast/backcast
    // Fills each gap by averaging forward forecast and backward forecast
    // Faster but less accurate than iterative method
    int ssa_opt_gapfill_simple(double *x, int N, int L, int rank, SSA_GapFillResult *result);

#ifdef SSA_OPT_IMPLEMENTATION

    static inline int ssa_opt_next_pow2(int n)
    {
        int p = 1;
        while (p < n)
            p <<= 1;
        return p;
    }
    static inline int ssa_opt_min(int a, int b) { return a < b ? a : b; }
    static inline int ssa_opt_max(int a, int b) { return a > b ? a : b; }
    static inline void *ssa_opt_alloc(size_t size) { return mkl_malloc(size, SSA_ALIGN); }
    static inline void ssa_opt_free_ptr(void *ptr) { mkl_free(ptr); }
    static inline double ssa_opt_dot(const double *a, const double *b, int n) { return cblas_ddot(n, a, 1, b, 1); }
    static inline double ssa_opt_nrm2(const double *v, int n) { return cblas_dnrm2(n, v, 1); }
    static inline void ssa_opt_scal(double *v, int n, double s) { cblas_dscal(n, s, v, 1); }
    static inline void ssa_opt_axpy(double *y, const double *x, double a, int n) { cblas_daxpy(n, a, x, 1, y, 1); }
    static inline void ssa_opt_copy(const double *src, double *dst, int n) { cblas_dcopy(n, src, 1, dst, 1); }
    static inline double ssa_opt_normalize(double *v, int n)
    {
        double norm = cblas_dnrm2(n, v, 1);
        if (norm > 1e-12)
            cblas_dscal(n, 1.0 / norm, v, 1);
        return norm;
    }
    static inline void ssa_opt_zero(double *v, int n) { memset(v, 0, n * sizeof(double)); }
    static inline void ssa_opt_complex_mul_r2c(const double *a, const double *b, double *c, int r2c_len) { vzMul(r2c_len, (const MKL_Complex16 *)a, (const MKL_Complex16 *)b, (MKL_Complex16 *)c); }

    // ========================================================================
    // SIMD-optimized helper functions (AVX2)
    // ========================================================================

    // AVX2 reverse copy: dst[i] = src[n-1-i]
    static inline void ssa_opt_reverse_copy(const double *src, double *dst, int n)
    {
        int i = 0;
#if defined(__AVX2__) || defined(__AVX__)
        // Process 4 doubles at a time with AVX
        const int simd_width = 4;
        const int n_simd = (n / simd_width) * simd_width;

        // AVX doesn't have a native reverse, so we load from end and shuffle
        for (; i + simd_width <= n; i += simd_width)
        {
            // Load 4 doubles from end of src (reversed position)
            __m256d v = _mm256_loadu_pd(&src[n - 1 - i - 3]);
            // Reverse the 4 elements within the vector
            v = _mm256_permute4x64_pd(v, 0x1B); // 0x1B = 0,1,2,3 -> 3,2,1,0
            _mm256_storeu_pd(&dst[i], v);
        }
#endif
        // Scalar cleanup
        for (; i < n; i++)
        {
            dst[i] = src[n - 1 - i];
        }
    }

    // AVX2 weighted squared norm: sum(h[t]^2 * w[t])
    static inline double ssa_opt_weighted_norm_sq(const double *h, const double *w, int n)
    {
        double result = 0.0;
        int i = 0;

#if defined(__AVX2__) || defined(__AVX__)
        __m256d sum_vec = _mm256_setzero_pd();

        // Process 4 doubles at a time
        for (; i + 4 <= n; i += 4)
        {
            __m256d h_vec = _mm256_loadu_pd(&h[i]);
            __m256d w_vec = _mm256_loadu_pd(&w[i]);
            __m256d h_sq = _mm256_mul_pd(h_vec, h_vec);
            __m256d prod = _mm256_mul_pd(h_sq, w_vec);
            sum_vec = _mm256_add_pd(sum_vec, prod);
        }

        // Horizontal sum
        __m128d low = _mm256_castpd256_pd128(sum_vec);
        __m128d high = _mm256_extractf128_pd(sum_vec, 1);
        __m128d sum128 = _mm_add_pd(low, high);
        sum128 = _mm_hadd_pd(sum128, sum128);
        result = _mm_cvtsd_f64(sum128);
#endif

        // Scalar cleanup
        for (; i < n; i++)
        {
            result += h[i] * h[i] * w[i];
        }

        return result;
    }

    // AVX2 weighted inner product: sum(a[t] * b[t] * w[t])
    static inline double ssa_opt_weighted_inner(const double *a, const double *b, const double *w, int n)
    {
        double result = 0.0;
        int i = 0;

#if defined(__AVX2__) || defined(__AVX__)
        __m256d sum_vec = _mm256_setzero_pd();

        for (; i + 4 <= n; i += 4)
        {
            __m256d a_vec = _mm256_loadu_pd(&a[i]);
            __m256d b_vec = _mm256_loadu_pd(&b[i]);
            __m256d w_vec = _mm256_loadu_pd(&w[i]);
            __m256d ab = _mm256_mul_pd(a_vec, b_vec);
            __m256d prod = _mm256_mul_pd(ab, w_vec);
            sum_vec = _mm256_add_pd(sum_vec, prod);
        }

        // Horizontal sum
        __m128d low = _mm256_castpd256_pd128(sum_vec);
        __m128d high = _mm256_extractf128_pd(sum_vec, 1);
        __m128d sum128 = _mm_add_pd(low, high);
        sum128 = _mm_hadd_pd(sum128, sum128);
        result = _mm_cvtsd_f64(sum128);
#endif

        // Scalar cleanup
        for (; i < n; i++)
        {
            result += a[i] * b[i] * w[i];
        }

        return result;
    }

    // AVX2 triple weighted computation: computes inner, norm_a_sq, norm_b_sq in one pass
    // Returns: inner = sum(a*b*w), norm_a_sq = sum(a*a*w), norm_b_sq = sum(b*b*w)
    static inline void ssa_opt_weighted_inner3(const double *a, const double *b, const double *w, int n,
                                               double *inner, double *norm_a_sq, double *norm_b_sq)
    {
        double r_inner = 0.0, r_norm_a = 0.0, r_norm_b = 0.0;
        int i = 0;

#if defined(__AVX2__) || defined(__AVX__)
        __m256d sum_inner = _mm256_setzero_pd();
        __m256d sum_norm_a = _mm256_setzero_pd();
        __m256d sum_norm_b = _mm256_setzero_pd();

        for (; i + 4 <= n; i += 4)
        {
            __m256d a_vec = _mm256_loadu_pd(&a[i]);
            __m256d b_vec = _mm256_loadu_pd(&b[i]);
            __m256d w_vec = _mm256_loadu_pd(&w[i]);

            // a*b*w
            __m256d ab = _mm256_mul_pd(a_vec, b_vec);
            sum_inner = _mm256_add_pd(sum_inner, _mm256_mul_pd(ab, w_vec));

            // a*a*w
            __m256d aa = _mm256_mul_pd(a_vec, a_vec);
            sum_norm_a = _mm256_add_pd(sum_norm_a, _mm256_mul_pd(aa, w_vec));

            // b*b*w
            __m256d bb = _mm256_mul_pd(b_vec, b_vec);
            sum_norm_b = _mm256_add_pd(sum_norm_b, _mm256_mul_pd(bb, w_vec));
        }

        // Horizontal sums
        {
            __m128d low = _mm256_castpd256_pd128(sum_inner);
            __m128d high = _mm256_extractf128_pd(sum_inner, 1);
            __m128d sum128 = _mm_add_pd(low, high);
            sum128 = _mm_hadd_pd(sum128, sum128);
            r_inner = _mm_cvtsd_f64(sum128);
        }
        {
            __m128d low = _mm256_castpd256_pd128(sum_norm_a);
            __m128d high = _mm256_extractf128_pd(sum_norm_a, 1);
            __m128d sum128 = _mm_add_pd(low, high);
            sum128 = _mm_hadd_pd(sum128, sum128);
            r_norm_a = _mm_cvtsd_f64(sum128);
        }
        {
            __m128d low = _mm256_castpd256_pd128(sum_norm_b);
            __m128d high = _mm256_extractf128_pd(sum_norm_b, 1);
            __m128d sum128 = _mm_add_pd(low, high);
            sum128 = _mm_hadd_pd(sum128, sum128);
            r_norm_b = _mm_cvtsd_f64(sum128);
        }
#endif

        // Scalar cleanup
        for (; i < n; i++)
        {
            double wi = w[i];
            r_inner += a[i] * b[i] * wi;
            r_norm_a += a[i] * a[i] * wi;
            r_norm_b += b[i] * b[i] * wi;
        }

        *inner = r_inner;
        *norm_a_sq = r_norm_a;
        *norm_b_sq = r_norm_b;
    }

    // AVX2 scale and weight: dst[t] = scale * src[t] * w[t]
    static inline void ssa_opt_scale_weighted(const double *src, const double *w, double scale, double *dst, int n)
    {
        int i = 0;

#if defined(__AVX2__) || defined(__AVX__)
        __m256d scale_vec = _mm256_set1_pd(scale);

        for (; i + 4 <= n; i += 4)
        {
            __m256d s_vec = _mm256_loadu_pd(&src[i]);
            __m256d w_vec = _mm256_loadu_pd(&w[i]);
            __m256d prod = _mm256_mul_pd(_mm256_mul_pd(s_vec, w_vec), scale_vec);
            _mm256_storeu_pd(&dst[i], prod);
        }
#endif

        // Scalar cleanup
        for (; i < n; i++)
        {
            dst[i] = scale * src[i] * w[i];
        }
    }

    /**
     * SIMD Complex multiply-accumulate: acc += u * v (fused, no intermediate storage)
     * Replaces: ssa_opt_complex_mul_r2c() + cblas_daxpy() with single fused operation
     */
    static inline void ssa_opt_complex_mul_acc(
        const double *__restrict u_fft,
        const double *__restrict v_fft,
        double *__restrict acc,
        int n)
    {
#ifdef SSA_SIMD_AVX2
        int i = 0;

        // Main loop: process 2 complex numbers (4 doubles) per iteration
        for (; i + 1 < n; i += 2)
        {
            __m256d u = _mm256_loadu_pd(&u_fft[2 * i]);
            __m256d v = _mm256_loadu_pd(&v_fft[2 * i]);
            __m256d a = _mm256_loadu_pd(&acc[2 * i]);

            __m256d u_swap = _mm256_permute_pd(u, 0b0101);
            __m256d v_re = _mm256_unpacklo_pd(v, v);
            __m256d v_im = _mm256_unpackhi_pd(v, v);

            __m256d prod1 = _mm256_mul_pd(u, v_re);
            __m256d prod2 = _mm256_mul_pd(u_swap, v_im);
            __m256d result = _mm256_addsub_pd(prod1, prod2);

            a = _mm256_add_pd(a, result);
            _mm256_storeu_pd(&acc[2 * i], a);
        }

        // Scalar cleanup
        for (; i < n; i++)
        {
            double u_re = u_fft[2 * i];
            double u_im = u_fft[2 * i + 1];
            double v_re = v_fft[2 * i];
            double v_im = v_fft[2 * i + 1];
            acc[2 * i] += u_re * v_re - u_im * v_im;
            acc[2 * i + 1] += u_re * v_im + u_im * v_re;
        }

#elif defined(SSA_SIMD_SSE2)
        for (int i = 0; i < n; i++)
        {
            __m128d u = _mm_loadu_pd(&u_fft[2 * i]);
            __m128d v = _mm_loadu_pd(&v_fft[2 * i]);
            __m128d a = _mm_loadu_pd(&acc[2 * i]);

            __m128d u_swap = _mm_shuffle_pd(u, u, 0b01);
            __m128d v_re = _mm_unpacklo_pd(v, v);
            __m128d v_im = _mm_unpackhi_pd(v, v);

            __m128d prod1 = _mm_mul_pd(u, v_re);
            __m128d prod2 = _mm_mul_pd(u_swap, v_im);

            __m128d sign = _mm_set_pd(1.0, -1.0);
            prod2 = _mm_mul_pd(prod2, sign);
            __m128d result = _mm_add_pd(prod1, prod2);

            a = _mm_add_pd(a, result);
            _mm_storeu_pd(&acc[2 * i], a);
        }

#else
        // Scalar fallback
        for (int i = 0; i < n; i++)
        {
            double u_re = u_fft[2 * i];
            double u_im = u_fft[2 * i + 1];
            double v_re = v_fft[2 * i];
            double v_im = v_fft[2 * i + 1];
            acc[2 * i] += u_re * v_re - u_im * v_im;
            acc[2 * i + 1] += u_re * v_im + u_im * v_re;
        }
#endif
    }

    // ========================================================================
    // End SIMD helpers
    // ========================================================================

    // Hankel matvec via R2C FFT: y = H·v where H[i,j] = x[i+j]
    // Exploits: H·v = conv(x, flip(v))[K-1 : K-1+L]
    // See docs/HANKEL_MATVEC.md for derivation
    static void ssa_opt_hankel_matvec(SSA_Opt *ssa, const double *v, double *y)
    {
        int K = ssa->K, L = ssa->L, fft_len = ssa->fft_len, r2c_len = ssa->r2c_len;
        ssa_opt_zero(ssa->ws_real, fft_len);
        ssa_opt_reverse_copy(v, ssa->ws_real, K);                                       // flip(v) for correlation
        DftiComputeForward(ssa->fft_r2c, ssa->ws_real, ssa->ws_complex);                // FFT(flip(v))
        ssa_opt_complex_mul_r2c(ssa->fft_x, ssa->ws_complex, ssa->ws_complex, r2c_len); // FFT(x) ⊙ FFT(flip(v))
        DftiComputeBackward(ssa->fft_c2r, ssa->ws_complex, ssa->ws_real);               // IFFT → conv result
        memcpy(y, ssa->ws_real + (K - 1), L * sizeof(double));                          // extract [K-1, K-1+L)
    }

    // Adjoint Hankel matvec: z = Hᵀ·u
    // Exploits: Hᵀ·u = conv(x, flip(u))[L-1 : L-1+K]
    static void ssa_opt_hankel_matvec_T(SSA_Opt *ssa, const double *u, double *y)
    {
        int K = ssa->K, L = ssa->L, fft_len = ssa->fft_len, r2c_len = ssa->r2c_len;
        ssa_opt_zero(ssa->ws_real, fft_len);
        ssa_opt_reverse_copy(u, ssa->ws_real, L);
        DftiComputeForward(ssa->fft_r2c, ssa->ws_real, ssa->ws_complex);
        ssa_opt_complex_mul_r2c(ssa->fft_x, ssa->ws_complex, ssa->ws_complex, r2c_len);
        DftiComputeBackward(ssa->fft_c2r, ssa->ws_complex, ssa->ws_real);
        memcpy(y, ssa->ws_real + (L - 1), K * sizeof(double)); // extract [L-1, L-1+K)
    }

    // Block Hankel matvec: Y = H·V where V is K×b, Y is L×b
    // Batches up to SSA_BATCH_SIZE vectors per MKL FFT call for efficiency
    static void ssa_opt_hankel_matvec_block(SSA_Opt *ssa, const double *V_block, double *Y_block, int b)
    {
        int K = ssa->K, L = ssa->L, fft_len = ssa->fft_len, r2c_len = ssa->r2c_len;
        double *ws_real = ssa->ws_batch_real, *ws_complex = ssa->ws_batch_complex;
        int col = 0;
        while (col < b)
        {
            int batch_count = ssa_opt_min(SSA_BATCH_SIZE, b - col);
            if (batch_count < 2)
            { // Fallback for tiny batches
                for (int i = 0; i < batch_count; i++)
                    ssa_opt_hankel_matvec(ssa, &V_block[(col + i) * K], &Y_block[(col + i) * L]);
                col += batch_count;
                continue;
            }
            // Pack reversed vectors contiguously for batched FFT
            memset(ws_real, 0, SSA_BATCH_SIZE * fft_len * sizeof(double));
            for (int i = 0; i < batch_count; i++)
            {
                const double *v = &V_block[(col + i) * K];
                double *dst = ws_real + i * fft_len;
                for (int j = 0; j < K; j++)
                    dst[j] = v[K - 1 - j];
            }
            DftiComputeForward(ssa->fft_r2c_batch, ws_real, ws_complex); // Batched FFT (single MKL call)
            for (int i = 0; i < batch_count; i++)
            {
                double *fft_v = ws_complex + i * 2 * r2c_len;
                ssa_opt_complex_mul_r2c(ssa->fft_x, fft_v, fft_v, r2c_len);
            }
            DftiComputeBackward(ssa->fft_c2r_batch, ws_complex, ws_real); // Batched IFFT
            for (int i = 0; i < batch_count; i++)
                memcpy(&Y_block[(col + i) * L], ws_real + i * fft_len + (K - 1), L * sizeof(double));
            col += batch_count;
        }
    }

    // Block adjoint Hankel matvec: Z = Hᵀ·U where U is L×b, Z is K×b
    static void ssa_opt_hankel_matvec_T_block(SSA_Opt *ssa, const double *U_block, double *Y_block, int b)
    {
        int K = ssa->K, L = ssa->L, fft_len = ssa->fft_len, r2c_len = ssa->r2c_len;
        double *ws_real = ssa->ws_batch_real, *ws_complex = ssa->ws_batch_complex;
        int col = 0;
        while (col < b)
        {
            int batch_count = ssa_opt_min(SSA_BATCH_SIZE, b - col);
            if (batch_count < 2)
            {
                for (int i = 0; i < batch_count; i++)
                    ssa_opt_hankel_matvec_T(ssa, &U_block[(col + i) * L], &Y_block[(col + i) * K]);
                col += batch_count;
                continue;
            }
            memset(ws_real, 0, SSA_BATCH_SIZE * fft_len * sizeof(double));
            for (int i = 0; i < batch_count; i++)
            {
                const double *u = &U_block[(col + i) * L];
                double *dst = ws_real + i * fft_len;
                for (int j = 0; j < L; j++)
                    dst[j] = u[L - 1 - j];
            }
            DftiComputeForward(ssa->fft_r2c_batch, ws_real, ws_complex);
            for (int i = 0; i < batch_count; i++)
            {
                double *fft_u = ws_complex + i * 2 * r2c_len;
                ssa_opt_complex_mul_r2c(ssa->fft_x, fft_u, fft_u, r2c_len);
            }
            DftiComputeBackward(ssa->fft_c2r_batch, ws_complex, ws_real);
            for (int i = 0; i < batch_count; i++)
                memcpy(&Y_block[(col + i) * K], ws_real + i * fft_len + (L - 1), K * sizeof(double));
            col += batch_count;
        }
    }

    /**
     * Initialize SSA structure with signal and parameters.
     *
     * This function sets up all data structures needed for FFT-accelerated SSA:
     * - Computes dimensions and FFT length
     * - Allocates all workspaces
     * - Creates MKL FFT descriptors (forward, inverse, batched)
     * - Pre-computes FFT of signal (reused in every Hankel matvec)
     * - Pre-computes diagonal averaging weights
     * - Creates per-thread FFT descriptor pool for OpenMP (avoids expensive
     *   DftiCommitDescriptor calls in hot loops like ssa_opt_cache_ffts)
     *
     * After init, call ssa_opt_decompose*() then ssa_opt_reconstruct().
     *
     * @param ssa   Output structure (caller allocates, we fill in)
     * @param x     Input signal, length N
     * @param N     Signal length
     * @param L     Window length (embedding dimension), typically N/4 to N/2
     * @return      0 on success, -1 on failure
     */
    int ssa_opt_init(SSA_Opt *ssa, const double *x, int N, int L)
    {
        // === Parameter Validation ===
        // N >= 4: need at least a few samples for meaningful SSA
        // L >= 2: window must have at least 2 elements
        // L <= N-1: need K = N-L+1 >= 2 lagged copies
        if (!ssa || !x || N < 4 || L < 2 || L > N - 1)
            return -1;

        // Zero-initialize entire structure (sets all pointers to NULL, flags to false)
        memset(ssa, 0, sizeof(SSA_Opt));

        // === Compute Dimensions ===
        ssa->N = N;
        ssa->L = L;
        ssa->K = N - L + 1; // Number of lagged windows (columns of trajectory matrix)

        // FFT length for convolution:
        // Hankel matvec y = H·v is computed as convolution of x (length N) and v (length K)
        // Convolution result has length N + K - 1
        // Round up to power of 2 for FFT efficiency (radix-2 is fastest)
        int conv_len = N + ssa->K - 1;
        int fft_n = ssa_opt_next_pow2(conv_len);
        ssa->fft_len = fft_n;

        // Real-to-Complex FFT output length:
        // For real input of length N, FFT output has Hermitian symmetry: X[k] = conj(X[N-k])
        // Only need to store first N/2+1 complex values (the rest are redundant)
        // This saves ~50% memory compared to full complex-to-complex FFT
        ssa->r2c_len = fft_n / 2 + 1;

        // === Allocate Core Workspaces ===
        // All allocations use 64-byte alignment for AVX-512 compatibility

        // ws_real: Real-valued input buffer for forward FFT
        // Used to hold zero-padded, reversed vectors before FFT
        ssa->ws_real = (double *)ssa_opt_alloc(fft_n * sizeof(double));

        // ws_complex: Complex output from forward FFT / input to inverse FFT
        // Stored as interleaved [re, im, re, im, ...], length 2 * r2c_len doubles
        ssa->ws_complex = (double *)ssa_opt_alloc(2 * ssa->r2c_len * sizeof(double));

        // ws_real2: Secondary real workspace for adjoint matvec (Hᵀ·u)
        // Needed because forward and adjoint use different extraction offsets
        ssa->ws_real2 = (double *)ssa_opt_alloc(fft_n * sizeof(double));

        // ws_batch_real/complex: Batched FFT workspaces
        // Process up to SSA_BATCH_SIZE (default 32) vectors in one MKL call
        // Reduces function call overhead, improves cache utilization
        ssa->ws_batch_real = (double *)ssa_opt_alloc(SSA_BATCH_SIZE * fft_n * sizeof(double));
        ssa->ws_batch_complex = (double *)ssa_opt_alloc(SSA_BATCH_SIZE * 2 * ssa->r2c_len * sizeof(double));

        // fft_x: Pre-computed FFT of the input signal
        // This is the key optimization: signal doesn't change during decomposition,
        // so we compute FFT(x) once here and reuse it in every Hankel matvec
        // Saves one FFT per matvec operation
        ssa->fft_x = (double *)ssa_opt_alloc(2 * ssa->r2c_len * sizeof(double));

        // Check all allocations succeeded
        if (!ssa->ws_real || !ssa->ws_complex || !ssa->ws_real2 ||
            !ssa->ws_batch_real || !ssa->ws_batch_complex || !ssa->fft_x)
        {
            ssa_opt_free(ssa); // Clean up partial allocations
            return -1;
        }

        // === Create MKL FFT Descriptors ===
        // MKL uses "descriptors" that encode FFT configuration (size, placement, scaling)
        // Creating descriptor is expensive, so we do it once and reuse

        MKL_LONG status;

        // --- Forward R2C (single vector) ---
        // Transforms real input to complex output
        status = DftiCreateDescriptor(&ssa->fft_r2c, DFTI_DOUBLE, DFTI_REAL, 1, fft_n);
        if (status != 0)
        {
            ssa_opt_free(ssa);
            return -1;
        }
        // DFTI_NOT_INPLACE: input and output are separate buffers (safer, no aliasing issues)
        DftiSetValue(ssa->fft_r2c, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
        // DFTI_COMPLEX_COMPLEX: store output as [re, im] pairs (not split re/im arrays)
        DftiSetValue(ssa->fft_r2c, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
        if (DftiCommitDescriptor(ssa->fft_r2c) != 0)
        {
            ssa_opt_free(ssa);
            return -1;
        }

        // --- Inverse C2R (single vector) ---
        // Transforms complex input back to real output
        status = DftiCreateDescriptor(&ssa->fft_c2r, DFTI_DOUBLE, DFTI_REAL, 1, fft_n);
        if (status != 0)
        {
            ssa_opt_free(ssa);
            return -1;
        }
        DftiSetValue(ssa->fft_c2r, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
        DftiSetValue(ssa->fft_c2r, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
        // DFTI_BACKWARD_SCALE: MKL's FFT is unnormalized by default (IFFT(FFT(x)) = N*x)
        // We apply 1/N scaling to inverse so IFFT(FFT(x)) = x
        DftiSetValue(ssa->fft_c2r, DFTI_BACKWARD_SCALE, 1.0 / fft_n);
        if (DftiCommitDescriptor(ssa->fft_c2r) != 0)
        {
            ssa_opt_free(ssa);
            return -1;
        }

        // --- Forward R2C Batched ---
        // Same as fft_r2c but processes SSA_BATCH_SIZE vectors in one call
        // Used by block power iteration and randomized SVD
        status = DftiCreateDescriptor(&ssa->fft_r2c_batch, DFTI_DOUBLE, DFTI_REAL, 1, fft_n);
        if (status != 0)
        {
            ssa_opt_free(ssa);
            return -1;
        }
        DftiSetValue(ssa->fft_r2c_batch, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
        DftiSetValue(ssa->fft_r2c_batch, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
        // NUMBER_OF_TRANSFORMS: how many FFTs to compute in parallel
        DftiSetValue(ssa->fft_r2c_batch, DFTI_NUMBER_OF_TRANSFORMS, (MKL_LONG)SSA_BATCH_SIZE);
        // INPUT_DISTANCE: stride between consecutive input vectors (each is fft_n doubles)
        DftiSetValue(ssa->fft_r2c_batch, DFTI_INPUT_DISTANCE, (MKL_LONG)fft_n);
        // OUTPUT_DISTANCE: stride between consecutive output vectors (each is r2c_len complex = 2*r2c_len doubles)
        // Note: MKL measures in complex units for complex data, so we use r2c_len not 2*r2c_len
        DftiSetValue(ssa->fft_r2c_batch, DFTI_OUTPUT_DISTANCE, (MKL_LONG)ssa->r2c_len);
        if (DftiCommitDescriptor(ssa->fft_r2c_batch) != 0)
        {
            ssa_opt_free(ssa);
            return -1;
        }

        // --- Inverse C2R Batched ---
        status = DftiCreateDescriptor(&ssa->fft_c2r_batch, DFTI_DOUBLE, DFTI_REAL, 1, fft_n);
        if (status != 0)
        {
            ssa_opt_free(ssa);
            return -1;
        }
        DftiSetValue(ssa->fft_c2r_batch, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
        DftiSetValue(ssa->fft_c2r_batch, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
        DftiSetValue(ssa->fft_c2r_batch, DFTI_BACKWARD_SCALE, 1.0 / fft_n);
        DftiSetValue(ssa->fft_c2r_batch, DFTI_NUMBER_OF_TRANSFORMS, (MKL_LONG)SSA_BATCH_SIZE);
        // For inverse: input is complex (r2c_len), output is real (fft_n)
        DftiSetValue(ssa->fft_c2r_batch, DFTI_INPUT_DISTANCE, (MKL_LONG)ssa->r2c_len);
        DftiSetValue(ssa->fft_c2r_batch, DFTI_OUTPUT_DISTANCE, (MKL_LONG)fft_n);
        if (DftiCommitDescriptor(ssa->fft_c2r_batch) != 0)
        {
            ssa_opt_free(ssa);
            return -1;
        }

        // === Initialize Per-Thread FFT Descriptor Pool (OpenMP Optimization) ===
        // DftiCommitDescriptor is extremely expensive (computes trig tables, factorizes length).
        // By creating one descriptor per thread at init time, we avoid this cost in hot loops
        // like ssa_opt_cache_ffts() which previously created/destroyed descriptors per call.
#ifdef _OPENMP
        {
            int max_threads = omp_get_max_threads();
            ssa->thread_pool_size = max_threads;
            ssa->thread_fft_pool = (DFTI_DESCRIPTOR_HANDLE *)malloc(max_threads * sizeof(DFTI_DESCRIPTOR_HANDLE));

            if (!ssa->thread_fft_pool)
            {
                ssa_opt_free(ssa);
                return -1;
            }

            // Initialize all handles to NULL first (for safe cleanup on partial failure)
            for (int t = 0; t < max_threads; t++)
            {
                ssa->thread_fft_pool[t] = NULL;
            }

            // Create and commit each thread's descriptor
            for (int t = 0; t < max_threads; t++)
            {
                status = DftiCreateDescriptor(&ssa->thread_fft_pool[t], DFTI_DOUBLE, DFTI_REAL, 1, fft_n);
                if (status != 0)
                {
                    ssa_opt_free(ssa);
                    return -1;
                }
                DftiSetValue(ssa->thread_fft_pool[t], DFTI_PLACEMENT, DFTI_NOT_INPLACE);
                DftiSetValue(ssa->thread_fft_pool[t], DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
                status = DftiCommitDescriptor(ssa->thread_fft_pool[t]);
                if (status != 0)
                {
                    ssa_opt_free(ssa);
                    return -1;
                }
            }
        }
#endif

        // === Initialize Random Number Generator ===
        // Used by randomized SVD for generating Gaussian random matrices
        // MT2203: Mersenne Twister variant optimized for parallel streams
        // Seed 42: fixed seed for reproducibility (can be changed via API if needed)
        if (vslNewStream(&ssa->rng, VSL_BRNG_MT2203, 42) != VSL_STATUS_OK)
        {
            ssa_opt_free(ssa);
            return -1;
        }

        // === Pre-compute FFT of Signal ===
        // This is computed once and reused in every Hankel matvec:
        //   y = H·v = IFFT(FFT(x) ⊙ FFT(flip(v)))
        // By storing FFT(x), we save one FFT per matvec
        ssa_opt_zero(ssa->ws_real, fft_n);           // Zero-pad to FFT length
        memcpy(ssa->ws_real, x, N * sizeof(double)); // Copy signal
        DftiComputeForward(ssa->fft_r2c, ssa->ws_real, ssa->fft_x);

        // === Pre-compute Diagonal Averaging Weights ===
        // Reconstruction uses diagonal averaging: each signal position t receives
        // contributions from multiple (i,j) pairs where i+j = t
        // The count of such pairs is: min(t+1, L, K, N-t)
        // We store 1/count for fast division during reconstruction
        ssa->inv_diag_count = (double *)ssa_opt_alloc(N * sizeof(double));
        if (!ssa->inv_diag_count)
        {
            ssa_opt_free(ssa);
            return -1;
        }
        for (int t = 0; t < N; t++)
        {
            // Count of anti-diagonal elements contributing to position t:
            // - Can't exceed t+1 (not enough earlier elements)
            // - Can't exceed L (window length)
            // - Can't exceed K (number of lagged copies)
            // - Can't exceed N-t (not enough later elements)
            int count = ssa_opt_min(ssa_opt_min(t + 1, L), ssa_opt_min(ssa->K, N - t));
            ssa->inv_diag_count[t] = (count > 0) ? 1.0 / count : 0.0;
        }

        // === Mark as Initialized ===
        ssa->initialized = true;
        return 0;
    }

    void ssa_opt_free(SSA_Opt *ssa)
    {
        if (!ssa)
            return;

        // Free MKL FFT descriptors
        if (ssa->fft_r2c)
            DftiFreeDescriptor(&ssa->fft_r2c);
        if (ssa->fft_c2r)
            DftiFreeDescriptor(&ssa->fft_c2r);
        if (ssa->fft_r2c_batch)
            DftiFreeDescriptor(&ssa->fft_r2c_batch);
        if (ssa->fft_c2r_batch)
            DftiFreeDescriptor(&ssa->fft_c2r_batch);
        if (ssa->fft_c2r_wcorr)
            DftiFreeDescriptor(&ssa->fft_c2r_wcorr);

        // Free per-thread FFT descriptor pool
#ifdef _OPENMP
        if (ssa->thread_fft_pool)
        {
            for (int t = 0; t < ssa->thread_pool_size; t++)
            {
                if (ssa->thread_fft_pool[t])
                    DftiFreeDescriptor(&ssa->thread_fft_pool[t]);
            }
            free(ssa->thread_fft_pool);
            ssa->thread_fft_pool = NULL;
            ssa->thread_pool_size = 0;
        }
#endif

        if (ssa->rng)
            vslDeleteStream(&ssa->rng);
        ssa_opt_free_ptr(ssa->fft_x);
        ssa_opt_free_ptr(ssa->ws_real);
        ssa_opt_free_ptr(ssa->ws_complex);
        ssa_opt_free_ptr(ssa->ws_real2);
        ssa_opt_free_ptr(ssa->ws_proj);
        ssa_opt_free_ptr(ssa->ws_batch_real);
        ssa_opt_free_ptr(ssa->ws_batch_complex);
        ssa_opt_free_ptr(ssa->U);
        ssa_opt_free_ptr(ssa->V);
        ssa_opt_free_ptr(ssa->sigma);
        ssa_opt_free_ptr(ssa->eigenvalues);
        ssa_opt_free_ptr(ssa->inv_diag_count);
        ssa_opt_free_ptr(ssa->U_fft);
        ssa_opt_free_ptr(ssa->V_fft);
        ssa_opt_free_ptr(ssa->wcorr_ws_complex);
        ssa_opt_free_ptr(ssa->wcorr_h);
        ssa_opt_free_ptr(ssa->wcorr_G);
        ssa_opt_free_ptr(ssa->wcorr_sqrt_inv_c);
        // Decompose workspace
        ssa_opt_free_ptr(ssa->decomp_Omega);
        ssa_opt_free_ptr(ssa->decomp_Y);
        ssa_opt_free_ptr(ssa->decomp_Q);
        ssa_opt_free_ptr(ssa->decomp_B);
        ssa_opt_free_ptr(ssa->decomp_tau);
        ssa_opt_free_ptr(ssa->decomp_B_left);
        ssa_opt_free_ptr(ssa->decomp_B_right_T);
        ssa_opt_free_ptr(ssa->decomp_S);
        ssa_opt_free_ptr(ssa->decomp_work);
        ssa_opt_free_ptr(ssa->decomp_iwork);
        memset(ssa, 0, sizeof(SSA_Opt));
    }

    void ssa_opt_free_prepared(SSA_Opt *ssa)
    {
        if (!ssa)
            return;
        ssa_opt_free_ptr(ssa->decomp_Omega);
        ssa->decomp_Omega = NULL;
        ssa_opt_free_ptr(ssa->decomp_Y);
        ssa->decomp_Y = NULL;
        ssa_opt_free_ptr(ssa->decomp_Q);
        ssa->decomp_Q = NULL;
        ssa_opt_free_ptr(ssa->decomp_B);
        ssa->decomp_B = NULL;
        ssa_opt_free_ptr(ssa->decomp_tau);
        ssa->decomp_tau = NULL;
        ssa_opt_free_ptr(ssa->decomp_B_left);
        ssa->decomp_B_left = NULL;
        ssa_opt_free_ptr(ssa->decomp_B_right_T);
        ssa->decomp_B_right_T = NULL;
        ssa_opt_free_ptr(ssa->decomp_S);
        ssa->decomp_S = NULL;
        ssa_opt_free_ptr(ssa->decomp_work);
        ssa->decomp_work = NULL;
        ssa_opt_free_ptr(ssa->decomp_iwork);
        ssa->decomp_iwork = NULL;
        ssa->prepared_kp = 0;
        ssa->prepared_lwork = 0;
        ssa->prepared = false;
    }

    /**
     * Pre-allocate all workspace for randomized SVD decomposition.
     *
     * This enables the "malloc-free hot path" - after calling prepare(),
     * subsequent calls to ssa_opt_decompose_randomized() do ZERO memory
     * allocations. This is critical for:
     *   - Real-time trading systems (no allocation jitter)
     *   - Streaming applications (deterministic latency)
     *   - High-frequency repeated decomposition
     *
     * The workspace is sized for the randomized SVD algorithm:
     *   1. Random projection: Y = H·Ω  where Ω is K×(k+p) Gaussian
     *   2. QR factorization: Q = orth(Y)
     *   3. Small projection: B = Hᵀ·Q
     *   4. Small SVD: B = U_B·Σ·V_Bᵀ via LAPACK DGESDD
     *   5. Recovery: U = Q·V_B
     *
     * @param ssa          Initialized SSA structure
     * @param max_k        Maximum number of components to extract
     * @param oversampling Extra random vectors for accuracy (default 8 if <=0)
     * @return             0 on success, -1 on failure
     *
     * Usage:
     *   ssa_opt_init(&ssa, x, N, L);
     *   ssa_opt_prepare(&ssa, 30, 8);  // Prepare for up to k=30 components
     *
     *   // Streaming loop - no malloc in this loop
     *   while (new_data_available()) {
     *       ssa_opt_update_signal(&ssa, new_x);
     *       ssa_opt_decompose_randomized(&ssa, 30, 8);  // Uses pre-allocated buffers
     *       ssa_opt_reconstruct(&ssa, group, n, output);
     *   }
     */
    int ssa_opt_prepare(SSA_Opt *ssa, int max_k, int oversampling)
    {
        // === Validation ===
        if (!ssa || !ssa->initialized || max_k < 1)
            return -1;

        // Free any previous preparation (allows re-prepare with different k)
        ssa_opt_free_prepared(ssa);

        int L = ssa->L, K = ssa->K;

        // === Compute Dimensions ===
        // Oversampling p: extra random vectors beyond k for numerical stability
        // Theory says p=5-10 gives near-optimal accuracy; default to 8
        int p = (oversampling <= 0) ? 8 : oversampling;

        // Total random vectors: k + p
        // But can't exceed matrix rank = min(L, K)
        int kp = max_k + p;
        kp = ssa_opt_min(kp, ssa_opt_min(L, K));

        // Actual k we can compute (may be less than requested if kp was clamped)
        int actual_k = ssa_opt_min(max_k, kp);

        // === Allocate Randomized SVD Workspace ===
        // These buffers hold intermediate results during decomposition

        // decomp_Omega: Random Gaussian matrix, K × (k+p)
        // Each column is an independent sample from N(0,1)
        // Used to probe the column space of H via Y = H·Ω
        ssa->decomp_Omega = (double *)ssa_opt_alloc(K * kp * sizeof(double));

        // decomp_Y: Random projection result, L × (k+p)
        // Y = H·Ω captures (with high probability) the top-k column space of H
        // Computed via batched FFT-accelerated Hankel matvec
        ssa->decomp_Y = (double *)ssa_opt_alloc(L * kp * sizeof(double));

        // decomp_Q: Orthonormal basis from QR(Y), L × (k+p)
        // Q spans the same subspace as Y but with orthonormal columns
        // Computed via Householder QR (LAPACK DGEQRF + DORGQR)
        ssa->decomp_Q = (double *)ssa_opt_alloc(L * kp * sizeof(double));

        // decomp_B: Projected small matrix, K × (k+p)
        // B = Hᵀ·Q projects H onto the subspace Q
        // B has same top singular values as H (key insight of algorithm)
        ssa->decomp_B = (double *)ssa_opt_alloc(K * kp * sizeof(double));

        // decomp_tau: Householder reflector coefficients from QR, length (k+p)
        // Internal to LAPACK's QR factorization
        ssa->decomp_tau = (double *)ssa_opt_alloc(kp * sizeof(double));

        // decomp_B_left: Left singular vectors of B, K × (k+p)
        // Output from DGESDD (divide-and-conquer SVD)
        ssa->decomp_B_left = (double *)ssa_opt_alloc(K * kp * sizeof(double));

        // decomp_B_right_T: Right singular vectors of B (transposed), (k+p) × (k+p)
        // Used to rotate Q back to original coordinates: U = Q · V_B
        ssa->decomp_B_right_T = (double *)ssa_opt_alloc(kp * kp * sizeof(double));

        // decomp_S: Singular values of B, length (k+p)
        // These are the final singular values (same as H's top singular values)
        ssa->decomp_S = (double *)ssa_opt_alloc(kp * sizeof(double));

        // decomp_iwork: Integer workspace for DGESDD, length 8*(k+p)
        // LAPACK's divide-and-conquer SVD needs integer scratch space
        ssa->decomp_iwork = (int *)ssa_opt_alloc(8 * kp * sizeof(int));

        // === Query LAPACK Workspace Size ===
        // LAPACK routines need workspace whose size depends on problem dimensions
        // We query the optimal size (lwork = -1 triggers query mode)
        double work_query;
        int lwork = -1;
        LAPACKE_dgesdd_work(LAPACK_COL_MAJOR, 'S',     // 'S' = economy SVD
                            K, kp,                     // Matrix dimensions
                            ssa->decomp_B, K,          // Input matrix (will be overwritten)
                            ssa->decomp_S,             // Singular values output
                            ssa->decomp_B_left, K,     // Left singular vectors
                            ssa->decomp_B_right_T, kp, // Right singular vectors (transposed)
                            &work_query, lwork,        // Query mode: returns optimal size in work_query
                            ssa->decomp_iwork);

        // Allocate optimal workspace (+1 for safety margin)
        lwork = (int)work_query + 1;
        ssa->decomp_work = (double *)ssa_opt_alloc(lwork * sizeof(double));
        ssa->prepared_lwork = lwork;

        // === Pre-allocate Result Arrays ===
        // U, V, sigma, eigenvalues are the final outputs
        // By allocating here, decompose_randomized() needs zero allocations

        // Free any existing result arrays (from previous decomposition)
        ssa_opt_free_ptr(ssa->U);
        ssa_opt_free_ptr(ssa->V);
        ssa_opt_free_ptr(ssa->sigma);
        ssa_opt_free_ptr(ssa->eigenvalues);

        // U: Left singular vectors of H, L × k
        // Each column U[:,i] is an "empirical orthogonal function"
        ssa->U = (double *)ssa_opt_alloc(L * actual_k * sizeof(double));

        // V: Right singular vectors of H, K × k
        // Each column V[:,i] is the corresponding "principal component"
        ssa->V = (double *)ssa_opt_alloc(K * actual_k * sizeof(double));

        // sigma: Singular values, length k
        // σ₀ ≥ σ₁ ≥ ... ≥ σₖ₋₁ (descending order)
        ssa->sigma = (double *)ssa_opt_alloc(actual_k * sizeof(double));

        // eigenvalues: λᵢ = σᵢ², length k
        // Eigenvalues of HᵀH (variance captured by each component)
        ssa->eigenvalues = (double *)ssa_opt_alloc(actual_k * sizeof(double));

        // Set n_components now so caller knows capacity
        ssa->n_components = actual_k;

        // === Verify All Allocations ===
        if (!ssa->decomp_Omega || !ssa->decomp_Y || !ssa->decomp_Q || !ssa->decomp_B ||
            !ssa->decomp_tau || !ssa->decomp_B_left || !ssa->decomp_B_right_T ||
            !ssa->decomp_S || !ssa->decomp_iwork || !ssa->decomp_work ||
            !ssa->U || !ssa->V || !ssa->sigma || !ssa->eigenvalues)
        {
            ssa_opt_free_prepared(ssa); // Clean up partial allocations
            return -1;
        }

        // === Mark as Prepared ===
        ssa->prepared_kp = kp; // Remember capacity for validation in decompose
        ssa->prepared = true;
        return 0;
    }

    int ssa_opt_update_signal(SSA_Opt *ssa, const double *new_x)
    {
        if (!ssa || !ssa->initialized || !new_x)
            return -1;
        int N = ssa->N, fft_len = ssa->fft_len;
        // Invalidate cached FFTs and decomposition
        ssa_opt_free_cached_ffts(ssa);
        ssa->decomposed = false;
        // Update FFT(x) - just memcpy + one FFT
        ssa_opt_zero(ssa->ws_real, fft_len);
        memcpy(ssa->ws_real, new_x, N * sizeof(double));
        DftiComputeForward(ssa->fft_r2c, ssa->ws_real, ssa->fft_x);
        return 0;
    }

    int ssa_opt_decompose(SSA_Opt *ssa, int k, int max_iter)
    {
        if (!ssa || !ssa->initialized || k < 1)
            return -1;
        ssa_opt_free_cached_ffts(ssa);
        int L = ssa->L, K = ssa->K;
        k = ssa_opt_min(k, ssa_opt_min(L, K));
        ssa->U = (double *)ssa_opt_alloc(L * k * sizeof(double));
        ssa->V = (double *)ssa_opt_alloc(K * k * sizeof(double));
        ssa->sigma = (double *)ssa_opt_alloc(k * sizeof(double));
        ssa->eigenvalues = (double *)ssa_opt_alloc(k * sizeof(double));
        if (!ssa->U || !ssa->V || !ssa->sigma || !ssa->eigenvalues)
            return -1;
        ssa->n_components = k;
        double *u = (double *)ssa_opt_alloc(L * sizeof(double));
        double *v = (double *)ssa_opt_alloc(K * sizeof(double));
        double *v_new = (double *)ssa_opt_alloc(K * sizeof(double));
        ssa->ws_proj = (double *)ssa_opt_alloc(k * sizeof(double));
        if (!u || !v || !v_new || !ssa->ws_proj)
        {
            ssa_opt_free_ptr(u);
            ssa_opt_free_ptr(v);
            ssa_opt_free_ptr(v_new);
            return -1;
        }
        ssa->total_variance = 0.0;
        for (int comp = 0; comp < k; comp++)
        {
            vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, ssa->rng, K, v, -0.5, 0.5);
            if (comp > 0)
            {
                cblas_dgemv(CblasColMajor, CblasTrans, K, comp, 1.0, ssa->V, K, v, 1, 0.0, ssa->ws_proj, 1);
                cblas_dgemv(CblasColMajor, CblasNoTrans, K, comp, -1.0, ssa->V, K, ssa->ws_proj, 1, 1.0, v, 1);
            }
            ssa_opt_normalize(v, K);
            for (int iter = 0; iter < max_iter; iter++)
            {
                ssa_opt_hankel_matvec(ssa, v, u);
                if (comp > 0)
                {
                    cblas_dgemv(CblasColMajor, CblasTrans, L, comp, 1.0, ssa->U, L, u, 1, 0.0, ssa->ws_proj, 1);
                    cblas_dgemv(CblasColMajor, CblasNoTrans, L, comp, -1.0, ssa->U, L, ssa->ws_proj, 1, 1.0, u, 1);
                }
                ssa_opt_hankel_matvec_T(ssa, u, v_new);
                if (comp > 0)
                {
                    cblas_dgemv(CblasColMajor, CblasTrans, K, comp, 1.0, ssa->V, K, v_new, 1, 0.0, ssa->ws_proj, 1);
                    cblas_dgemv(CblasColMajor, CblasNoTrans, K, comp, -1.0, ssa->V, K, ssa->ws_proj, 1, 1.0, v_new, 1);
                }
                ssa_opt_normalize(v_new, K);
                double diff_same = 0.0, diff_flip = 0.0;
                for (int i = 0; i < K; i++)
                {
                    double d_same = v[i] - v_new[i], d_flip = v[i] + v_new[i];
                    diff_same += d_same * d_same;
                    diff_flip += d_flip * d_flip;
                }
                double diff = (diff_same < diff_flip) ? diff_same : diff_flip;
                ssa_opt_copy(v_new, v, K);
                if (sqrt(diff) < SSA_CONVERGENCE_TOL && iter > 10)
                    break;
            }
            if (comp > 0)
            {
                cblas_dgemv(CblasColMajor, CblasTrans, K, comp, 1.0, ssa->V, K, v, 1, 0.0, ssa->ws_proj, 1);
                cblas_dgemv(CblasColMajor, CblasNoTrans, K, comp, -1.0, ssa->V, K, ssa->ws_proj, 1, 1.0, v, 1);
            }
            ssa_opt_normalize(v, K);
            ssa_opt_hankel_matvec(ssa, v, u);
            if (comp > 0)
            {
                cblas_dgemv(CblasColMajor, CblasTrans, L, comp, 1.0, ssa->U, L, u, 1, 0.0, ssa->ws_proj, 1);
                cblas_dgemv(CblasColMajor, CblasNoTrans, L, comp, -1.0, ssa->U, L, ssa->ws_proj, 1, 1.0, u, 1);
            }
            double sigma = ssa_opt_normalize(u, L);
            ssa_opt_hankel_matvec_T(ssa, u, v);
            if (comp > 0)
            {
                cblas_dgemv(CblasColMajor, CblasTrans, K, comp, 1.0, ssa->V, K, v, 1, 0.0, ssa->ws_proj, 1);
                cblas_dgemv(CblasColMajor, CblasNoTrans, K, comp, -1.0, ssa->V, K, ssa->ws_proj, 1, 1.0, v, 1);
            }
            if (sigma > 1e-12)
                ssa_opt_scal(v, K, 1.0 / sigma);
            ssa_opt_copy(u, &ssa->U[comp * L], L);
            ssa_opt_copy(v, &ssa->V[comp * K], K);
            ssa->sigma[comp] = sigma;
            ssa->eigenvalues[comp] = sigma * sigma;
            ssa->total_variance += sigma * sigma;
        }
        for (int i = 0; i < k - 1; i++)
            for (int j = i + 1; j < k; j++)
                if (ssa->sigma[j] > ssa->sigma[i])
                {
                    double tmp = ssa->sigma[i];
                    ssa->sigma[i] = ssa->sigma[j];
                    ssa->sigma[j] = tmp;
                    tmp = ssa->eigenvalues[i];
                    ssa->eigenvalues[i] = ssa->eigenvalues[j];
                    ssa->eigenvalues[j] = tmp;
                    cblas_dswap(L, &ssa->U[i * L], 1, &ssa->U[j * L], 1);
                    cblas_dswap(K, &ssa->V[i * K], 1, &ssa->V[j * K], 1);
                }
        for (int i = 0; i < k; i++)
        {
            double sum = 0;
            for (int t = 0; t < L; t++)
                sum += ssa->U[i * L + t];
            if (sum < 0)
            {
                ssa_opt_scal(&ssa->U[i * L], L, -1.0);
                ssa_opt_scal(&ssa->V[i * K], K, -1.0);
            }
        }
        ssa_opt_free_ptr(u);
        ssa_opt_free_ptr(v);
        ssa_opt_free_ptr(v_new);
        ssa->decomposed = true;
        return 0;
    }

    // ========================================================================
    // Randomized SVD Decomposition
    // Algorithm: Y=H·Ω → Q=qr(Y) → B=Qᵀ·H → SVD(B) → U=Q·Uᵦ
    // See docs/RANDOMIZED_SVD.md for full derivation
    // ========================================================================
    int ssa_opt_decompose_randomized(SSA_Opt *ssa, int k, int oversampling)
    {
        if (!ssa || !ssa->initialized || k < 1)
            return -1;
        ssa_opt_free_cached_ffts(ssa);
        int L = ssa->L, K = ssa->K;
        int p = (oversampling <= 0) ? 8 : oversampling; // p=8 default oversampling
        int kp = k + p;
        kp = ssa_opt_min(kp, ssa_opt_min(L, K));
        k = ssa_opt_min(k, kp);

        // Require prepare() for malloc-free hot path
        if (!ssa->prepared || kp > ssa->prepared_kp)
            return -1;

        // Use pre-allocated workspace (zero allocations from here)
        double *Omega = ssa->decomp_Omega;
        double *Y = ssa->decomp_Y;
        double *Q = ssa->decomp_Q;
        double *B = ssa->decomp_B;
        double *tau = ssa->decomp_tau;
        double *B_left = ssa->decomp_B_left;
        double *B_right_T = ssa->decomp_B_right_T;
        double *S_svd = ssa->decomp_S;
        double *work = ssa->decomp_work;
        int *iwork = ssa->decomp_iwork;
        int lwork = ssa->prepared_lwork;

        // U, V, sigma, eigenvalues already allocated by prepare()
        // Just update count to reflect actual k being used
        ssa->n_components = k;
        ssa->total_variance = 0.0;

        // Step 1: Random projection Y = H·Ω (captures column space of H)
        vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER, ssa->rng, K * kp, Omega, 0.0, 1.0);
        ssa_opt_hankel_matvec_block(ssa, Omega, Y, kp);

        // Step 2: QR factorization Q = orth(Y)
        cblas_dcopy(L * kp, Y, 1, Q, 1);
        LAPACKE_dgeqrf(LAPACK_COL_MAJOR, L, kp, Q, L, tau);
        LAPACKE_dorgqr(LAPACK_COL_MAJOR, L, kp, kp, Q, L, tau);

        // Step 3: Project to small matrix B = Hᵀ·Q (kp×K matrix, but stored K×kp)
        ssa_opt_hankel_matvec_T_block(ssa, Q, B, kp);

        // Step 4: SVD of small matrix B = Uᵦ·Σ·Vᵀ (GESDD divide-and-conquer)
        int info = LAPACKE_dgesdd_work(LAPACK_COL_MAJOR, 'S', K, kp, B, K, S_svd, B_left, K, B_right_T, kp, work, lwork, iwork);
        if (info != 0)
            return -1;

        // Step 5: Recover U = Q·Vᵦᵀ (rotate Q by right singular vectors of B)
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, L, k, kp, 1.0, Q, L, B_right_T, kp, 0.0, ssa->U, L);
        for (int i = 0; i < k; i++)
            cblas_dcopy(K, &B_left[i * K], 1, &ssa->V[i * K], 1);

        // Store singular values and eigenvalues (σ² for variance)
        for (int i = 0; i < k; i++)
        {
            ssa->sigma[i] = S_svd[i];
            ssa->eigenvalues[i] = S_svd[i] * S_svd[i];
            ssa->total_variance += ssa->eigenvalues[i];
        }

        // Fix sign convention: ensure sum(U[:,i]) > 0 for reproducibility
        for (int i = 0; i < k; i++)
        {
            double sum = 0;
            for (int t = 0; t < L; t++)
                sum += ssa->U[i * L + t];
            if (sum < 0)
            {
                ssa_opt_scal(&ssa->U[i * L], L, -1.0);
                ssa_opt_scal(&ssa->V[i * K], K, -1.0);
            }
        }

        ssa->decomposed = true;
        return 0;
    }

    int ssa_opt_decompose_block(SSA_Opt *ssa, int k, int block_size, int max_iter)
    {
        if (!ssa || !ssa->initialized || k < 1 || block_size < 1)
            return -1;
        ssa_opt_free_cached_ffts(ssa);
        int L = ssa->L, K = ssa->K;
        if (block_size <= 0)
            block_size = SSA_BATCH_SIZE;
        int b = ssa_opt_min(block_size, ssa_opt_min(k, ssa_opt_min(L, K)));
        k = ssa_opt_min(k, ssa_opt_min(L, K));
        ssa->U = (double *)ssa_opt_alloc(L * k * sizeof(double));
        ssa->V = (double *)ssa_opt_alloc(K * k * sizeof(double));
        ssa->sigma = (double *)ssa_opt_alloc(k * sizeof(double));
        ssa->eigenvalues = (double *)ssa_opt_alloc(k * sizeof(double));
        if (!ssa->U || !ssa->V || !ssa->sigma || !ssa->eigenvalues)
            return -1;
        ssa->n_components = k;
        ssa->total_variance = 0.0;
        double *V_block = (double *)ssa_opt_alloc(K * b * sizeof(double));
        double *U_block = (double *)ssa_opt_alloc(L * b * sizeof(double));
        double *U_block2 = (double *)ssa_opt_alloc(L * b * sizeof(double));
        double *tau_u = (double *)ssa_opt_alloc(b * sizeof(double));
        double *tau_v = (double *)ssa_opt_alloc(b * sizeof(double));
        double *M = (double *)ssa_opt_alloc(b * b * sizeof(double));
        double *U_small = (double *)ssa_opt_alloc(b * b * sizeof(double));
        double *Vt_small = (double *)ssa_opt_alloc(b * b * sizeof(double));
        double *S_small = (double *)ssa_opt_alloc(b * sizeof(double));
        double *superb = (double *)ssa_opt_alloc(b * sizeof(double));
        double *work = (double *)ssa_opt_alloc(ssa_opt_max(L, K) * b * sizeof(double));
        if (!V_block || !U_block || !U_block2 || !tau_u || !tau_v || !M || !U_small || !Vt_small || !S_small || !superb || !work)
        {
            ssa_opt_free_ptr(V_block);
            ssa_opt_free_ptr(U_block);
            ssa_opt_free_ptr(U_block2);
            ssa_opt_free_ptr(tau_u);
            ssa_opt_free_ptr(tau_v);
            ssa_opt_free_ptr(M);
            ssa_opt_free_ptr(U_small);
            ssa_opt_free_ptr(Vt_small);
            ssa_opt_free_ptr(S_small);
            ssa_opt_free_ptr(superb);
            ssa_opt_free_ptr(work);
            return -1;
        }
        int comp = 0;
        while (comp < k)
        {
            int cur_b = ssa_opt_min(b, k - comp);
            vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, ssa->rng, K * cur_b, V_block, -0.5, 0.5);
            if (comp > 0)
            {
                cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, comp, cur_b, K, 1.0, ssa->V, K, V_block, K, 0.0, work, comp);
                cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, K, cur_b, comp, -1.0, ssa->V, K, work, comp, 1.0, V_block, K);
            }
            LAPACKE_dgeqrf(LAPACK_COL_MAJOR, K, cur_b, V_block, K, tau_v);
            LAPACKE_dorgqr(LAPACK_COL_MAJOR, K, cur_b, cur_b, V_block, K, tau_v);
            const int QR_INTERVAL = 5;
            for (int iter = 0; iter < max_iter; iter++)
            {
                ssa_opt_hankel_matvec_block(ssa, V_block, U_block, cur_b);
                if (comp > 0)
                {
                    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, comp, cur_b, L, 1.0, ssa->U, L, U_block, L, 0.0, work, comp);
                    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, L, cur_b, comp, -1.0, ssa->U, L, work, comp, 1.0, U_block, L);
                }
                if ((iter % QR_INTERVAL == 0) || (iter == max_iter - 1))
                {
                    LAPACKE_dgeqrf(LAPACK_COL_MAJOR, L, cur_b, U_block, L, tau_u);
                    LAPACKE_dorgqr(LAPACK_COL_MAJOR, L, cur_b, cur_b, U_block, L, tau_u);
                }
                ssa_opt_hankel_matvec_T_block(ssa, U_block, V_block, cur_b);
                if (comp > 0)
                {
                    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, comp, cur_b, K, 1.0, ssa->V, K, V_block, K, 0.0, work, comp);
                    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, K, cur_b, comp, -1.0, ssa->V, K, work, comp, 1.0, V_block, K);
                }
                if ((iter > 0 && iter % QR_INTERVAL == 0) || (iter == max_iter - 1))
                {
                    LAPACKE_dgeqrf(LAPACK_COL_MAJOR, K, cur_b, V_block, K, tau_v);
                    LAPACKE_dorgqr(LAPACK_COL_MAJOR, K, cur_b, cur_b, V_block, K, tau_v);
                }
            }
            ssa_opt_hankel_matvec_block(ssa, V_block, U_block2, cur_b);
            if (comp > 0)
            {
                cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, comp, cur_b, L, 1.0, ssa->U, L, U_block2, L, 0.0, work, comp);
                cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, L, cur_b, comp, -1.0, ssa->U, L, work, comp, 1.0, U_block2, L);
            }
            cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, cur_b, cur_b, L, 1.0, U_block, L, U_block2, L, 0.0, M, cur_b);
            int svd_info = LAPACKE_dgesvd(LAPACK_COL_MAJOR, 'A', 'A', cur_b, cur_b, M, cur_b, S_small, U_small, cur_b, Vt_small, cur_b, superb);
            if (svd_info != 0)
            {
                for (int i = 0; i < cur_b; i++)
                    S_small[i] = cblas_dnrm2(L, &U_block2[i * L], 1);
                memset(U_small, 0, cur_b * cur_b * sizeof(double));
                memset(Vt_small, 0, cur_b * cur_b * sizeof(double));
                for (int i = 0; i < cur_b; i++)
                {
                    U_small[i + i * cur_b] = 1.0;
                    Vt_small[i + i * cur_b] = 1.0;
                }
            }
            cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, L, cur_b, cur_b, 1.0, U_block, L, U_small, cur_b, 0.0, U_block2, L);
            cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, K, cur_b, cur_b, 1.0, V_block, K, Vt_small, cur_b, 0.0, work, K);
            for (int i = 0; i < cur_b; i++)
            {
                double sigma = S_small[i];
                ssa->sigma[comp + i] = sigma;
                ssa->eigenvalues[comp + i] = sigma * sigma;
                ssa->total_variance += sigma * sigma;
                cblas_dcopy(L, &U_block2[i * L], 1, &ssa->U[(comp + i) * L], 1);
                cblas_dcopy(K, &work[i * K], 1, &ssa->V[(comp + i) * K], 1);
            }
            for (int i = 0; i < cur_b; i++)
                cblas_dcopy(L, &ssa->U[(comp + i) * L], 1, &U_block[i * L], 1);
            ssa_opt_hankel_matvec_T_block(ssa, U_block, V_block, cur_b);
            if (comp > 0)
            {
                cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, comp, cur_b, K, 1.0, ssa->V, K, V_block, K, 0.0, work, comp);
                cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, K, cur_b, comp, -1.0, ssa->V, K, work, comp, 1.0, V_block, K);
            }
            for (int i = 0; i < cur_b; i++)
            {
                double sigma = ssa->sigma[comp + i];
                double *v_col = &V_block[i * K];
                if (sigma > 1e-12)
                    cblas_dscal(K, 1.0 / sigma, v_col, 1);
                cblas_dcopy(K, v_col, 1, &ssa->V[(comp + i) * K], 1);
            }
            comp += cur_b;
        }
        for (int i = 0; i < k - 1; i++)
            for (int j = i + 1; j < k; j++)
                if (ssa->sigma[j] > ssa->sigma[i])
                {
                    double tmp = ssa->sigma[i];
                    ssa->sigma[i] = ssa->sigma[j];
                    ssa->sigma[j] = tmp;
                    tmp = ssa->eigenvalues[i];
                    ssa->eigenvalues[i] = ssa->eigenvalues[j];
                    ssa->eigenvalues[j] = tmp;
                    cblas_dswap(L, &ssa->U[i * L], 1, &ssa->U[j * L], 1);
                    cblas_dswap(K, &ssa->V[i * K], 1, &ssa->V[j * K], 1);
                }
        for (int i = 0; i < k; i++)
        {
            double sum = 0;
            for (int t = 0; t < L; t++)
                sum += ssa->U[i * L + t];
            if (sum < 0)
            {
                ssa_opt_scal(&ssa->U[i * L], L, -1.0);
                ssa_opt_scal(&ssa->V[i * K], K, -1.0);
            }
        }
        ssa_opt_free_ptr(V_block);
        ssa_opt_free_ptr(U_block);
        ssa_opt_free_ptr(U_block2);
        ssa_opt_free_ptr(tau_u);
        ssa_opt_free_ptr(tau_v);
        ssa_opt_free_ptr(M);
        ssa_opt_free_ptr(U_small);
        ssa_opt_free_ptr(Vt_small);
        ssa_opt_free_ptr(S_small);
        ssa_opt_free_ptr(superb);
        ssa_opt_free_ptr(work);
        ssa->decomposed = true;
        return 0;
    }

    int ssa_opt_extend(SSA_Opt *ssa, int additional_k, int max_iter)
    {
        if (!ssa || !ssa->decomposed || additional_k < 1)
            return -1;
        ssa_opt_free_cached_ffts(ssa);
        int L = ssa->L, K = ssa->K, old_k = ssa->n_components, new_k = old_k + additional_k;
        new_k = ssa_opt_min(new_k, ssa_opt_min(L, K));
        if (new_k <= old_k)
            return 0;
        double *U_new = (double *)ssa_opt_alloc(L * new_k * sizeof(double));
        double *V_new = (double *)ssa_opt_alloc(K * new_k * sizeof(double));
        double *sigma_new = (double *)ssa_opt_alloc(new_k * sizeof(double));
        double *eigen_new = (double *)ssa_opt_alloc(new_k * sizeof(double));
        if (!U_new || !V_new || !sigma_new || !eigen_new)
        {
            ssa_opt_free_ptr(U_new);
            ssa_opt_free_ptr(V_new);
            ssa_opt_free_ptr(sigma_new);
            ssa_opt_free_ptr(eigen_new);
            return -1;
        }
        memcpy(U_new, ssa->U, L * old_k * sizeof(double));
        memcpy(V_new, ssa->V, K * old_k * sizeof(double));
        memcpy(sigma_new, ssa->sigma, old_k * sizeof(double));
        memcpy(eigen_new, ssa->eigenvalues, old_k * sizeof(double));
        ssa_opt_free_ptr(ssa->U);
        ssa_opt_free_ptr(ssa->V);
        ssa_opt_free_ptr(ssa->sigma);
        ssa_opt_free_ptr(ssa->eigenvalues);
        ssa->U = U_new;
        ssa->V = V_new;
        ssa->sigma = sigma_new;
        ssa->eigenvalues = eigen_new;
        if (ssa->ws_proj)
            ssa_opt_free_ptr(ssa->ws_proj);
        ssa->ws_proj = (double *)ssa_opt_alloc(new_k * sizeof(double));
        if (!ssa->ws_proj)
            return -1;
        double *u = (double *)ssa_opt_alloc(L * sizeof(double)), *v = (double *)ssa_opt_alloc(K * sizeof(double)), *v_new = (double *)ssa_opt_alloc(K * sizeof(double));
        if (!u || !v || !v_new)
        {
            ssa_opt_free_ptr(u);
            ssa_opt_free_ptr(v);
            ssa_opt_free_ptr(v_new);
            return -1;
        }
        for (int comp = old_k; comp < new_k; comp++)
        {
            vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, ssa->rng, K, v, -0.5, 0.5);
            cblas_dgemv(CblasColMajor, CblasTrans, K, comp, 1.0, ssa->V, K, v, 1, 0.0, ssa->ws_proj, 1);
            cblas_dgemv(CblasColMajor, CblasNoTrans, K, comp, -1.0, ssa->V, K, ssa->ws_proj, 1, 1.0, v, 1);
            ssa_opt_normalize(v, K);
            for (int iter = 0; iter < max_iter; iter++)
            {
                ssa_opt_hankel_matvec(ssa, v, u);
                cblas_dgemv(CblasColMajor, CblasTrans, L, comp, 1.0, ssa->U, L, u, 1, 0.0, ssa->ws_proj, 1);
                cblas_dgemv(CblasColMajor, CblasNoTrans, L, comp, -1.0, ssa->U, L, ssa->ws_proj, 1, 1.0, u, 1);
                ssa_opt_normalize(u, L);
                ssa_opt_hankel_matvec_T(ssa, u, v_new);
                cblas_dgemv(CblasColMajor, CblasTrans, K, comp, 1.0, ssa->V, K, v_new, 1, 0.0, ssa->ws_proj, 1);
                cblas_dgemv(CblasColMajor, CblasNoTrans, K, comp, -1.0, ssa->V, K, ssa->ws_proj, 1, 1.0, v_new, 1);
                ssa_opt_normalize(v_new, K);
                double diff_same = 0.0, diff_flip = 0.0;
                for (int i = 0; i < K; i++)
                {
                    double d_same = v[i] - v_new[i], d_flip = v[i] + v_new[i];
                    diff_same += d_same * d_same;
                    diff_flip += d_flip * d_flip;
                }
                double diff = (diff_same < diff_flip) ? diff_same : diff_flip;
                ssa_opt_copy(v_new, v, K);
                if (sqrt(diff) < SSA_CONVERGENCE_TOL && iter > 10)
                    break;
            }
            cblas_dgemv(CblasColMajor, CblasTrans, K, comp, 1.0, ssa->V, K, v, 1, 0.0, ssa->ws_proj, 1);
            cblas_dgemv(CblasColMajor, CblasNoTrans, K, comp, -1.0, ssa->V, K, ssa->ws_proj, 1, 1.0, v, 1);
            ssa_opt_normalize(v, K);
            ssa_opt_hankel_matvec(ssa, v, u);
            cblas_dgemv(CblasColMajor, CblasTrans, L, comp, 1.0, ssa->U, L, u, 1, 0.0, ssa->ws_proj, 1);
            cblas_dgemv(CblasColMajor, CblasNoTrans, L, comp, -1.0, ssa->U, L, ssa->ws_proj, 1, 1.0, u, 1);
            double sigma = ssa_opt_normalize(u, L);
            ssa_opt_hankel_matvec_T(ssa, u, v);
            cblas_dgemv(CblasColMajor, CblasTrans, K, comp, 1.0, ssa->V, K, v, 1, 0.0, ssa->ws_proj, 1);
            cblas_dgemv(CblasColMajor, CblasNoTrans, K, comp, -1.0, ssa->V, K, ssa->ws_proj, 1, 1.0, v, 1);
            if (sigma > 1e-12)
                ssa_opt_scal(v, K, 1.0 / sigma);
            ssa_opt_copy(u, &ssa->U[comp * L], L);
            ssa_opt_copy(v, &ssa->V[comp * K], K);
            ssa->sigma[comp] = sigma;
            ssa->eigenvalues[comp] = sigma * sigma;
            ssa->total_variance += sigma * sigma;
        }
        for (int i = 0; i < new_k - 1; i++)
            for (int j = i + 1; j < new_k; j++)
                if (ssa->sigma[j] > ssa->sigma[i])
                {
                    double tmp = ssa->sigma[i];
                    ssa->sigma[i] = ssa->sigma[j];
                    ssa->sigma[j] = tmp;
                    tmp = ssa->eigenvalues[i];
                    ssa->eigenvalues[i] = ssa->eigenvalues[j];
                    ssa->eigenvalues[j] = tmp;
                    cblas_dswap(L, &ssa->U[i * L], 1, &ssa->U[j * L], 1);
                    cblas_dswap(K, &ssa->V[i * K], 1, &ssa->V[j * K], 1);
                }
        for (int i = old_k; i < new_k; i++)
        {
            double sum = 0;
            for (int t = 0; t < L; t++)
                sum += ssa->U[i * L + t];
            if (sum < 0)
            {
                ssa_opt_scal(&ssa->U[i * L], L, -1.0);
                ssa_opt_scal(&ssa->V[i * K], K, -1.0);
            }
        }
        ssa->n_components = new_k;
        ssa_opt_free_ptr(u);
        ssa_opt_free_ptr(v);
        ssa_opt_free_ptr(v_new);
        return 0;
    }

    void ssa_opt_free_cached_ffts(SSA_Opt *ssa)
    {
        if (!ssa)
            return;
        ssa_opt_free_ptr(ssa->U_fft);
        ssa_opt_free_ptr(ssa->V_fft);
        if (ssa->fft_c2r_wcorr)
            DftiFreeDescriptor(&ssa->fft_c2r_wcorr);
        ssa->U_fft = NULL;
        ssa->V_fft = NULL;
        ssa->fft_c2r_wcorr = NULL;
        ssa->fft_cached = false;
        ssa_opt_free_ptr(ssa->wcorr_ws_complex);
        ssa_opt_free_ptr(ssa->wcorr_h);
        ssa_opt_free_ptr(ssa->wcorr_G);
        ssa_opt_free_ptr(ssa->wcorr_sqrt_inv_c);
        ssa->wcorr_ws_complex = NULL;
        ssa->wcorr_h = NULL;
        ssa->wcorr_G = NULL;
        ssa->wcorr_sqrt_inv_c = NULL;
    }

    /**
     * Pre-compute and cache FFTs of eigenvectors for fast reconstruction and W-correlation.
     *
     * This is an optional optimization. After decomposition, reconstruction requires:
     *   x_i[t] = diagonal_average(σᵢ · uᵢ · vᵢᵀ)
     *
     * With frequency-domain accumulation, this becomes:
     *   FFT(x_reconstructed) = Σᵢ FFT(σᵢ·uᵢ) ⊙ FFT(vᵢ)
     *
     * By caching FFT(σᵢ·uᵢ) and FFT(vᵢ), reconstruction reduces to:
     *   - n complex element-wise multiplies (cached FFTs)
     *   - 1 inverse FFT (instead of n forward + n inverse)
     *
     * For n=50 components, this is ~100× faster than naive reconstruction.
     *
     * Also pre-allocates W-correlation workspace for ssa_opt_wcorr_matrix_fast().
     *
     * Call this after decomposition if you plan to:
     *   - Call ssa_opt_reconstruct() multiple times with different groups
     *   - Call ssa_opt_wcorr_matrix_fast()
     *
     * Memory cost: ~4 * k * fft_len doubles (typically 1-5 MB)
     *
     * OPTIMIZATION: Uses pre-allocated per-thread FFT descriptors from init()
     * instead of creating/destroying descriptors per call. This eliminates
     * the expensive DftiCommitDescriptor overhead in hot loops.
     *
     * @param ssa  Decomposed SSA structure
     * @return     0 on success, -1 on failure
     */
    int ssa_opt_cache_ffts(SSA_Opt *ssa)
    {
        if (!ssa || !ssa->decomposed)
            return -1;

        // Free any existing cache (allows re-caching after new decomposition)
        ssa_opt_free_cached_ffts(ssa);

        int N = ssa->N, L = ssa->L, K = ssa->K;
        int fft_len = ssa->fft_len, r2c_len = ssa->r2c_len;
        int k = ssa->n_components;

        // === Allocate FFT Cache Arrays ===
        // Each cached FFT is r2c_len complex numbers = 2 * r2c_len doubles
        // Total: k cached FFTs for U, k cached FFTs for V
        size_t cache_size = 2 * r2c_len * k * sizeof(double);

        // U_fft[i] = FFT(σᵢ · U[:,i]) - scaled left singular vectors
        // Pre-multiplying by σᵢ saves one multiply per component during reconstruction
        ssa->U_fft = (double *)ssa_opt_alloc(cache_size);

        // V_fft[i] = FFT(V[:,i]) - right singular vectors
        ssa->V_fft = (double *)ssa_opt_alloc(cache_size);

        if (!ssa->U_fft || !ssa->V_fft)
        {
            ssa_opt_free_cached_ffts(ssa);
            return -1;
        }

#ifdef _OPENMP
        // =========================================================================
        // PARALLEL PATH: Use pre-allocated per-thread FFT descriptors
        // This eliminates the expensive DftiCommitDescriptor calls that were
        // previously done inside this function on every call.
        // =========================================================================
        int n_threads = ssa->thread_pool_size;

        // Allocate per-thread workspaces for zero-padded input
        double *ws_pool = (double *)ssa_opt_alloc(n_threads * fft_len * sizeof(double));
        if (!ws_pool)
        {
            ssa_opt_free_cached_ffts(ssa);
            return -1;
        }

        // === Compute FFT of Each Scaled Left Singular Vector (PARALLEL) ===
        // U_fft[i] = FFT(σᵢ · uᵢ)
        // The σᵢ scaling is baked in so reconstruction just needs element-wise multiply
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            double *ws_real = ws_pool + tid * fft_len;
            DFTI_DESCRIPTOR_HANDLE my_fft = ssa->thread_fft_pool[tid]; // Pre-allocated!
            int i; // Declare BEFORE pragma for MSVC OpenMP 2.0 compatibility

            #pragma omp for schedule(static)
            for (i = 0; i < k; i++)
            {
                double sigma = ssa->sigma[i];
                const double *u_vec = &ssa->U[i * L];
                double *dst = &ssa->U_fft[i * 2 * r2c_len];

                memset(ws_real, 0, fft_len * sizeof(double));
                for (int j = 0; j < L; j++)
                    ws_real[j] = sigma * u_vec[j];

                DftiComputeForward(my_fft, ws_real, dst);
            }
        }

        // === Compute FFT of Each Right Singular Vector (PARALLEL) ===
        // V_fft[i] = FFT(vᵢ)
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            double *ws_real = ws_pool + tid * fft_len;
            DFTI_DESCRIPTOR_HANDLE my_fft = ssa->thread_fft_pool[tid]; // Pre-allocated!
            int i; // Declare BEFORE pragma for MSVC OpenMP 2.0 compatibility

            #pragma omp for schedule(static)
            for (i = 0; i < k; i++)
            {
                const double *v_vec = &ssa->V[i * K];
                double *dst = &ssa->V_fft[i * 2 * r2c_len];

                memset(ws_real, 0, fft_len * sizeof(double));
                for (int j = 0; j < K; j++)
                    ws_real[j] = v_vec[j];

                DftiComputeForward(my_fft, ws_real, dst);
            }
        }

        // Free per-thread workspace (but NOT the descriptors - they're reused!)
        ssa_opt_free_ptr(ws_pool);

#else
        // =========================================================================
        // SEQUENTIAL PATH: Original single-threaded implementation
        // =========================================================================

        // === Compute FFT of Each Scaled Left Singular Vector ===
        // U_fft[i] = FFT(σᵢ · uᵢ)
        // The σᵢ scaling is baked in so reconstruction just needs element-wise multiply
        for (int i = 0; i < k; i++)
        {
            double sigma = ssa->sigma[i];
            const double *u_vec = &ssa->U[i * L];       // u_i has length L
            double *dst = &ssa->U_fft[i * 2 * r2c_len]; // Destination in cache

            // Zero-pad to FFT length and scale by σᵢ
            ssa_opt_zero(ssa->ws_real, fft_len);
            for (int j = 0; j < L; j++)
                ssa->ws_real[j] = sigma * u_vec[j];

            // Forward FFT directly into cache
            DftiComputeForward(ssa->fft_r2c, ssa->ws_real, dst);
        }

        // === Compute FFT of Each Right Singular Vector ===
        // V_fft[i] = FFT(vᵢ)
        for (int i = 0; i < k; i++)
        {
            const double *v_vec = &ssa->V[i * K];       // v_i has length K
            double *dst = &ssa->V_fft[i * 2 * r2c_len]; // Destination in cache

            // Zero-pad to FFT length (no scaling needed for V)
            ssa_opt_zero(ssa->ws_real, fft_len);
            for (int j = 0; j < K; j++)
                ssa->ws_real[j] = v_vec[j];

            // Forward FFT directly into cache
            DftiComputeForward(ssa->fft_r2c, ssa->ws_real, dst);
        }
#endif

        // === Allocate W-Correlation Workspace ===
        // W-correlation matrix W[i,j] = weighted correlation between components i and j
        // Fast computation uses DSYRK: W = G · Gᵀ where G is pre-computed weighted matrix
        // This block pre-allocates all workspace needed by ssa_opt_wcorr_matrix_fast()
        {
            MKL_LONG status;
            double scale = 1.0 / fft_len; // IFFT normalization

            // wcorr_ws_complex: workspace for batched FFT output
            // Holds k complex vectors of length r2c_len
            ssa->wcorr_ws_complex = (double *)ssa_opt_alloc(k * 2 * r2c_len * sizeof(double));

            // wcorr_h: reconstructed components before weighting, k × fft_len
            // Each row is IFFT(U_fft[i] ⊙ V_fft[i]) = diagonal sums for component i
            ssa->wcorr_h = (double *)ssa_opt_alloc(k * fft_len * sizeof(double));

            // wcorr_G: weighted/normalized matrix for DSYRK, k × N
            // G[i,t] = (hᵢ[t] / ||hᵢ||_w) · √w[t]
            // Then W = G · Gᵀ gives the W-correlation matrix directly
            ssa->wcorr_G = (double *)ssa_opt_alloc(k * N * sizeof(double));

            // wcorr_sqrt_inv_c: precomputed √(1/count[t]) for fast weighting
            // Used to convert raw diagonal sums to weighted values
            ssa->wcorr_sqrt_inv_c = (double *)ssa_opt_alloc(N * sizeof(double));

            if (!ssa->wcorr_ws_complex || !ssa->wcorr_h ||
                !ssa->wcorr_G || !ssa->wcorr_sqrt_inv_c)
            {
                ssa_opt_free_cached_ffts(ssa);
                return -1;
            }

            // Pre-compute √(1/count) for each position
            // count[t] = number of anti-diagonal elements at position t
            // inv_diag_count[t] = 1/count[t] (already computed in init)
            for (int t = 0; t < N; t++)
                ssa->wcorr_sqrt_inv_c[t] = sqrt(ssa->inv_diag_count[t]);

            // === Create Batched IFFT Descriptor for W-correlation ===
            // This descriptor computes k IFFTs in a single MKL call
            // Used to transform all k component products simultaneously
            status = DftiCreateDescriptor(&ssa->fft_c2r_wcorr, DFTI_DOUBLE, DFTI_REAL, 1, fft_len);
            if (status != 0)
            {
                ssa_opt_free_cached_ffts(ssa);
                return -1;
            }

            // Configure as batched C2R (complex-to-real) transform
            DftiSetValue(ssa->fft_c2r_wcorr, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
            DftiSetValue(ssa->fft_c2r_wcorr, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);

            // NUMBER_OF_TRANSFORMS: process all k components in one call
            DftiSetValue(ssa->fft_c2r_wcorr, DFTI_NUMBER_OF_TRANSFORMS, (MKL_LONG)k);

            // INPUT_DISTANCE: stride between consecutive complex inputs
            // Each input is r2c_len complex numbers (interleaved re/im)
            DftiSetValue(ssa->fft_c2r_wcorr, DFTI_INPUT_DISTANCE, (MKL_LONG)r2c_len);

            // OUTPUT_DISTANCE: stride between consecutive real outputs
            // Each output is fft_len real numbers
            DftiSetValue(ssa->fft_c2r_wcorr, DFTI_OUTPUT_DISTANCE, (MKL_LONG)fft_len);

            // Apply 1/N scaling to inverse FFT
            DftiSetValue(ssa->fft_c2r_wcorr, DFTI_BACKWARD_SCALE, scale);

            status = DftiCommitDescriptor(ssa->fft_c2r_wcorr);
            if (status != 0)
            {
                ssa_opt_free_cached_ffts(ssa);
                return -1;
            }
        }

        ssa->fft_cached = true;
        return 0;
    }

    int ssa_opt_reconstruct(const SSA_Opt *ssa, const int *group, int n_group, double *output)
    {
        if (!ssa || !ssa->decomposed || !group || !output || n_group < 1)
            return -1;

        int N = ssa->N, L = ssa->L, K = ssa->K;
        int fft_len = ssa->fft_len, r2c_len = ssa->r2c_len;
        SSA_Opt *ssa_mut = (SSA_Opt *)ssa;

        // Use batch workspace as accumulator
        double *freq_accum = ssa_mut->ws_batch_complex;
        ssa_opt_zero(freq_accum, 2 * r2c_len);

        if (ssa->fft_cached && ssa->U_fft && ssa->V_fft)
        {
            // ===================================================================
            // FAST PATH: Use cached FFTs with SIMD fused multiply-accumulate
            // Replaces: complex_mul + daxpy with single fused operation
            // ===================================================================
            for (int g = 0; g < n_group; g++)
            {
                int idx = group[g];
                if (idx < 0 || idx >= ssa->n_components)
                    continue;

                const double *u_fft_cached = &ssa->U_fft[idx * 2 * r2c_len];
                const double *v_fft_cached = &ssa->V_fft[idx * 2 * r2c_len];

                // SIMD fused multiply-accumulate: freq_accum += u_fft * v_fft
                ssa_opt_complex_mul_acc(u_fft_cached, v_fft_cached, freq_accum, r2c_len);
            }
        }
        else
        {
            // ===================================================================
            // SLOW PATH: Compute FFTs on-the-fly (no cache)
            // Still uses SIMD accumulate
            // ===================================================================
            double *temp_fft2 = ssa_mut->ws_batch_complex + 2 * r2c_len;

            for (int g = 0; g < n_group; g++)
            {
                int idx = group[g];
                if (idx < 0 || idx >= ssa->n_components)
                    continue;

                double sigma = ssa->sigma[idx];
                const double *u_vec = &ssa->U[idx * L];
                const double *v_vec = &ssa->V[idx * K];

                // Compute FFT(σ * u)
                ssa_opt_zero(ssa_mut->ws_real, fft_len);
                for (int i = 0; i < L; i++)
                    ssa_mut->ws_real[i] = sigma * u_vec[i];
                DftiComputeForward(ssa_mut->fft_r2c, ssa_mut->ws_real, ssa_mut->ws_complex);

                // Compute FFT(v)
                ssa_opt_zero(ssa_mut->ws_real2, fft_len);
                for (int i = 0; i < K; i++)
                    ssa_mut->ws_real2[i] = v_vec[i];
                DftiComputeForward(ssa_mut->fft_r2c, ssa_mut->ws_real2, temp_fft2);

                // SIMD fused multiply-accumulate: freq_accum += ws_complex * temp_fft2
                ssa_opt_complex_mul_acc(ssa_mut->ws_complex, temp_fft2, freq_accum, r2c_len);
            }
        }

        // Inverse FFT to get time-domain result
        DftiComputeBackward(ssa_mut->fft_c2r, freq_accum, ssa_mut->ws_real);

        // Copy and apply diagonal averaging weights
        memcpy(output, ssa_mut->ws_real, N * sizeof(double));
        vdMul(N, output, ssa->inv_diag_count, output);

        return 0;
    }

    // Forward declaration for auto-dispatch
    int ssa_opt_wcorr_matrix_fast(const SSA_Opt *ssa, double *W);

    int ssa_opt_wcorr_matrix(const SSA_Opt *ssa, double *W)
    {
        if (!ssa || !ssa->decomposed || !W)
            return -1;
        // AUTO-DISPATCH: Use fast path if workspace is ready
        if (ssa->fft_cached && ssa->wcorr_ws_complex)
            return ssa_opt_wcorr_matrix_fast(ssa, W);
        // Original sequential implementation
        int N = ssa->N, L = ssa->L, K = ssa->K, n = ssa->n_components, fft_len = ssa->fft_len, r2c_len = ssa->r2c_len;
        SSA_Opt *ssa_mut = (SSA_Opt *)ssa;
        int use_cache = (ssa->U_fft != NULL && ssa->V_fft != NULL);
        double *sqrt_inv_c = (double *)ssa_opt_alloc(N * sizeof(double));
        if (!sqrt_inv_c)
            return -1;
        for (int t = 0; t < N; t++)
            sqrt_inv_c[t] = sqrt(ssa->inv_diag_count[t]);
        double *G = (double *)ssa_opt_alloc(n * N * sizeof(double));
        double *norms = (double *)ssa_opt_alloc(n * sizeof(double));
        double *h_temp = (double *)ssa_opt_alloc(fft_len * sizeof(double));
        double *u_fft_local = NULL, *v_fft_local = NULL;
        if (!use_cache)
        {
            u_fft_local = (double *)ssa_opt_alloc(2 * r2c_len * sizeof(double));
            v_fft_local = (double *)ssa_opt_alloc(2 * r2c_len * sizeof(double));
        }
        if (!G || !norms || !h_temp || (!use_cache && (!u_fft_local || !v_fft_local)))
        {
            ssa_opt_free_ptr(sqrt_inv_c);
            ssa_opt_free_ptr(G);
            ssa_opt_free_ptr(norms);
            ssa_opt_free_ptr(h_temp);
            ssa_opt_free_ptr(u_fft_local);
            ssa_opt_free_ptr(v_fft_local);
            return -1;
        }
        for (int i = 0; i < n; i++)
        {
            double sigma = ssa->sigma[i];
            const double *u_fft_ptr, *v_fft_ptr;
            if (use_cache)
            {
                u_fft_ptr = &ssa->U_fft[i * 2 * r2c_len];
                v_fft_ptr = &ssa->V_fft[i * 2 * r2c_len];
            }
            else
            {
                const double *u_vec = &ssa->U[i * L];
                const double *v_vec = &ssa->V[i * K];
                ssa_opt_zero(ssa_mut->ws_real, fft_len);
                memcpy(ssa_mut->ws_real, u_vec, L * sizeof(double));
                DftiComputeForward(ssa_mut->fft_r2c, ssa_mut->ws_real, u_fft_local);
                ssa_opt_zero(ssa_mut->ws_real, fft_len);
                memcpy(ssa_mut->ws_real, v_vec, K * sizeof(double));
                DftiComputeForward(ssa_mut->fft_r2c, ssa_mut->ws_real, v_fft_local);
                u_fft_ptr = u_fft_local;
                v_fft_ptr = v_fft_local;
            }
            ssa_opt_complex_mul_r2c(u_fft_ptr, v_fft_ptr, ssa_mut->ws_complex, r2c_len);
            DftiComputeBackward(ssa_mut->fft_c2r, ssa_mut->ws_complex, h_temp);
            double norm_sq = ssa_opt_weighted_norm_sq(h_temp, ssa->inv_diag_count, N);
            norm_sq *= sigma * sigma;
            norms[i] = sqrt(norm_sq);
            double scale = (norms[i] > 1e-12) ? sigma / norms[i] : 0.0;
            double *g_row = &G[i * N];
            ssa_opt_scale_weighted(h_temp, sqrt_inv_c, scale, g_row, N);
        }
        cblas_dsyrk(CblasRowMajor, CblasUpper, CblasNoTrans, n, N, 1.0, G, N, 0.0, W, n);
        for (int i = 0; i < n; i++)
            for (int j = i + 1; j < n; j++)
                W[j * n + i] = W[i * n + j];
        ssa_opt_free_ptr(sqrt_inv_c);
        ssa_opt_free_ptr(G);
        ssa_opt_free_ptr(norms);
        ssa_opt_free_ptr(h_temp);
        ssa_opt_free_ptr(u_fft_local);
        ssa_opt_free_ptr(v_fft_local);
        return 0;
    }

    // ========================================================================
    // W-Correlation Matrix (Fast Path)
    // Computes W[i,j] = ⟨Xᵢ,Xⱼ⟩_w / (||Xᵢ||_w · ||Xⱼ||_w) for all pairs
    // Key optimization: Use DSYRK instead of O(n²) pairwise correlations
    // See docs/WCORR.md for full derivation
    // ========================================================================
    int ssa_opt_wcorr_matrix_fast(const SSA_Opt *ssa, double *W)
    {
        int N, n, fft_len, r2c_len, i, t, k, ii, jj;
        if (!ssa || !ssa->decomposed || !W)
            return -1;
        if (!ssa->fft_cached || !ssa->wcorr_ws_complex)
            return ssa_opt_wcorr_matrix(ssa, W);
        N = ssa->N;
        n = ssa->n_components;
        fft_len = ssa->fft_len;
        r2c_len = ssa->r2c_len;
        double *all_ws_complex = ssa->wcorr_ws_complex, *all_h = ssa->wcorr_h, *G = ssa->wcorr_G, *sqrt_inv_c = ssa->wcorr_sqrt_inv_c;

// Step 1: Compute FFT(uᵢ) ⊙ FFT(vᵢ) for all components (parallel)
// Uses pre-cached U_fft, V_fft from ssa_opt_cache_ffts()
#pragma omp parallel for private(k)
        for (i = 0; i < n; i++)
        {
            const double *u_fft = &ssa->U_fft[i * 2 * r2c_len], *v_fft = &ssa->V_fft[i * 2 * r2c_len];
            double *ws = &all_ws_complex[i * 2 * r2c_len];
            for (k = 0; k < r2c_len; k++)
            {
                double ar = u_fft[2 * k], ai = u_fft[2 * k + 1], br = v_fft[2 * k], bi = v_fft[2 * k + 1];
                ws[2 * k] = ar * br - ai * bi;
                ws[2 * k + 1] = ar * bi + ai * br;
            }
        }

        // Step 2: Batched IFFT for all n components (single MKL call)
        {
            MKL_LONG status = DftiComputeBackward(ssa->fft_c2r_wcorr, all_ws_complex, all_h);
            if (status != 0)
                return -1;
        }

// Step 3: Normalize and weight: G[i,t] = (σᵢ/||hᵢ||_w) · hᵢ[t] · √w[t]
// AVX2-optimized weighted_norm_sq and scale_weighted
#pragma omp parallel for private(t)
        for (i = 0; i < n; i++)
        {
            double *h = &all_h[i * fft_len], sigma = ssa->sigma[i], norm_sq, norm, scale, *g_row;
            norm_sq = ssa_opt_weighted_norm_sq(h, ssa->inv_diag_count, N); // ||hᵢ||²_w
            norm_sq *= sigma * sigma;
            norm = sqrt(norm_sq);
            scale = (norm > 1e-12) ? sigma / norm : 0.0;
            g_row = &G[i * N];
            ssa_opt_scale_weighted(h, sqrt_inv_c, scale, g_row, N);
        }

        // Step 4: W = G·Gᵀ via DSYRK (upper triangle only, then mirror)
        // DSYRK is O(n²·N/2) with BLAS3 cache efficiency vs O(n²·N) naive
        cblas_dsyrk(CblasRowMajor, CblasUpper, CblasNoTrans, n, N, 1.0, G, N, 0.0, W, n);
        for (ii = 0; ii < n; ii++)
            for (jj = ii + 1; jj < n; jj++)
                W[jj * n + ii] = W[ii * n + jj];
        return 0;
    }

    // Single-pair W-correlation (for selective queries)
    double ssa_opt_wcorr_pair(const SSA_Opt *ssa, int i, int j)
    {
        if (!ssa || !ssa->decomposed || i < 0 || i >= ssa->n_components || j < 0 || j >= ssa->n_components)
            return 0.0;
        int N = ssa->N, L = ssa->L, K = ssa->K, fft_len = ssa->fft_len, r2c_len = ssa->r2c_len;
        SSA_Opt *ssa_mut = (SSA_Opt *)ssa;
        int use_cache = (ssa->U_fft != NULL && ssa->V_fft != NULL);
        double *h_i = (double *)ssa_opt_alloc(N * sizeof(double)), *h_j = (double *)ssa_opt_alloc(N * sizeof(double));
        double *u_fft_local = NULL, *v_fft_local = NULL;
        if (!use_cache)
        {
            u_fft_local = (double *)ssa_opt_alloc(2 * r2c_len * sizeof(double));
            v_fft_local = (double *)ssa_opt_alloc(2 * r2c_len * sizeof(double));
        }
        if (!h_i || !h_j || (!use_cache && (!u_fft_local || !v_fft_local)))
        {
            ssa_opt_free_ptr(h_i);
            ssa_opt_free_ptr(h_j);
            ssa_opt_free_ptr(u_fft_local);
            ssa_opt_free_ptr(v_fft_local);
            return 0.0;
        }
        {
            const double *u_fft_ptr, *v_fft_ptr;
            if (use_cache)
            {
                u_fft_ptr = &ssa->U_fft[i * 2 * r2c_len];
                v_fft_ptr = &ssa->V_fft[i * 2 * r2c_len];
            }
            else
            {
                const double *u_vec = &ssa->U[i * L];
                const double *v_vec = &ssa->V[i * K];
                ssa_opt_zero(ssa_mut->ws_real, fft_len);
                memcpy(ssa_mut->ws_real, u_vec, L * sizeof(double));
                DftiComputeForward(ssa_mut->fft_r2c, ssa_mut->ws_real, u_fft_local);
                ssa_opt_zero(ssa_mut->ws_real, fft_len);
                memcpy(ssa_mut->ws_real, v_vec, K * sizeof(double));
                DftiComputeForward(ssa_mut->fft_r2c, ssa_mut->ws_real, v_fft_local);
                u_fft_ptr = u_fft_local;
                v_fft_ptr = v_fft_local;
            }
            ssa_opt_complex_mul_r2c(u_fft_ptr, v_fft_ptr, ssa_mut->ws_complex, r2c_len);
            DftiComputeBackward(ssa_mut->fft_c2r, ssa_mut->ws_complex, ssa_mut->ws_real);
            memcpy(h_i, ssa_mut->ws_real, N * sizeof(double));
        }
        {
            const double *u_fft_ptr, *v_fft_ptr;
            if (use_cache)
            {
                u_fft_ptr = &ssa->U_fft[j * 2 * r2c_len];
                v_fft_ptr = &ssa->V_fft[j * 2 * r2c_len];
            }
            else
            {
                const double *u_vec = &ssa->U[j * L];
                const double *v_vec = &ssa->V[j * K];
                ssa_opt_zero(ssa_mut->ws_real, fft_len);
                memcpy(ssa_mut->ws_real, u_vec, L * sizeof(double));
                DftiComputeForward(ssa_mut->fft_r2c, ssa_mut->ws_real, u_fft_local);
                ssa_opt_zero(ssa_mut->ws_real, fft_len);
                memcpy(ssa_mut->ws_real, v_vec, K * sizeof(double));
                DftiComputeForward(ssa_mut->fft_r2c, ssa_mut->ws_real, v_fft_local);
                u_fft_ptr = u_fft_local;
                v_fft_ptr = v_fft_local;
            }
            ssa_opt_complex_mul_r2c(u_fft_ptr, v_fft_ptr, ssa_mut->ws_complex, r2c_len);
            DftiComputeBackward(ssa_mut->fft_c2r, ssa_mut->ws_complex, ssa_mut->ws_real);
            memcpy(h_j, ssa_mut->ws_real, N * sizeof(double));
        }
        double sigma_i = ssa->sigma[i], sigma_j = ssa->sigma[j], inner, norm_i_sq, norm_j_sq;
        ssa_opt_weighted_inner3(h_i, h_j, ssa->inv_diag_count, N, &inner, &norm_i_sq, &norm_j_sq);
        inner *= sigma_i * sigma_j;
        norm_i_sq *= sigma_i * sigma_i;
        norm_j_sq *= sigma_j * sigma_j;
        ssa_opt_free_ptr(h_i);
        ssa_opt_free_ptr(h_j);
        ssa_opt_free_ptr(u_fft_local);
        ssa_opt_free_ptr(v_fft_local);
        double denom = sqrt(norm_i_sq) * sqrt(norm_j_sq);
        return (denom > 1e-12) ? inner / denom : 0.0;
    }

    int ssa_opt_component_stats(const SSA_Opt *ssa, SSA_ComponentStats *stats)
    {
        if (!ssa || !ssa->decomposed || !stats)
            return -1;
        int n = ssa->n_components;
        if (n < 2)
            return -1;
        memset(stats, 0, sizeof(SSA_ComponentStats));
        stats->n = n;
        stats->singular_values = (double *)ssa_opt_alloc(n * sizeof(double));
        stats->log_sv = (double *)ssa_opt_alloc(n * sizeof(double));
        stats->gaps = (double *)ssa_opt_alloc((n - 1) * sizeof(double));
        stats->cumulative_var = (double *)ssa_opt_alloc(n * sizeof(double));
        stats->second_diff = (double *)ssa_opt_alloc(n * sizeof(double));
        if (!stats->singular_values || !stats->log_sv || !stats->gaps || !stats->cumulative_var || !stats->second_diff)
        {
            ssa_opt_component_stats_free(stats);
            return -1;
        }
        for (int i = 0; i < n; i++)
        {
            stats->singular_values[i] = ssa->sigma[i];
            stats->log_sv[i] = log(ssa->sigma[i] + 1e-300);
        }
        double max_gap = 0.0;
        int max_gap_idx = 0;
        for (int i = 0; i < n - 1; i++)
        {
            double gap = ssa->sigma[i] / (ssa->sigma[i + 1] + 1e-300);
            stats->gaps[i] = gap;
            if (gap > max_gap)
            {
                max_gap = gap;
                max_gap_idx = i;
            }
        }
        double cumsum = 0.0;
        for (int i = 0; i < n; i++)
        {
            cumsum += ssa->eigenvalues[i];
            stats->cumulative_var[i] = cumsum / ssa->total_variance;
        }
        stats->second_diff[0] = 0.0;
        stats->second_diff[n - 1] = 0.0;
        for (int i = 1; i < n - 1; i++)
            stats->second_diff[i] = stats->log_sv[i - 1] - 2.0 * stats->log_sv[i] + stats->log_sv[i + 1];
        stats->suggested_signal = max_gap_idx + 1;
        stats->gap_threshold = max_gap;
        return 0;
    }

    void ssa_opt_component_stats_free(SSA_ComponentStats *stats)
    {
        if (!stats)
            return;
        ssa_opt_free_ptr(stats->singular_values);
        ssa_opt_free_ptr(stats->log_sv);
        ssa_opt_free_ptr(stats->gaps);
        ssa_opt_free_ptr(stats->cumulative_var);
        ssa_opt_free_ptr(stats->second_diff);
        memset(stats, 0, sizeof(SSA_ComponentStats));
    }

    int ssa_opt_find_periodic_pairs(const SSA_Opt *ssa, int *pairs, int max_pairs,
                                    double sv_tol, double wcorr_thresh)
    {
        if (!ssa || !ssa->decomposed || !pairs || max_pairs < 1)
            return 0;

        // Default tolerances if not specified
        if (sv_tol <= 0)
            sv_tol = 0.1; // 10% singular value difference allowed
        if (wcorr_thresh <= 0)
            wcorr_thresh = 0.5; // 50% W-correlation minimum

        int n = ssa->n_components, n_pairs = 0;

        // Track which components have been paired (each can only be in one pair)
        bool *used = (bool *)calloc(n, sizeof(bool));
        if (!used)
            return 0;

        // Greedy pairing: scan components in order of decreasing singular value
        // (components are already sorted by σ descending)
        for (int i = 0; i < n - 1 && n_pairs < max_pairs; i++)
        {
            if (used[i])
                continue; // Already paired

            // Look for a partner among remaining components
            for (int j = i + 1; j < n && n_pairs < max_pairs; j++)
            {
                if (used[j])
                    continue; // Already paired

                // === Test 1: Singular Value Similarity ===
                // Sine/cosine pairs have nearly equal energy (σᵢ ≈ σⱼ)
                // Ratio should be close to 1.0
                double sv_ratio = ssa->sigma[j] / (ssa->sigma[i] + 1e-300);
                if (fabs(1.0 - sv_ratio) > sv_tol)
                    continue; // Singular values too different, not a pair

                // === Test 2: W-Correlation ===
                // Components representing the same frequency have high W-correlation
                // (they're "similar" in the weighted inner product sense)
                double wcorr = fabs(ssa_opt_wcorr_pair(ssa, i, j));
                if (wcorr < wcorr_thresh)
                    continue; // Not correlated enough, not a pair

                // === Found a Pair ===
                pairs[2 * n_pairs] = i;
                pairs[2 * n_pairs + 1] = j;
                used[i] = true;
                used[j] = true;
                n_pairs++;

                break; // Move to next i (each component pairs with at most one other)
            }
        }

        free(used);
        return n_pairs;
    }

    void ssa_opt_lrf_free(SSA_LRF *lrf)
    {
        if (!lrf)
            return;
        ssa_opt_free_ptr(lrf->R);
        memset(lrf, 0, sizeof(SSA_LRF));
    }

    int ssa_opt_compute_lrf(const SSA_Opt *ssa, const int *group, int n_group, SSA_LRF *lrf)
    {
        if (!ssa || !ssa->decomposed || !group || !lrf || n_group < 1)
            return -1;
        int L = ssa->L;
        memset(lrf, 0, sizeof(SSA_LRF));
        lrf->L = L;
        lrf->valid = false;
        double *pi = (double *)ssa_opt_alloc(n_group * sizeof(double));
        if (!pi)
            return -1;
        double nu_sq = 0.0;
        for (int g = 0; g < n_group; g++)
        {
            int idx = group[g];
            if (idx < 0 || idx >= ssa->n_components)
            {
                ssa_opt_free_ptr(pi);
                return -1;
            }
            pi[g] = ssa->U[idx * L + (L - 1)];
            nu_sq += pi[g] * pi[g];
        }
        lrf->verticality = nu_sq;
        if (nu_sq >= 1.0 - 1e-10)
        {
            ssa_opt_free_ptr(pi);
            lrf->valid = false;
            return -1;
        }
        lrf->R = (double *)ssa_opt_alloc((L - 1) * sizeof(double));
        if (!lrf->R)
        {
            ssa_opt_free_ptr(pi);
            return -1;
        }
        double scale = 1.0 / (1.0 - nu_sq);
        for (int j = 0; j < L - 1; j++)
        {
            double sum = 0.0;
            for (int g = 0; g < n_group; g++)
            {
                int idx = group[g];
                sum += pi[g] * ssa->U[idx * L + j];
            }
            lrf->R[j] = sum * scale;
        }
        ssa_opt_free_ptr(pi);
        lrf->valid = true;
        return 0;
    }

    int ssa_opt_forecast_with_lrf(const SSA_LRF *lrf, const double *base_signal, int base_len, int n_forecast, double *output)
    {
        if (!lrf || !lrf->valid || !lrf->R || !base_signal || !output)
            return -1;
        int L = lrf->L;
        if (base_len < L - 1 || n_forecast < 1)
            return -1;
        int window_size = L - 1;
        double *buffer = (double *)ssa_opt_alloc((window_size + n_forecast) * sizeof(double));
        if (!buffer)
            return -1;
        for (int i = 0; i < window_size; i++)
            buffer[i] = base_signal[base_len - window_size + i];
        for (int h = 0; h < n_forecast; h++)
        {
            double forecast = 0.0;
            for (int j = 0; j < window_size; j++)
                forecast += lrf->R[j] * buffer[h + j];
            buffer[window_size + h] = forecast;
            output[h] = forecast;
        }
        ssa_opt_free_ptr(buffer);
        return 0;
    }

    int ssa_opt_forecast(const SSA_Opt *ssa, const int *group, int n_group, int n_forecast, double *output)
    {
        if (!ssa || !ssa->decomposed || !group || !output || n_group < 1 || n_forecast < 1)
            return -1;
        int N = ssa->N;
        SSA_LRF lrf = {0};
        if (ssa_opt_compute_lrf(ssa, group, n_group, &lrf) != 0)
            return -1;
        double *reconstructed = (double *)ssa_opt_alloc(N * sizeof(double));
        if (!reconstructed)
        {
            ssa_opt_lrf_free(&lrf);
            return -1;
        }
        if (ssa_opt_reconstruct(ssa, group, n_group, reconstructed) != 0)
        {
            ssa_opt_free_ptr(reconstructed);
            ssa_opt_lrf_free(&lrf);
            return -1;
        }
        int result = ssa_opt_forecast_with_lrf(&lrf, reconstructed, N, n_forecast, output);
        ssa_opt_free_ptr(reconstructed);
        ssa_opt_lrf_free(&lrf);
        return result;
    }

    int ssa_opt_forecast_full(const SSA_Opt *ssa, const int *group, int n_group, int n_forecast, double *output)
    {
        if (!ssa || !ssa->decomposed || !group || !output || n_group < 1 || n_forecast < 1)
            return -1;
        int N = ssa->N;
        SSA_LRF lrf = {0};
        if (ssa_opt_compute_lrf(ssa, group, n_group, &lrf) != 0)
            return -1;
        if (ssa_opt_reconstruct(ssa, group, n_group, output) != 0)
        {
            ssa_opt_lrf_free(&lrf);
            return -1;
        }
        int result = ssa_opt_forecast_with_lrf(&lrf, output, N, n_forecast, output + N);
        ssa_opt_lrf_free(&lrf);
        return result;
    }

    // ========================================================================
    // Vector Forecast (V-forecast)
    // ========================================================================
    // Alternative to recurrent forecast. Instead of precomputing LRR coefficients,
    // V-forecast projects directly onto eigenvector subspace at each step.
    //
    // Formula: x_new = (1/(1-ν²)) * Σ_i π_i * (U_trunc_i · z)
    // where:
    //   z = last L-1 values of signal
    //   U_trunc_i = eigenvector i without last element
    //   π_i = last element of eigenvector i
    //   ν² = Σ π_i²
    //
    // Advantages over R-forecast:
    //   - Can be more numerically stable for long forecasts
    //   - Natural extension to weighted/oblique variants
    //   - No need to store LRR coefficients
    // ========================================================================

    int ssa_opt_vforecast(const SSA_Opt *ssa, const int *group, int n_group, int n_forecast, double *output)
    {
        if (!ssa || !ssa->decomposed || !group || !output || n_group < 1 || n_forecast < 1)
            return -1;
        int N = ssa->N, L = ssa->L;

        // Compute verticality ν² = Σ π_i²
        double nu_sq = 0.0;
        for (int g = 0; g < n_group; g++)
        {
            int idx = group[g];
            if (idx < 0 || idx >= ssa->n_components)
                return -1;
            double pi_i = ssa->U[idx * L + (L - 1)];
            nu_sq += pi_i * pi_i;
        }
        if (nu_sq >= 1.0 - 1e-10)
            return -1; // Not forecastable (vertical eigenvectors)
        double scale = 1.0 / (1.0 - nu_sq);

        // Get reconstructed signal and allocate space for forecasts
        double *signal = (double *)ssa_opt_alloc((N + n_forecast) * sizeof(double));
        if (!signal)
            return -1;
        if (ssa_opt_reconstruct(ssa, group, n_group, signal) != 0)
        {
            ssa_opt_free_ptr(signal);
            return -1;
        }

        // V-forecast loop
        for (int h = 0; h < n_forecast; h++)
        {
            double *z = &signal[N + h - (L - 1)]; // Last L-1 values
            double x_new = 0.0;

            // For each component in group: x_new += π_i * (U_trunc_i · z)
            for (int g = 0; g < n_group; g++)
            {
                int idx = group[g];
                const double *u_col = &ssa->U[idx * L];
                double pi_i = u_col[L - 1];

                // Inner product of U_trunc (first L-1 elements) with z
                double dot = 0.0;
                for (int j = 0; j < L - 1; j++)
                {
                    dot += u_col[j] * z[j];
                }
                x_new += pi_i * dot;
            }
            x_new *= scale;

            signal[N + h] = x_new;
            output[h] = x_new;
        }

        ssa_opt_free_ptr(signal);
        return 0;
    }

    int ssa_opt_vforecast_full(const SSA_Opt *ssa, const int *group, int n_group, int n_forecast, double *output)
    {
        if (!ssa || !ssa->decomposed || !group || !output || n_group < 1 || n_forecast < 1)
            return -1;
        int N = ssa->N, L = ssa->L;

        // Compute verticality
        double nu_sq = 0.0;
        for (int g = 0; g < n_group; g++)
        {
            int idx = group[g];
            if (idx < 0 || idx >= ssa->n_components)
                return -1;
            double pi_i = ssa->U[idx * L + (L - 1)];
            nu_sq += pi_i * pi_i;
        }
        if (nu_sq >= 1.0 - 1e-10)
            return -1;
        double scale = 1.0 / (1.0 - nu_sq);

        // Reconstruct into output buffer (first N values)
        if (ssa_opt_reconstruct(ssa, group, n_group, output) != 0)
            return -1;

        // V-forecast loop (appending to output)
        for (int h = 0; h < n_forecast; h++)
        {
            double *z = &output[N + h - (L - 1)];
            double x_new = 0.0;

            for (int g = 0; g < n_group; g++)
            {
                int idx = group[g];
                const double *u_col = &ssa->U[idx * L];
                double pi_i = u_col[L - 1];

                double dot = 0.0;
                for (int j = 0; j < L - 1; j++)
                {
                    dot += u_col[j] * z[j];
                }
                x_new += pi_i * dot;
            }
            x_new *= scale;
            output[N + h] = x_new;
        }

        return 0;
    }

    // Optimized V-forecast for hot path - uses BLAS for inner products
    int ssa_opt_vforecast_fast(const SSA_Opt *ssa, const int *group, int n_group,
                               const double *base_signal, int base_len, int n_forecast, double *output)
    {
        if (!ssa || !ssa->decomposed || !group || !base_signal || !output)
            return -1;
        if (n_group < 1 || n_forecast < 1 || base_len < ssa->L - 1)
            return -1;
        int L = ssa->L;

        // Precompute π values and ν²
        double *pi = (double *)ssa_opt_alloc(n_group * sizeof(double));
        if (!pi)
            return -1;
        double nu_sq = 0.0;
        for (int g = 0; g < n_group; g++)
        {
            int idx = group[g];
            if (idx < 0 || idx >= ssa->n_components)
            {
                ssa_opt_free_ptr(pi);
                return -1;
            }
            pi[g] = ssa->U[idx * L + (L - 1)];
            nu_sq += pi[g] * pi[g];
        }
        if (nu_sq >= 1.0 - 1e-10)
        {
            ssa_opt_free_ptr(pi);
            return -1;
        }
        double scale = 1.0 / (1.0 - nu_sq);

        // Working buffer for signal extension
        int window = L - 1;
        double *buffer = (double *)ssa_opt_alloc((window + n_forecast) * sizeof(double));
        if (!buffer)
        {
            ssa_opt_free_ptr(pi);
            return -1;
        }

        // Copy last L-1 values of base signal
        memcpy(buffer, &base_signal[base_len - window], window * sizeof(double));

        // V-forecast loop with BLAS
        for (int h = 0; h < n_forecast; h++)
        {
            double *z = &buffer[h];
            double x_new = 0.0;

            for (int g = 0; g < n_group; g++)
            {
                int idx = group[g];
                // BLAS dot product: U_trunc · z
                double dot = cblas_ddot(window, &ssa->U[idx * L], 1, z, 1);
                x_new += pi[g] * dot;
            }
            x_new *= scale;

            buffer[window + h] = x_new;
            output[h] = x_new;
        }

        ssa_opt_free_ptr(pi);
        ssa_opt_free_ptr(buffer);
        return 0;
    }

    int ssa_opt_get_trend(const SSA_Opt *ssa, double *output)
    {
        int group[] = {0};
        return ssa_opt_reconstruct(ssa, group, 1, output);
    }
    int ssa_opt_get_noise(const SSA_Opt *ssa, int noise_start, double *output)
    {
        if (!ssa || !ssa->decomposed || noise_start < 0)
            return -1;
        int n_noise = ssa->n_components - noise_start;
        if (n_noise <= 0)
        {
            ssa_opt_zero(output, ssa->N);
            return 0;
        }
        int *group = (int *)malloc(n_noise * sizeof(int));
        if (!group)
            return -1;
        for (int i = 0; i < n_noise; i++)
            group[i] = noise_start + i;
        int ret = ssa_opt_reconstruct(ssa, group, n_noise, output);
        free(group);
        return ret;
    }
    double ssa_opt_variance_explained(const SSA_Opt *ssa, int start, int end)
    {
        if (!ssa || !ssa->decomposed || start < 0 || ssa->total_variance <= 0)
            return 0.0;
        if (end < 0 || end >= ssa->n_components)
            end = ssa->n_components - 1;
        double sum = 0;
        for (int i = start; i <= end; i++)
            sum += ssa->eigenvalues[i];
        return sum / ssa->total_variance;
    }

    int ssa_opt_get_singular_values(const SSA_Opt *ssa, double *output, int max_n)
    {
        if (!ssa || !ssa->decomposed || !output)
            return -1;
        int n = ssa_opt_min(max_n, ssa->n_components);
        memcpy(output, ssa->sigma, n * sizeof(double));
        return n;
    }

    int ssa_opt_get_eigenvalues(const SSA_Opt *ssa, double *output, int max_n)
    {
        if (!ssa || !ssa->decomposed || !output)
            return -1;
        int n = ssa_opt_min(max_n, ssa->n_components);
        memcpy(output, ssa->eigenvalues, n * sizeof(double));
        return n;
    }

    double ssa_opt_get_total_variance(const SSA_Opt *ssa)
    {
        if (!ssa || !ssa->decomposed)
            return 0.0;
        return ssa->total_variance;
    }

    // ========================================================================
    // Cadzow Iterations (Finite-Rank Signal Approximation)
    // ========================================================================
    // Iteratively projects signal onto the space of signals with exactly
    // rank-r trajectory matrices. Each iteration:
    //   1. Form trajectory matrix
    //   2. SVD → keep rank r (low-rank projection)
    //   3. Hankel averaging → back to signal (Hankel projection)
    //
    // Converges to signal whose trajectory matrix is EXACTLY rank r.
    // Useful for:
    //   - Denoising (cleaner than single-pass SSA)
    //   - Enforcing exact periodicity structure
    //   - Signal recovery from corrupted data
    //
    // Returns: number of iterations performed, or -1 on error
    // ========================================================================

    int ssa_opt_cadzow(const double *x, int N, int L, int rank, int max_iter, double tol,
                       double *output, SSA_CadzowResult *result)
    {
        if (!x || !output || N < 4 || L < 2 || L > N - 1 || rank < 1 || max_iter < 1)
            return -1;

        // Working buffers
        double *y = (double *)ssa_opt_alloc(N * sizeof(double));
        double *y_new = (double *)ssa_opt_alloc(N * sizeof(double));
        int *group = (int *)malloc(rank * sizeof(int));
        if (!y || !y_new || !group)
        {
            ssa_opt_free_ptr(y);
            ssa_opt_free_ptr(y_new);
            free(group);
            return -1;
        }

        // Group = [0, 1, ..., rank-1]
        for (int i = 0; i < rank; i++)
            group[i] = i;

        // Initialize y = x
        memcpy(y, x, N * sizeof(double));

        // Compute initial signal norm for relative tolerance
        double y_norm = cblas_dnrm2(N, y, 1);
        if (y_norm < 1e-15)
            y_norm = 1.0;

        int iter;
        double diff = 0.0;
        bool converged = false;

        for (iter = 0; iter < max_iter; iter++)
        {
            // Initialize SSA with current signal
            SSA_Opt ssa = {0};
            if (ssa_opt_init(&ssa, y, N, L) != 0)
            {
                ssa_opt_free_ptr(y);
                ssa_opt_free_ptr(y_new);
                free(group);
                return -1;
            }

            // Decompose to get rank components
            // Use randomized for speed if rank is small relative to L
            int K = N - L + 1;
            int use_randomized = (rank + 8 < ssa_opt_min(L, K) / 2);

            if (use_randomized)
            {
                if (ssa_opt_prepare(&ssa, rank, 8) != 0 ||
                    ssa_opt_decompose_randomized(&ssa, rank, 8) != 0)
                {
                    ssa_opt_free(&ssa);
                    // Fallback to block method
                    use_randomized = 0;
                }
            }

            if (!use_randomized)
            {
                if (ssa_opt_decompose_block(&ssa, rank, ssa_opt_min(rank, 16), 3) != 0)
                {
                    ssa_opt_free(&ssa);
                    ssa_opt_free_ptr(y);
                    ssa_opt_free_ptr(y_new);
                    free(group);
                    return -1;
                }
            }

            // Reconstruct with rank components (Hankel averaging)
            if (ssa_opt_reconstruct(&ssa, group, rank, y_new) != 0)
            {
                ssa_opt_free(&ssa);
                ssa_opt_free_ptr(y);
                ssa_opt_free_ptr(y_new);
                free(group);
                return -1;
            }

            ssa_opt_free(&ssa);

            // Compute difference ||y_new - y||
            double diff_sq = 0.0;
            for (int i = 0; i < N; i++)
            {
                double d = y_new[i] - y[i];
                diff_sq += d * d;
            }
            diff = sqrt(diff_sq) / y_norm;

            // Swap buffers
            double *tmp = y;
            y = y_new;
            y_new = tmp;

            // Check convergence
            if (diff < tol)
            {
                converged = true;
                iter++; // Count this iteration
                break;
            }
        }

        // Copy result to output
        memcpy(output, y, N * sizeof(double));

        // Fill result struct if provided
        if (result)
        {
            result->iterations = iter;
            result->final_diff = diff;
            result->converged = converged ? 1.0 : 0.0;
        }

        ssa_opt_free_ptr(y);
        ssa_opt_free_ptr(y_new);
        free(group);

        return iter;
    }

    // Weighted Cadzow with alpha blending (for regularization)
    // output = alpha * cadzow_result + (1-alpha) * original
    int ssa_opt_cadzow_weighted(const double *x, int N, int L, int rank, int max_iter,
                                double tol, double alpha, double *output, SSA_CadzowResult *result)
    {
        if (alpha <= 0.0 || alpha > 1.0)
            return -1;

        int ret = ssa_opt_cadzow(x, N, L, rank, max_iter, tol, output, result);
        if (ret < 0)
            return ret;

        // Blend if alpha < 1
        if (alpha < 1.0)
        {
            double beta = 1.0 - alpha;
            for (int i = 0; i < N; i++)
            {
                output[i] = alpha * output[i] + beta * x[i];
            }
        }

        return ret;
    }

    // In-place Cadzow using existing SSA object (faster for repeated calls)
    // Modifies the SSA object's signal and redecomposes
    int ssa_opt_cadzow_inplace(SSA_Opt *ssa, int rank, int max_iter, double tol, SSA_CadzowResult *result)
    {
        if (!ssa || !ssa->initialized || rank < 1 || max_iter < 1)
            return -1;
        int N = ssa->N, L = ssa->L;

        double *y_new = (double *)ssa_opt_alloc(N * sizeof(double));
        double *y_current = (double *)ssa_opt_alloc(N * sizeof(double));
        int *group = (int *)malloc(rank * sizeof(int));
        if (!y_new || !y_current || !group)
        {
            ssa_opt_free_ptr(y_new);
            ssa_opt_free_ptr(y_current);
            free(group);
            return -1;
        }

        for (int i = 0; i < rank; i++)
            group[i] = i;

        // Get initial signal from FFT (inverse of what init does)
        // Actually we need to store original signal - use ws_real2 as temp
        // Simpler: reconstruct all components to get current signal approximation
        // But for first iter, we need original signal...

        // Let's use a different approach: caller should have original signal
        // We'll extract it by doing full reconstruction if already decomposed

        double y_norm = 1.0;
        int iter;
        double diff = 0.0;
        bool converged = false;

        for (iter = 0; iter < max_iter; iter++)
        {
            // Decompose (or re-decompose)
            int K = N - L + 1;
            int use_randomized = (rank + 8 < ssa_opt_min(L, K) / 2);

            // Clear previous decomposition
            ssa->decomposed = false;
            ssa->n_components = 0;
            ssa->total_variance = 0.0;
            ssa_opt_free_cached_ffts(ssa);

            if (use_randomized && ssa->prepared && rank + 8 <= ssa->prepared_kp)
            {
                if (ssa_opt_decompose_randomized(ssa, rank, 8) != 0)
                {
                    use_randomized = 0;
                }
            }

            if (!use_randomized || !ssa->decomposed)
            {
                if (ssa_opt_decompose_block(ssa, rank, ssa_opt_min(rank, 16), 3) != 0)
                {
                    ssa_opt_free_ptr(y_new);
                    ssa_opt_free_ptr(y_current);
                    free(group);
                    return -1;
                }
            }

            // Save current reconstruction for comparison
            if (iter > 0)
            {
                memcpy(y_current, y_new, N * sizeof(double));
            }

            // Reconstruct
            if (ssa_opt_reconstruct(ssa, group, rank, y_new) != 0)
            {
                ssa_opt_free_ptr(y_new);
                ssa_opt_free_ptr(y_current);
                free(group);
                return -1;
            }

            if (iter == 0)
            {
                y_norm = cblas_dnrm2(N, y_new, 1);
                if (y_norm < 1e-15)
                    y_norm = 1.0;
            }
            else
            {
                // Compute difference
                double diff_sq = 0.0;
                for (int i = 0; i < N; i++)
                {
                    double d = y_new[i] - y_current[i];
                    diff_sq += d * d;
                }
                diff = sqrt(diff_sq) / y_norm;

                if (diff < tol)
                {
                    converged = true;
                    iter++;
                    break;
                }
            }

            // Update signal for next iteration
            if (iter < max_iter - 1)
            {
                ssa_opt_update_signal(ssa, y_new);
            }
        }

        if (result)
        {
            result->iterations = iter;
            result->final_diff = diff;
            result->converged = converged ? 1.0 : 0.0;
        }

        ssa_opt_free_ptr(y_new);
        ssa_opt_free_ptr(y_current);
        free(group);

        return iter;
    }

    // ========================================================================
    // Gap Filling - Handle Missing Values in Time Series
    // ========================================================================

    // Helper: check if value is NaN
    static inline int ssa_is_nan(double x)
    {
        return isnan(x);
    }

    // Helper: linear interpolation for initial gap filling
    static void ssa_linear_interp_gaps(double *x, const int *gap_mask, int N)
    {
        int i = 0;
        while (i < N)
        {
            if (gap_mask[i])
            {
                // Find gap boundaries
                int gap_start = i;
                while (i < N && gap_mask[i])
                    i++;
                int gap_end = i; // exclusive

                // Find left and right valid values
                double left_val = 0.0, right_val = 0.0;
                int left_idx = gap_start - 1;
                int right_idx = gap_end;

                if (left_idx >= 0)
                    left_val = x[left_idx];
                else
                    left_val = (right_idx < N) ? x[right_idx] : 0.0;

                if (right_idx < N)
                    right_val = x[right_idx];
                else
                    right_val = left_val;

                // Linear interpolation
                int gap_len = gap_end - gap_start;
                for (int j = 0; j < gap_len; j++)
                {
                    double t = (gap_len > 1) ? (double)(j + 1) / (gap_len + 1) : 0.5;
                    x[gap_start + j] = left_val + t * (right_val - left_val);
                }
            }
            else
            {
                i++;
            }
        }
    }

    int ssa_opt_gapfill(double *x, int N, int L, int rank, int max_iter, double tol,
                        SSA_GapFillResult *result)
    {
        if (!x || N < 4 || L < 2 || L > N - 1 || rank < 1)
            return -1;

        // Identify gaps (NaN positions)
        int *gap_mask = (int *)calloc(N, sizeof(int));
        if (!gap_mask)
            return -1;

        int n_gaps = 0;
        for (int i = 0; i < N; i++)
        {
            if (ssa_is_nan(x[i]))
            {
                gap_mask[i] = 1;
                n_gaps++;
            }
        }

        // No gaps to fill
        if (n_gaps == 0)
        {
            free(gap_mask);
            if (result)
            {
                result->iterations = 0;
                result->final_diff = 0.0;
                result->converged = 1;
                result->n_gaps = 0;
            }
            return 0;
        }

        // Check we have enough non-gap data
        if (N - n_gaps < L + rank)
        {
            free(gap_mask);
            return -1; // Not enough data for SSA
        }

        // Initialize gaps with linear interpolation
        ssa_linear_interp_gaps(x, gap_mask, N);

        // Save gap values for convergence check
        double *gap_values_old = (double *)ssa_opt_alloc(n_gaps * sizeof(double));
        double *gap_values_new = (double *)ssa_opt_alloc(n_gaps * sizeof(double));
        if (!gap_values_old || !gap_values_new)
        {
            free(gap_mask);
            if (gap_values_old)
                ssa_opt_free_ptr(gap_values_old);
            if (gap_values_new)
                ssa_opt_free_ptr(gap_values_new);
            return -1;
        }

        // Extract initial gap values
        int gi = 0;
        for (int i = 0; i < N; i++)
        {
            if (gap_mask[i])
                gap_values_old[gi++] = x[i];
        }

        // Allocate reconstruction buffer
        double *recon = (double *)ssa_opt_alloc(N * sizeof(double));
        if (!recon)
        {
            free(gap_mask);
            ssa_opt_free_ptr(gap_values_old);
            ssa_opt_free_ptr(gap_values_new);
            return -1;
        }

        // Iterative gap filling
        int iter;
        double diff = 1e10;
        int converged = 0;

        // Create component group
        int *group = (int *)malloc(rank * sizeof(int));
        if (!group)
        {
            free(gap_mask);
            ssa_opt_free_ptr(gap_values_old);
            ssa_opt_free_ptr(gap_values_new);
            ssa_opt_free_ptr(recon);
            return -1;
        }
        for (int i = 0; i < rank; i++)
            group[i] = i;

        for (iter = 0; iter < max_iter; iter++)
        {
            // SSA decomposition on current signal
            SSA_Opt ssa;
            if (ssa_opt_init(&ssa, x, N, L) != 0)
            {
                free(gap_mask);
                free(group);
                ssa_opt_free_ptr(gap_values_old);
                ssa_opt_free_ptr(gap_values_new);
                ssa_opt_free_ptr(recon);
                return -1;
            }

            // Prepare and decompose
            ssa_opt_prepare(&ssa, rank, 8);
            if (ssa_opt_decompose_randomized(&ssa, rank, 8) != 0)
            {
                ssa_opt_free(&ssa);
                free(gap_mask);
                free(group);
                ssa_opt_free_ptr(gap_values_old);
                ssa_opt_free_ptr(gap_values_new);
                ssa_opt_free_ptr(recon);
                return -1;
            }

            // Reconstruct
            if (ssa_opt_reconstruct(&ssa, group, rank, recon) != 0)
            {
                ssa_opt_free(&ssa);
                free(gap_mask);
                free(group);
                ssa_opt_free_ptr(gap_values_old);
                ssa_opt_free_ptr(gap_values_new);
                ssa_opt_free_ptr(recon);
                return -1;
            }

            ssa_opt_free(&ssa);

            // Update only gap positions
            gi = 0;
            for (int i = 0; i < N; i++)
            {
                if (gap_mask[i])
                {
                    gap_values_new[gi++] = recon[i];
                    x[i] = recon[i];
                }
            }

            // Check convergence: relative change in gap values
            double sum_sq_diff = 0.0, sum_sq_old = 0.0;
            for (int i = 0; i < n_gaps; i++)
            {
                double d = gap_values_new[i] - gap_values_old[i];
                sum_sq_diff += d * d;
                sum_sq_old += gap_values_old[i] * gap_values_old[i];
            }

            diff = (sum_sq_old > 1e-12) ? sqrt(sum_sq_diff / sum_sq_old) : sqrt(sum_sq_diff);

            if (diff < tol)
            {
                converged = 1;
                iter++; // Count this iteration
                break;
            }

            // Swap old/new
            memcpy(gap_values_old, gap_values_new, n_gaps * sizeof(double));
        }

        // Fill result
        if (result)
        {
            result->iterations = iter;
            result->final_diff = diff;
            result->converged = converged;
            result->n_gaps = n_gaps;
        }

        // Cleanup
        free(gap_mask);
        free(group);
        ssa_opt_free_ptr(gap_values_old);
        ssa_opt_free_ptr(gap_values_new);
        ssa_opt_free_ptr(recon);

        return 0;
    }

    int ssa_opt_gapfill_simple(double *x, int N, int L, int rank, SSA_GapFillResult *result)
    {
        if (!x || N < 4 || L < 2 || L > N - 1 || rank < 1)
            return -1;

        // Identify gaps
        int *gap_mask = (int *)calloc(N, sizeof(int));
        if (!gap_mask)
            return -1;

        int n_gaps = 0;
        for (int i = 0; i < N; i++)
        {
            if (ssa_is_nan(x[i]))
            {
                gap_mask[i] = 1;
                n_gaps++;
            }
        }

        if (n_gaps == 0)
        {
            free(gap_mask);
            if (result)
            {
                result->iterations = 1;
                result->final_diff = 0.0;
                result->converged = 1;
                result->n_gaps = 0;
            }
            return 0;
        }

        // Find contiguous segments without gaps
        // For each gap, forecast from left and backcast from right

        // Create group
        int *group = (int *)malloc(rank * sizeof(int));
        if (!group)
        {
            free(gap_mask);
            return -1;
        }
        for (int i = 0; i < rank; i++)
            group[i] = i;

        // Process each contiguous gap region
        int i = 0;
        while (i < N)
        {
            if (!gap_mask[i])
            {
                i++;
                continue;
            }

            // Found gap start
            int gap_start = i;
            while (i < N && gap_mask[i])
                i++;
            int gap_end = i; // exclusive
            int gap_len = gap_end - gap_start;

            // Get left segment (before gap)
            int left_len = gap_start;
            // Get right segment (after gap)
            int right_start = gap_end;
            int right_len = N - gap_end;

            double *forecast_left = NULL;
            double *forecast_right = NULL;

            // Forward forecast from left segment (if long enough)
            if (left_len >= L + rank)
            {
                SSA_Opt ssa_left;
                double *left_data = (double *)ssa_opt_alloc(left_len * sizeof(double));
                if (left_data)
                {
                    memcpy(left_data, x, left_len * sizeof(double));

                    if (ssa_opt_init(&ssa_left, left_data, left_len, L) == 0)
                    {
                        ssa_opt_prepare(&ssa_left, rank, 8);
                        if (ssa_opt_decompose_randomized(&ssa_left, rank, 8) == 0)
                        {
                            forecast_left = (double *)ssa_opt_alloc(gap_len * sizeof(double));
                            if (forecast_left)
                            {
                                ssa_opt_forecast(&ssa_left, group, rank, gap_len, forecast_left);
                            }
                        }
                        ssa_opt_free(&ssa_left);
                    }
                    ssa_opt_free_ptr(left_data);
                }
            }

            // Backward forecast from right segment (if long enough)
            if (right_len >= L + rank)
            {
                SSA_Opt ssa_right;
                // Reverse the right segment for backward forecasting
                double *right_data = (double *)ssa_opt_alloc(right_len * sizeof(double));
                if (right_data)
                {
                    // Copy reversed
                    for (int j = 0; j < right_len; j++)
                    {
                        right_data[j] = x[N - 1 - j];
                    }

                    if (ssa_opt_init(&ssa_right, right_data, right_len, L) == 0)
                    {
                        ssa_opt_prepare(&ssa_right, rank, 8);
                        if (ssa_opt_decompose_randomized(&ssa_right, rank, 8) == 0)
                        {
                            double *backcast_rev = (double *)ssa_opt_alloc(gap_len * sizeof(double));
                            if (backcast_rev)
                            {
                                ssa_opt_forecast(&ssa_right, group, rank, gap_len, backcast_rev);
                                // Reverse backcast to get forward order
                                forecast_right = (double *)ssa_opt_alloc(gap_len * sizeof(double));
                                if (forecast_right)
                                {
                                    for (int j = 0; j < gap_len; j++)
                                    {
                                        forecast_right[j] = backcast_rev[gap_len - 1 - j];
                                    }
                                }
                                ssa_opt_free_ptr(backcast_rev);
                            }
                        }
                        ssa_opt_free(&ssa_right);
                    }
                    ssa_opt_free_ptr(right_data);
                }
            }

            // Fill gap with weighted average of forecasts
            for (int j = 0; j < gap_len; j++)
            {
                double val = 0.0;
                double weight = 0.0;

                if (forecast_left)
                {
                    // Weight decreases with distance from left
                    double w = (double)(gap_len - j) / gap_len;
                    val += w * forecast_left[j];
                    weight += w;
                }

                if (forecast_right)
                {
                    // Weight increases with distance from left (closer to right)
                    double w = (double)(j + 1) / gap_len;
                    val += w * forecast_right[j];
                    weight += w;
                }

                if (weight > 0)
                {
                    x[gap_start + j] = val / weight;
                }
                else
                {
                    // Fallback: simple average of neighbors
                    double left_val = (gap_start > 0) ? x[gap_start - 1] : 0.0;
                    double right_val = (gap_end < N) ? x[gap_end] : left_val;
                    x[gap_start + j] = (left_val + right_val) / 2.0;
                }
            }

            if (forecast_left)
                ssa_opt_free_ptr(forecast_left);
            if (forecast_right)
                ssa_opt_free_ptr(forecast_right);
        }

        // Result
        if (result)
        {
            result->iterations = 1;
            result->final_diff = 0.0;
            result->converged = 1;
            result->n_gaps = n_gaps;
        }

        free(gap_mask);
        free(group);

        return 0;
    }

    // MSSA Implementation
    void mssa_opt_free(MSSA_Opt *mssa)
    {
        if (!mssa)
            return;
        if (mssa->fft_r2c)
            DftiFreeDescriptor(&mssa->fft_r2c);
        if (mssa->fft_c2r)
            DftiFreeDescriptor(&mssa->fft_c2r);
        if (mssa->fft_r2c_batch)
            DftiFreeDescriptor(&mssa->fft_r2c_batch);
        if (mssa->fft_c2r_batch)
            DftiFreeDescriptor(&mssa->fft_c2r_batch);
        if (mssa->rng)
            vslDeleteStream(&mssa->rng);
        ssa_opt_free_ptr(mssa->fft_x);
        ssa_opt_free_ptr(mssa->ws_real);
        ssa_opt_free_ptr(mssa->ws_complex);
        ssa_opt_free_ptr(mssa->ws_batch_real);
        ssa_opt_free_ptr(mssa->ws_batch_complex);
        ssa_opt_free_ptr(mssa->U);
        ssa_opt_free_ptr(mssa->V);
        ssa_opt_free_ptr(mssa->sigma);
        ssa_opt_free_ptr(mssa->eigenvalues);
        ssa_opt_free_ptr(mssa->inv_diag_count);
        memset(mssa, 0, sizeof(MSSA_Opt));
    }

    int mssa_opt_init(MSSA_Opt *mssa, const double *X, int M, int N, int L)
    {
        // === Parameter Validation ===
        // M >= 1: at least one series (M=1 degenerates to standard SSA)
        // N >= 4: minimum viable length
        // L >= 2, L <= N-1: same constraints as standard SSA
        if (!mssa || !X || M < 1 || N < 4 || L < 2 || L > N - 1)
            return -1;

        // Zero-initialize entire structure
        memset(mssa, 0, sizeof(MSSA_Opt));

        // === Store Dimensions ===
        mssa->M = M;         // Number of series
        mssa->N = N;         // Length of each series
        mssa->L = L;         // Window length
        mssa->K = N - L + 1; // Number of lagged copies (same for all series)

        // FFT length: same calculation as SSA
        // Convolution result has length N + K - 1, round to power of 2
        int conv_len = N + mssa->K - 1;
        int fft_n = ssa_opt_next_pow2(conv_len);
        mssa->fft_len = fft_n;
        mssa->r2c_len = fft_n / 2 + 1; // Hermitian symmetry

        // === Allocate Workspaces ===
        // Single-vector workspaces (same as SSA)
        mssa->ws_real = (double *)ssa_opt_alloc(fft_n * sizeof(double));
        mssa->ws_complex = (double *)ssa_opt_alloc(2 * mssa->r2c_len * sizeof(double));

        // Batched workspaces: sized for M series (not SSA_BATCH_SIZE)
        // Used for parallel FFT across all M channels
        mssa->ws_batch_real = (double *)ssa_opt_alloc(M * fft_n * sizeof(double));
        mssa->ws_batch_complex = (double *)ssa_opt_alloc(M * 2 * mssa->r2c_len * sizeof(double));

        // Pre-computed FFT of all M signals
        // fft_x[m] = FFT(X[m, :]) for each series m
        mssa->fft_x = (double *)ssa_opt_alloc(M * 2 * mssa->r2c_len * sizeof(double));

        if (!mssa->ws_real || !mssa->ws_complex || !mssa->ws_batch_real ||
            !mssa->ws_batch_complex || !mssa->fft_x)
        {
            mssa_opt_free(mssa);
            return -1;
        }

        // === Create FFT Descriptors ===
        MKL_LONG status;

        // --- Single-vector R2C (for one series at a time) ---
        status = DftiCreateDescriptor(&mssa->fft_r2c, DFTI_DOUBLE, DFTI_REAL, 1, fft_n);
        if (status != 0)
        {
            mssa_opt_free(mssa);
            return -1;
        }
        DftiSetValue(mssa->fft_r2c, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
        DftiSetValue(mssa->fft_r2c, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
        if (DftiCommitDescriptor(mssa->fft_r2c) != 0)
        {
            mssa_opt_free(mssa);
            return -1;
        }

        // --- Single-vector C2R (inverse) ---
        status = DftiCreateDescriptor(&mssa->fft_c2r, DFTI_DOUBLE, DFTI_REAL, 1, fft_n);
        if (status != 0)
        {
            mssa_opt_free(mssa);
            return -1;
        }
        DftiSetValue(mssa->fft_c2r, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
        DftiSetValue(mssa->fft_c2r, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
        DftiSetValue(mssa->fft_c2r, DFTI_BACKWARD_SCALE, 1.0 / fft_n);
        if (DftiCommitDescriptor(mssa->fft_c2r) != 0)
        {
            mssa_opt_free(mssa);
            return -1;
        }

        // --- Batched R2C: process all M series in one MKL call ---
        // Used for parallel Hankel matvec across all channels
        status = DftiCreateDescriptor(&mssa->fft_r2c_batch, DFTI_DOUBLE, DFTI_REAL, 1, fft_n);
        if (status != 0)
        {
            mssa_opt_free(mssa);
            return -1;
        }
        DftiSetValue(mssa->fft_r2c_batch, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
        DftiSetValue(mssa->fft_r2c_batch, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
        // NUMBER_OF_TRANSFORMS = M (one FFT per series)
        DftiSetValue(mssa->fft_r2c_batch, DFTI_NUMBER_OF_TRANSFORMS, (MKL_LONG)M);
        DftiSetValue(mssa->fft_r2c_batch, DFTI_INPUT_DISTANCE, (MKL_LONG)fft_n);
        DftiSetValue(mssa->fft_r2c_batch, DFTI_OUTPUT_DISTANCE, (MKL_LONG)mssa->r2c_len);
        if (DftiCommitDescriptor(mssa->fft_r2c_batch) != 0)
        {
            mssa_opt_free(mssa);
            return -1;
        }

        // --- Batched C2R: inverse for all M series ---
        status = DftiCreateDescriptor(&mssa->fft_c2r_batch, DFTI_DOUBLE, DFTI_REAL, 1, fft_n);
        if (status != 0)
        {
            mssa_opt_free(mssa);
            return -1;
        }
        DftiSetValue(mssa->fft_c2r_batch, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
        DftiSetValue(mssa->fft_c2r_batch, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
        DftiSetValue(mssa->fft_c2r_batch, DFTI_BACKWARD_SCALE, 1.0 / fft_n);
        DftiSetValue(mssa->fft_c2r_batch, DFTI_NUMBER_OF_TRANSFORMS, (MKL_LONG)M);
        DftiSetValue(mssa->fft_c2r_batch, DFTI_INPUT_DISTANCE, (MKL_LONG)mssa->r2c_len);
        DftiSetValue(mssa->fft_c2r_batch, DFTI_OUTPUT_DISTANCE, (MKL_LONG)fft_n);
        if (DftiCommitDescriptor(mssa->fft_c2r_batch) != 0)
        {
            mssa_opt_free(mssa);
            return -1;
        }

        // === Pre-compute FFT of All M Signals ===
        // Each series gets its own cached FFT for fast Hankel matvec
        // fft_x layout: [FFT(X₀), FFT(X₁), ..., FFT(Xₘ₋₁)]
        for (int m = 0; m < M; m++)
        {
            double *fft_xm = mssa->fft_x + m * 2 * mssa->r2c_len; // Destination for series m

            // Zero-pad and copy series m
            ssa_opt_zero(mssa->ws_real, fft_n);
            memcpy(mssa->ws_real, X + m * N, N * sizeof(double)); // X[m, :]

            // Compute and store FFT
            DftiComputeForward(mssa->fft_r2c, mssa->ws_real, fft_xm);
        }

        // === Initialize RNG ===
        // MT19937 for randomized SVD (different BRNG than SSA for variety)
        if (vslNewStream(&mssa->rng, VSL_BRNG_MT19937, 42) != 0)
        {
            mssa_opt_free(mssa);
            return -1;
        }

        // === Pre-compute Diagonal Averaging Weights ===
        // Same formula as SSA - reconstruction uses same diagonal averaging
        // (weights are identical for all M series since they have same N, L, K)
        mssa->inv_diag_count = (double *)ssa_opt_alloc(N * sizeof(double));
        if (!mssa->inv_diag_count)
        {
            mssa_opt_free(mssa);
            return -1;
        }
        for (int t = 0; t < N; t++)
        {
            int count = ssa_opt_min(ssa_opt_min(t + 1, L), ssa_opt_min(mssa->K, N - t));
            mssa->inv_diag_count[t] = (count > 0) ? 1.0 / count : 0.0;
        }

        mssa->initialized = true;
        return 0;
    }

    static void mssa_opt_hankel_matvec(MSSA_Opt *mssa, const double *v, double *y)
    {
        int M = mssa->M, K = mssa->K, L = mssa->L, fft_len = mssa->fft_len, r2c_len = mssa->r2c_len;
        ssa_opt_zero(mssa->ws_real, fft_len);
        ssa_opt_reverse_copy(v, mssa->ws_real, K);
        DftiComputeForward(mssa->fft_r2c, mssa->ws_real, mssa->ws_complex);
        for (int m = 0; m < M; m++)
        {
            double *fft_xm = mssa->fft_x + m * 2 * r2c_len, *ym = y + m * L, *ws_out = mssa->ws_batch_complex;
            vzMul(r2c_len, (const MKL_Complex16 *)fft_xm, (const MKL_Complex16 *)mssa->ws_complex, (MKL_Complex16 *)ws_out);
            DftiComputeBackward(mssa->fft_c2r, ws_out, mssa->ws_batch_real);
            memcpy(ym, mssa->ws_batch_real + (K - 1), L * sizeof(double));
        }
    }

    static void mssa_opt_hankel_matvec_T(MSSA_Opt *mssa, const double *u, double *y)
    {
        int M = mssa->M, K = mssa->K, L = mssa->L, fft_len = mssa->fft_len, r2c_len = mssa->r2c_len;
        ssa_opt_zero(y, K);
        for (int m = 0; m < M; m++)
        {
            const double *um = u + m * L;
            double *fft_xm = mssa->fft_x + m * 2 * r2c_len;
            ssa_opt_zero(mssa->ws_real, fft_len);
            ssa_opt_reverse_copy(um, mssa->ws_real, L);
            DftiComputeForward(mssa->fft_r2c, mssa->ws_real, mssa->ws_complex);
            double *ws_out = mssa->ws_batch_complex;
            vzMul(r2c_len, (const MKL_Complex16 *)fft_xm, (const MKL_Complex16 *)mssa->ws_complex, (MKL_Complex16 *)ws_out);
            DftiComputeBackward(mssa->fft_c2r, ws_out, mssa->ws_batch_real);
            for (int j = 0; j < K; j++)
                y[j] += mssa->ws_batch_real[(L - 1) + j];
        }
    }

    static void mssa_opt_hankel_matvec_batch(MSSA_Opt *mssa, const double *V_block, double *Y_block, int b)
    {
        int ML = mssa->M * mssa->L;
        for (int j = 0; j < b; j++)
            mssa_opt_hankel_matvec(mssa, V_block + j * mssa->K, Y_block + j * ML);
    }
    static void mssa_opt_hankel_matvec_T_batch(MSSA_Opt *mssa, const double *U_block, double *Y_block, int b)
    {
        int ML = mssa->M * mssa->L, K = mssa->K;
        for (int j = 0; j < b; j++)
            mssa_opt_hankel_matvec_T(mssa, U_block + j * ML, Y_block + j * K);
    }

    int mssa_opt_decompose(MSSA_Opt *mssa, int k, int oversampling)
    {
        if (!mssa || !mssa->initialized || k < 1)
            return -1;
        int M = mssa->M, L = mssa->L, K = mssa->K, ML = M * L;
        int p = (oversampling <= 0) ? 8 : oversampling, kp = k + p;
        kp = ssa_opt_min(kp, ssa_opt_min(ML, K));
        k = ssa_opt_min(k, kp);
        mssa->U = (double *)ssa_opt_alloc(ML * k * sizeof(double));
        mssa->V = (double *)ssa_opt_alloc(K * k * sizeof(double));
        mssa->sigma = (double *)ssa_opt_alloc(k * sizeof(double));
        mssa->eigenvalues = (double *)ssa_opt_alloc(k * sizeof(double));
        if (!mssa->U || !mssa->V || !mssa->sigma || !mssa->eigenvalues)
            return -1;
        mssa->n_components = k;
        mssa->total_variance = 0.0;
        double *Omega = (double *)ssa_opt_alloc(K * kp * sizeof(double)), *Y = (double *)ssa_opt_alloc(ML * kp * sizeof(double));
        double *Q = (double *)ssa_opt_alloc(ML * kp * sizeof(double)), *B = (double *)ssa_opt_alloc(K * kp * sizeof(double));
        double *tau = (double *)ssa_opt_alloc(kp * sizeof(double));
        double *U_svd = (double *)ssa_opt_alloc(K * kp * sizeof(double)), *Vt_svd = (double *)ssa_opt_alloc(kp * kp * sizeof(double));
        double *S_svd = (double *)ssa_opt_alloc(kp * sizeof(double));
        double work_query;
        int *iwork = (int *)ssa_opt_alloc(8 * kp * sizeof(int));
        int lwork = -1, info;
        LAPACKE_dgesdd_work(LAPACK_COL_MAJOR, 'S', K, kp, B, K, S_svd, U_svd, K, Vt_svd, kp, &work_query, lwork, iwork);
        lwork = (int)work_query + 1;
        double *work = (double *)ssa_opt_alloc(lwork * sizeof(double));
        if (!Omega || !Y || !Q || !B || !tau || !U_svd || !Vt_svd || !S_svd || !iwork || !work)
        {
            ssa_opt_free_ptr(Omega);
            ssa_opt_free_ptr(Y);
            ssa_opt_free_ptr(Q);
            ssa_opt_free_ptr(B);
            ssa_opt_free_ptr(tau);
            ssa_opt_free_ptr(U_svd);
            ssa_opt_free_ptr(Vt_svd);
            ssa_opt_free_ptr(S_svd);
            ssa_opt_free_ptr(iwork);
            ssa_opt_free_ptr(work);
            return -1;
        }
        vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER, mssa->rng, K * kp, Omega, 0.0, 1.0);
        mssa_opt_hankel_matvec_batch(mssa, Omega, Y, kp);
        cblas_dcopy(ML * kp, Y, 1, Q, 1);
        LAPACKE_dgeqrf(LAPACK_COL_MAJOR, ML, kp, Q, ML, tau);
        LAPACKE_dorgqr(LAPACK_COL_MAJOR, ML, kp, kp, Q, ML, tau);
        mssa_opt_hankel_matvec_T_batch(mssa, Q, B, kp);
        info = LAPACKE_dgesdd_work(LAPACK_COL_MAJOR, 'S', K, kp, B, K, S_svd, U_svd, K, Vt_svd, kp, work, lwork, iwork);
        if (info != 0)
        {
            ssa_opt_free_ptr(Omega);
            ssa_opt_free_ptr(Y);
            ssa_opt_free_ptr(Q);
            ssa_opt_free_ptr(B);
            ssa_opt_free_ptr(tau);
            ssa_opt_free_ptr(U_svd);
            ssa_opt_free_ptr(Vt_svd);
            ssa_opt_free_ptr(S_svd);
            ssa_opt_free_ptr(iwork);
            ssa_opt_free_ptr(work);
            return -1;
        }
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, ML, k, kp, 1.0, Q, ML, Vt_svd, kp, 0.0, mssa->U, ML);
        for (int i = 0; i < k; i++)
            cblas_dcopy(K, &U_svd[i * K], 1, &mssa->V[i * K], 1);
        for (int i = 0; i < k; i++)
        {
            mssa->sigma[i] = S_svd[i];
            mssa->eigenvalues[i] = S_svd[i] * S_svd[i];
            mssa->total_variance += mssa->eigenvalues[i];
        }
        for (int i = 0; i < k; i++)
        {
            double sum = 0;
            for (int t = 0; t < ML; t++)
                sum += mssa->U[i * ML + t];
            if (sum < 0)
            {
                cblas_dscal(ML, -1.0, &mssa->U[i * ML], 1);
                cblas_dscal(K, -1.0, &mssa->V[i * K], 1);
            }
        }
        ssa_opt_free_ptr(Omega);
        ssa_opt_free_ptr(Y);
        ssa_opt_free_ptr(Q);
        ssa_opt_free_ptr(B);
        ssa_opt_free_ptr(tau);
        ssa_opt_free_ptr(U_svd);
        ssa_opt_free_ptr(Vt_svd);
        ssa_opt_free_ptr(S_svd);
        ssa_opt_free_ptr(iwork);
        ssa_opt_free_ptr(work);
        mssa->decomposed = true;
        return 0;
    }

    int mssa_opt_reconstruct(const MSSA_Opt *mssa, int series_idx, const int *group, int n_group, double *output)
    {
        if (!mssa || !mssa->decomposed || !group || !output || n_group < 1 || series_idx < 0 || series_idx >= mssa->M)
            return -1;
        int M = mssa->M, N = mssa->N, L = mssa->L, K = mssa->K, ML = M * L, fft_len = mssa->fft_len, r2c_len = mssa->r2c_len;
        ssa_opt_zero(output, N);
        MSSA_Opt *mssa_mut = (MSSA_Opt *)mssa;
        for (int g = 0; g < n_group; g++)
        {
            int idx = group[g];
            if (idx < 0 || idx >= mssa->n_components)
                continue;
            double sigma = mssa->sigma[idx];
            const double *u_full = &mssa->U[idx * ML], *u_m = u_full + series_idx * L, *v = &mssa->V[idx * K];
            ssa_opt_zero(mssa_mut->ws_real, fft_len);
            for (int i = 0; i < L; i++)
                mssa_mut->ws_real[i] = sigma * u_m[i];
            DftiComputeForward(mssa_mut->fft_r2c, mssa_mut->ws_real, mssa_mut->ws_complex);
            ssa_opt_zero(mssa_mut->ws_batch_real, fft_len);
            for (int i = 0; i < K; i++)
                mssa_mut->ws_batch_real[i] = v[i];
            double *ws_v = mssa_mut->ws_batch_complex;
            DftiComputeForward(mssa_mut->fft_r2c, mssa_mut->ws_batch_real, ws_v);
            vzMul(r2c_len, (const MKL_Complex16 *)mssa_mut->ws_complex, (const MKL_Complex16 *)ws_v, (MKL_Complex16 *)mssa_mut->ws_complex);
            DftiComputeBackward(mssa_mut->fft_c2r, mssa_mut->ws_complex, mssa_mut->ws_real);
            for (int t = 0; t < N; t++)
                output[t] += mssa_mut->ws_real[t];
        }
        vdMul(N, output, mssa->inv_diag_count, output);
        return 0;
    }

    int mssa_opt_reconstruct_all(const MSSA_Opt *mssa, const int *group, int n_group, double *output)
    {
        if (!mssa || !mssa->decomposed || !group || !output || n_group < 1)
            return -1;
        int M = mssa->M, N = mssa->N;
        for (int m = 0; m < M; m++)
            if (mssa_opt_reconstruct(mssa, m, group, n_group, output + m * N) != 0)
                return -1;
        return 0;
    }

    int mssa_opt_series_contributions(const MSSA_Opt *mssa, double *contributions)
    {
        if (!mssa || !mssa->decomposed || !contributions)
            return -1;
        int M = mssa->M, L = mssa->L, ML = M * L, k = mssa->n_components;
        for (int i = 0; i < k; i++)
        {
            const double *u_col = &mssa->U[i * ML];
            for (int m = 0; m < M; m++)
            {
                const double *u_block = u_col + m * L;
                double norm_sq = cblas_ddot(L, u_block, 1, u_block, 1);
                contributions[m * k + i] = norm_sq;
            }
        }
        return 0;
    }

    double mssa_opt_variance_explained(const MSSA_Opt *mssa, int start, int end)
    {
        if (!mssa || !mssa->decomposed || start < 0 || mssa->total_variance <= 0)
            return 0.0;
        if (end < 0 || end >= mssa->n_components)
            end = mssa->n_components - 1;
        double sum = 0;
        for (int i = start; i <= end; i++)
            sum += mssa->eigenvalues[i];
        return sum / mssa->total_variance;
    }

    // ========================================================================
    // ESPRIT - Estimation of Signal Parameters via Rotational Invariance
    // ========================================================================

    void ssa_opt_parestimate_free(SSA_ParEstimate *result)
    {
        if (!result)
            return;
        if (result->periods)
            ssa_opt_free_ptr(result->periods);
        if (result->frequencies)
            ssa_opt_free_ptr(result->frequencies);
        if (result->moduli)
            ssa_opt_free_ptr(result->moduli);
        if (result->rates)
            ssa_opt_free_ptr(result->rates);
        memset(result, 0, sizeof(*result));
    }

    int ssa_opt_parestimate(const SSA_Opt *ssa, const int *group, int n_group, SSA_ParEstimate *result)
    {
        if (!ssa || !ssa->decomposed || !result)
            return -1;

        // Default: use all components
        int k = (n_group > 0 && group) ? n_group : ssa->n_components;
        if (k < 1 || k > ssa->n_components)
            return -1;

        int L = ssa->L;
        if (L < 3)
            return -1; // Need at least 3 rows for ESPRIT

        // Build index array if not provided
        int *idx = NULL;
        int idx_allocated = 0;
        if (group && n_group > 0)
        {
            idx = (int *)group;
        }
        else
        {
            idx = (int *)ssa_opt_alloc(k * sizeof(int));
            if (!idx)
                return -1;
            for (int i = 0; i < k; i++)
                idx[i] = i;
            idx_allocated = 1;
        }

        // Extract selected eigenvectors: U_sel is (L x k)
        double *U_sel = (double *)ssa_opt_alloc(L * k * sizeof(double));
        if (!U_sel)
        {
            if (idx_allocated)
                ssa_opt_free_ptr(idx);
            return -1;
        }

        // Copy selected eigenvectors (column-major)
        for (int j = 0; j < k; j++)
        {
            int comp_idx = idx[j];
            if (comp_idx < 0 || comp_idx >= ssa->n_components)
            {
                ssa_opt_free_ptr(U_sel);
                if (idx_allocated)
                    ssa_opt_free_ptr(idx);
                return -1;
            }
            cblas_dcopy(L, &ssa->U[comp_idx * L], 1, &U_sel[j * L], 1);
        }

        if (idx_allocated)
            ssa_opt_free_ptr(idx);

        // ESPRIT: Use shift-invariance property
        // U_up = U[0:L-1, :], U_down = U[1:L, :]
        // Shift matrix Z = pinv(U_up) @ U_down
        // Eigenvalues of Z give signal poles

        int Lm1 = L - 1;

        // U_up: rows 0 to L-2 (Lm1 x k)
        double *U_up = (double *)ssa_opt_alloc(Lm1 * k * sizeof(double));
        // U_down: rows 1 to L-1 (Lm1 x k)
        double *U_down = (double *)ssa_opt_alloc(Lm1 * k * sizeof(double));

        if (!U_up || !U_down)
        {
            if (U_up)
                ssa_opt_free_ptr(U_up);
            if (U_down)
                ssa_opt_free_ptr(U_down);
            ssa_opt_free_ptr(U_sel);
            return -1;
        }

        // Copy with shift (column-major storage)
        for (int j = 0; j < k; j++)
        {
            cblas_dcopy(Lm1, &U_sel[j * L], 1, &U_up[j * Lm1], 1);       // rows 0:L-2
            cblas_dcopy(Lm1, &U_sel[j * L + 1], 1, &U_down[j * Lm1], 1); // rows 1:L-1
        }

        ssa_opt_free_ptr(U_sel);

        // Compute Z = pinv(U_up) @ U_down using least squares
        // Z is k x k, we solve: U_up @ Z = U_down in least squares sense
        // This is: min ||U_up @ Z - U_down||_F
        // For each column z_j of Z: solve U_up @ z_j = u_down_j

        // Use LAPACK DGELS: solves min ||A*X - B||
        // A = U_up (Lm1 x k), B = U_down (Lm1 x k), X = Z (k x k)

        double *Z = (double *)ssa_opt_alloc(k * k * sizeof(double));
        double *work_dgels = (double *)ssa_opt_alloc(Lm1 * k * sizeof(double));

        if (!Z || !work_dgels)
        {
            if (Z)
                ssa_opt_free_ptr(Z);
            if (work_dgels)
                ssa_opt_free_ptr(work_dgels);
            ssa_opt_free_ptr(U_up);
            ssa_opt_free_ptr(U_down);
            return -1;
        }

        // Copy U_up for DGELS (it gets overwritten)
        double *U_up_copy = (double *)ssa_opt_alloc(Lm1 * k * sizeof(double));
        if (!U_up_copy)
        {
            ssa_opt_free_ptr(Z);
            ssa_opt_free_ptr(work_dgels);
            ssa_opt_free_ptr(U_up);
            ssa_opt_free_ptr(U_down);
            return -1;
        }
        cblas_dcopy(Lm1 * k, U_up, 1, U_up_copy, 1);

        // Solve least squares: U_up @ Z = U_down
        // DGELS expects column-major, solves A*X=B
        // A is Lm1 x k, B is Lm1 x k, solution X is k x k (stored in first k rows of B)
        lapack_int info;
        lapack_int lwork = -1;
        double work_query;

        // Query workspace size
        info = LAPACKE_dgels_work(LAPACK_COL_MAJOR, 'N', Lm1, k, k,
                                  U_up_copy, Lm1, U_down, Lm1, &work_query, lwork);

        lwork = (lapack_int)work_query + 1;
        double *work = (double *)ssa_opt_alloc(lwork * sizeof(double));
        if (!work)
        {
            ssa_opt_free_ptr(U_up_copy);
            ssa_opt_free_ptr(Z);
            ssa_opt_free_ptr(work_dgels);
            ssa_opt_free_ptr(U_up);
            ssa_opt_free_ptr(U_down);
            return -1;
        }

        // Solve
        info = LAPACKE_dgels_work(LAPACK_COL_MAJOR, 'N', Lm1, k, k,
                                  U_up_copy, Lm1, U_down, Lm1, work, lwork);

        ssa_opt_free_ptr(work);
        ssa_opt_free_ptr(U_up_copy);
        ssa_opt_free_ptr(U_up);

        if (info != 0)
        {
            ssa_opt_free_ptr(Z);
            ssa_opt_free_ptr(work_dgels);
            ssa_opt_free_ptr(U_down);
            return -1;
        }

        // Z is now in first k rows of U_down (k x k, column-major)
        for (int j = 0; j < k; j++)
        {
            cblas_dcopy(k, &U_down[j * Lm1], 1, &Z[j * k], 1);
        }

        ssa_opt_free_ptr(U_down);
        ssa_opt_free_ptr(work_dgels);

        // Compute eigenvalues of Z (k x k)
        // Use DGEEV for general eigenvalue problem (Z may not be symmetric)
        double *wr = (double *)ssa_opt_alloc(k * sizeof(double)); // Real parts
        double *wi = (double *)ssa_opt_alloc(k * sizeof(double)); // Imaginary parts

        if (!wr || !wi)
        {
            if (wr)
                ssa_opt_free_ptr(wr);
            if (wi)
                ssa_opt_free_ptr(wi);
            ssa_opt_free_ptr(Z);
            return -1;
        }

        // DGEEV: compute eigenvalues only (no eigenvectors needed)
        info = LAPACKE_dgeev(LAPACK_COL_MAJOR, 'N', 'N', k, Z, k, wr, wi, NULL, 1, NULL, 1);

        ssa_opt_free_ptr(Z);

        if (info != 0)
        {
            ssa_opt_free_ptr(wr);
            ssa_opt_free_ptr(wi);
            return -1;
        }

        // Allocate result arrays
        result->periods = (double *)ssa_opt_alloc(k * sizeof(double));
        result->frequencies = (double *)ssa_opt_alloc(k * sizeof(double));
        result->moduli = (double *)ssa_opt_alloc(k * sizeof(double));
        result->rates = (double *)ssa_opt_alloc(k * sizeof(double));
        result->n_components = k;

        if (!result->periods || !result->frequencies || !result->moduli || !result->rates)
        {
            ssa_opt_parestimate_free(result);
            ssa_opt_free_ptr(wr);
            ssa_opt_free_ptr(wi);
            return -1;
        }

        // Convert eigenvalues to periods/frequencies
        // eigenvalue = modulus * exp(i * argument)
        // frequency = argument / (2 * pi)
        // period = 1 / |frequency| (in samples)
        // rate = log(modulus)

        const double TWO_PI = 2.0 * M_PI;

        for (int i = 0; i < k; i++)
        {
            double re = wr[i];
            double im = wi[i];

            // Modulus (damping factor)
            double mod = sqrt(re * re + im * im);
            result->moduli[i] = mod;

            // Damping rate
            result->rates[i] = (mod > 1e-12) ? log(mod) : -30.0; // -30 ~ effectively zero

            // Argument (phase angle)
            double arg = atan2(im, re);

            // Frequency (cycles per sample)
            double freq = arg / TWO_PI;
            result->frequencies[i] = freq;

            // Period (samples per cycle)
            if (fabs(freq) > 1e-12)
            {
                result->periods[i] = 1.0 / fabs(freq);
            }
            else
            {
                result->periods[i] = INFINITY; // DC component (trend)
            }
        }

        ssa_opt_free_ptr(wr);
        ssa_opt_free_ptr(wi);

        return 0;
    }

#endif // SSA_OPT_IMPLEMENTATION

#ifdef __cplusplus
}
#endif

#endif // SSA_OPT_R2C_H