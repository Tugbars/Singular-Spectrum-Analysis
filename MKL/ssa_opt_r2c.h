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

// ============================================================================
// PRECISION CONFIGURATION
// ============================================================================
// Compile with -DSSA_USE_FLOAT for single precision (faster, less accurate)
// Default is double precision
//
// Float benefits:
//   - 2x less memory bandwidth (critical for memory-bound workloads)
//   - 2x more SIMD throughput (8 floats vs 4 doubles per AVX register)
//   - ~30-50% overall speedup on memory-bound systems
//
// Float precision (~7 digits) is sufficient for:
//   - SSA eigenvalue gaps (typically 1e-1 to 1e-3)
//   - Randomized SVD (already approximate)
//   - Trading signal extraction (noise floor >> float epsilon)
// ============================================================================

#ifdef SSA_USE_FLOAT
    typedef float ssa_real;
    #define SSA_DFTI_PRECISION      DFTI_SINGLE
    #define SSA_MKL_COMPLEX         MKL_Complex8
    #define SSA_CONVERGENCE_TOL_DEFAULT 1e-6f
    
    // BLAS
    #define ssa_cblas_dot           cblas_sdot
    #define ssa_cblas_nrm2          cblas_snrm2
    #define ssa_cblas_scal          cblas_sscal
    #define ssa_cblas_axpy          cblas_saxpy
    #define ssa_cblas_copy          cblas_scopy
    #define ssa_cblas_swap          cblas_sswap
    #define ssa_cblas_gemv          cblas_sgemv
    #define ssa_cblas_gemm          cblas_sgemm
    
    // LAPACK
    #define ssa_LAPACKE_geqrf       LAPACKE_sgeqrf
    #define ssa_LAPACKE_orgqr       LAPACKE_sorgqr
    #define ssa_LAPACKE_gesdd_work  LAPACKE_sgesdd_work
    #define ssa_LAPACKE_gesvd       LAPACKE_sgesvd
    
    // VML (complex operations)
    #define ssa_vMul                vcMul
    #define ssa_vMulByConj          vcMulByConj
    #define ssa_vMul_real           vsMul
    
    // VSL (random number generation)
    #define ssa_vRngGaussian        vsRngGaussian
    #define ssa_vRngUniform         vsRngUniform
    #define SSA_RNG_GAUSSIAN_METHOD VSL_RNG_METHOD_GAUSSIAN_BOXMULLER
    #define SSA_RNG_UNIFORM_METHOD  VSL_RNG_METHOD_UNIFORM_STD
    
    // Math functions
    #define ssa_sqrt                sqrtf
    #define ssa_fabs                fabsf
    
#else
    typedef double ssa_real;
    #define SSA_DFTI_PRECISION      DFTI_DOUBLE
    #define SSA_MKL_COMPLEX         MKL_Complex16
    #define SSA_CONVERGENCE_TOL_DEFAULT 1e-12
    
    // BLAS
    #define ssa_cblas_dot           cblas_ddot
    #define ssa_cblas_nrm2          cblas_dnrm2
    #define ssa_cblas_scal          cblas_dscal
    #define ssa_cblas_axpy          cblas_daxpy
    #define ssa_cblas_copy          cblas_dcopy
    #define ssa_cblas_swap          cblas_dswap
    #define ssa_cblas_gemv          cblas_dgemv
    #define ssa_cblas_gemm          cblas_dgemm
    
    // LAPACK
    #define ssa_LAPACKE_geqrf       LAPACKE_dgeqrf
    #define ssa_LAPACKE_orgqr       LAPACKE_dorgqr
    #define ssa_LAPACKE_gesdd_work  LAPACKE_dgesdd_work
    #define ssa_LAPACKE_gesvd       LAPACKE_dgesvd
    
    // VML (complex operations)
    #define ssa_vMul                vzMul
    #define ssa_vMulByConj          vzMulByConj
    #define ssa_vMul_real           vdMul
    
    // VSL (random number generation)
    #define ssa_vRngGaussian        vdRngGaussian
    #define ssa_vRngUniform         vdRngUniform
    #define SSA_RNG_GAUSSIAN_METHOD VSL_RNG_METHOD_GAUSSIAN_BOXMULLER
    #define SSA_RNG_UNIFORM_METHOD  VSL_RNG_METHOD_UNIFORM_STD
    
    // Math functions
    #define ssa_sqrt                sqrt
    #define ssa_fabs                fabs
    
#endif

// ============================================================================
// CONFIGURATION CONSTANTS
// ============================================================================

#ifndef SSA_CONVERGENCE_TOL
#define SSA_CONVERGENCE_TOL SSA_CONVERGENCE_TOL_DEFAULT
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
        int batch_size; // Dynamic batch size for FFT (cache-optimized at init)

        // === MKL FFT Descriptors ===
        DFTI_DESCRIPTOR_HANDLE fft_r2c;       // Forward: real → complex (single vector)
        DFTI_DESCRIPTOR_HANDLE fft_c2r;       // Inverse: complex → real (single vector)
        DFTI_DESCRIPTOR_HANDLE fft_r2c_batch; // Forward batched: processes batch_size vectors
        DFTI_DESCRIPTOR_HANDLE fft_c2r_batch; // Inverse batched: processes batch_size vectors
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
        ssa_real *fft_x;      // Pre-computed FFT of signal x, length 2*r2c_len (complex)
        ssa_real *ws_real;    // Real workspace for FFT input, length fft_len
        ssa_real *ws_complex; // Complex workspace for FFT output, length 2*r2c_len
        ssa_real *ws_real2;   // Secondary real workspace for adjoint matvec
        ssa_real *ws_proj;    // Projection workspace for power iteration

        // === Batched FFT Workspaces ===
        ssa_real *ws_batch_real;    // Batched real input, fft_len * batch_size
        ssa_real *ws_batch_complex; // Batched complex output, 2*r2c_len * batch_size

        // === SVD Results ===
        ssa_real *U;           // Left singular vectors, L × n_components (column-major)
        ssa_real *V;           // Right singular vectors, K × n_components (column-major)
        ssa_real *sigma;       // Singular values σ₀ ≥ σ₁ ≥ ... ≥ σₖ₋₁
        ssa_real *eigenvalues; // Eigenvalues λᵢ = σᵢ² (of HᵀH)
        int n_components;      // Number of computed components

        // === Reconstruction Cache (frequency-domain accumulation) ===
        ssa_real *inv_diag_count; // 1/w[t] where w[t] = diagonal averaging weight, length N
        ssa_real *U_fft;          // Cached FFT of scaled U columns: FFT(σᵢ·Uᵢ), n_components × 2*r2c_len
        ssa_real *V_fft;          // Cached FFT of V columns: FFT(Vᵢ), n_components × 2*r2c_len
        bool fft_cached;          // True if U_fft/V_fft are valid

        // === W-Correlation Workspace (DSYRK optimization) ===
        ssa_real *wcorr_ws_complex; // Complex workspace for batched component FFTs
        ssa_real *wcorr_h;          // Reconstructed components before weighting, n × N
        ssa_real *wcorr_G;          // Weighted/normalized matrix for DSYRK, n × N
        ssa_real *wcorr_sqrt_inv_c; // Precomputed √(1/w[t]) for fast weighting

        // === Randomized SVD Workspace (malloc-free hot path) ===
        int prepared_kp;          // Max k+p this workspace supports (set by ssa_opt_prepare)
        int prepared_lwork;       // LAPACK workspace size for DGESDD
        ssa_real *decomp_Omega;     // Random Gaussian matrix, K × (k+p)
        ssa_real *decomp_Y;         // Random projection Y = H·Ω, L × (k+p)
        ssa_real *decomp_Q;         // Orthonormal basis from QR(Y), L × (k+p)
        ssa_real *decomp_B;         // Projected matrix B = Hᵀ·Q, K × (k+p)
        ssa_real *decomp_tau;       // Householder reflectors from QR, length k+p
        ssa_real *decomp_B_left;    // Left singular vectors of B (DGESDD output)
        ssa_real *decomp_B_right_T; // Right singular vectors of B transposed
        ssa_real *decomp_S;         // Singular values from small SVD, length k+p
        ssa_real *decomp_work;      // LAPACK DGESDD workspace, length prepared_lwork
        int *decomp_iwork;          // LAPACK DGESDD integer workspace, length 8*(k+p)
        bool prepared;              // True if randomized workspace is allocated

        // === State Flags ===
        bool initialized; // True after ssa_opt_init() succeeds
        bool decomposed;  // True after any decompose function succeeds

        // === Statistics ===
        ssa_real total_variance; // Sum of all eigenvalues (trace of HᵀH)

    } SSA_Opt;

    // === Component Statistics (for automatic grouping / rank selection) ===
    typedef struct
    {
        int n; // Number of components analyzed

        // === Singular Value Analysis ===
        ssa_real *singular_values; // σᵢ values, length n (descending order)
        ssa_real *log_sv;          // log(σᵢ), useful for scree plot visualization
        ssa_real *gaps;            // Relative gaps: (σᵢ - σᵢ₊₁) / σᵢ, length n-1
                                   // Large gap suggests signal/noise boundary

        // === Variance Analysis ===
        ssa_real *cumulative_var; // Cumulative variance explained: Σⱼ₌₀ⁱ λⱼ / Σλ
                                  // cumulative_var[i] = fraction of total variance in components 0..i

        // === Automatic Rank Selection ===
        ssa_real *second_diff;    // Second difference of log(σ): Δ²log(σᵢ), length n-2
                                  // Peak indicates "elbow" in scree plot
        int suggested_signal;     // Suggested number of signal components (auto-detected)
        ssa_real gap_threshold;   // Threshold used for gap detection (default 0.1 = 10% drop)

    } SSA_ComponentStats;

    // === Linear Recurrence Formula (for R-forecasting) ===
    typedef struct
    {
        ssa_real *R; // LRF coefficients [a₁, a₂, ..., aₗ₋₁], length L-1
                     // Forecasts via: x[n] = Σⱼ aⱼ · x[n-j]
                     // Derived from last row of eigenvectors (shifted structure)

        int L; // Window length (determines LRF order = L-1)

        ssa_real verticality; // ν² = Σᵢ πᵢ² where πᵢ = U[L-1, i] (last row of U)
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
        ssa_real *fft_x;            // Pre-computed FFT of all M signals, M × 2*r2c_len
                                    // fft_x[m * 2*r2c_len ...] = FFT(series m)
        ssa_real *ws_real;          // Real workspace, length fft_len
        ssa_real *ws_complex;       // Complex workspace, length 2*r2c_len
        ssa_real *ws_batch_real;    // Batched real workspace
        ssa_real *ws_batch_complex; // Batched complex workspace

        // === SVD Results ===
        // Block trajectory matrix is (M·L) × K, stacking L-lagged windows from all series
        ssa_real *U;           // Left singular vectors, (M·L) × n_components
                               // U is block-structured: U[m*L : (m+1)*L, :] for series m
        ssa_real *V;           // Right singular vectors, K × n_components (shared across series)
        ssa_real *sigma;       // Singular values, length n_components
        ssa_real *eigenvalues; // Eigenvalues λᵢ = σᵢ²
        int n_components;      // Number of computed components

        // === Reconstruction ===
        ssa_real *inv_diag_count; // Diagonal averaging weights (same for all series)

        // === State ===
        bool initialized;      // True after mssa_opt_init() succeeds
        bool decomposed;       // True after mssa_opt_decompose() succeeds
        ssa_real total_variance; // Total variance across all M series

    } MSSA_Opt;

    // Public API declarations
    int ssa_opt_init(SSA_Opt *ssa, const ssa_real *x, int N, int L);
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
    int ssa_opt_update_signal(SSA_Opt *ssa, const ssa_real *new_x);

    int ssa_opt_extend(SSA_Opt *ssa, int additional_k, int max_iter);
    int ssa_opt_reconstruct(const SSA_Opt *ssa, const int *group, int n_group, ssa_real *output);
    int ssa_opt_cache_ffts(SSA_Opt *ssa);
    void ssa_opt_free_cached_ffts(SSA_Opt *ssa);
    int ssa_opt_wcorr_matrix(const SSA_Opt *ssa, ssa_real *W);
    int ssa_opt_wcorr_matrix_fast(const SSA_Opt *ssa, ssa_real *W);
    ssa_real ssa_opt_wcorr_pair(const SSA_Opt *ssa, int i, int j);
    int ssa_opt_component_stats(const SSA_Opt *ssa, SSA_ComponentStats *stats);
    void ssa_opt_component_stats_free(SSA_ComponentStats *stats);
    int ssa_opt_find_periodic_pairs(const SSA_Opt *ssa, int *pairs, int max_pairs, ssa_real sv_tol, ssa_real wcorr_thresh);
    int ssa_opt_compute_lrf(const SSA_Opt *ssa, const int *group, int n_group, SSA_LRF *lrf);
    void ssa_opt_lrf_free(SSA_LRF *lrf);
    int ssa_opt_forecast(const SSA_Opt *ssa, const int *group, int n_group, int n_forecast, ssa_real *output);
    int ssa_opt_forecast_full(const SSA_Opt *ssa, const int *group, int n_group, int n_forecast, ssa_real *output);
    int ssa_opt_forecast_with_lrf(const SSA_LRF *lrf, const ssa_real *base_signal, int base_len, int n_forecast, ssa_real *output);

    // Vector forecast (V-forecast) - alternative to recurrent forecast
    // Projects onto eigenvector subspace at each step instead of using LRR
    int ssa_opt_vforecast(const SSA_Opt *ssa, const int *group, int n_group, int n_forecast, ssa_real *output);
    int ssa_opt_vforecast_full(const SSA_Opt *ssa, const int *group, int n_group, int n_forecast, ssa_real *output);
    int ssa_opt_vforecast_fast(const SSA_Opt *ssa, const int *group, int n_group,
                               const ssa_real *base_signal, int base_len, int n_forecast, ssa_real *output);

    int mssa_opt_init(MSSA_Opt *mssa, const ssa_real *X, int M, int N, int L);
    int mssa_opt_decompose(MSSA_Opt *mssa, int k, int oversampling);
    int mssa_opt_reconstruct(const MSSA_Opt *mssa, int series_idx, const int *group, int n_group, ssa_real *output);
    int mssa_opt_reconstruct_all(const MSSA_Opt *mssa, const int *group, int n_group, ssa_real *output);
    int mssa_opt_series_contributions(const MSSA_Opt *mssa, ssa_real *contributions);
    ssa_real mssa_opt_variance_explained(const MSSA_Opt *mssa, int start, int end);
    void mssa_opt_free(MSSA_Opt *mssa);
    int ssa_opt_get_trend(const SSA_Opt *ssa, ssa_real *output);
    int ssa_opt_get_noise(const SSA_Opt *ssa, int noise_start, ssa_real *output);
    ssa_real ssa_opt_variance_explained(const SSA_Opt *ssa, int start, int end);

    // Getters for decomposition results
    int ssa_opt_get_singular_values(const SSA_Opt *ssa, ssa_real *output, int max_n);
    int ssa_opt_get_eigenvalues(const SSA_Opt *ssa, ssa_real *output, int max_n);
    ssa_real ssa_opt_get_total_variance(const SSA_Opt *ssa);

    // Cadzow iterations - iterative finite-rank signal approximation
    typedef struct
    {
        int iterations;       // Iterations performed
        ssa_real final_diff;  // Final relative difference
        ssa_real converged;   // 1.0 if converged, 0.0 if hit max_iter
    } SSA_CadzowResult;

    int ssa_opt_cadzow(const ssa_real *x, int N, int L, int rank, int max_iter, ssa_real tol,
                       ssa_real *output, SSA_CadzowResult *result);
    int ssa_opt_cadzow_weighted(const ssa_real *x, int N, int L, int rank, int max_iter,
                                ssa_real tol, ssa_real alpha, ssa_real *output, SSA_CadzowResult *result);
    int ssa_opt_cadzow_inplace(SSA_Opt *ssa, int rank, int max_iter, ssa_real tol, SSA_CadzowResult *result);

    // ESPRIT - Estimation of Signal Parameters via Rotational Invariance
    // Extracts frequencies/periods from eigenvectors
    typedef struct
    {
        ssa_real *periods;     // Estimated periods (in samples), length = n_components
        ssa_real *frequencies; // Frequencies (cycles per sample), length = n_components
        ssa_real *moduli;      // |eigenvalue| - damping factor (1.0 = undamped sinusoid)
        ssa_real *rates;       // log(|eigenvalue|) - damping rate (0 = undamped)
        int n_components;      // Number of components analyzed
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
        int iterations;       // Iterations performed
        ssa_real final_diff;  // Final relative change in gap values
        int converged;        // 1 if converged, 0 if hit max_iter
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
    int ssa_opt_gapfill(ssa_real *x, int N, int L, int rank, int max_iter, ssa_real tol,
                        SSA_GapFillResult *result);

    // Simple gap filling using forecast/backcast
    // Fills each gap by averaging forward forecast and backward forecast
    // Faster but less accurate than iterative method
    int ssa_opt_gapfill_simple(ssa_real *x, int N, int L, int rank, SSA_GapFillResult *result);

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
    static inline ssa_real ssa_opt_dot(const ssa_real *a, const ssa_real *b, int n) { return ssa_cblas_dot(n, a, 1, b, 1); }
    static inline ssa_real ssa_opt_nrm2(const ssa_real *v, int n) { return ssa_cblas_nrm2(n, v, 1); }
    static inline void ssa_opt_scal(ssa_real *v, int n, ssa_real s) { ssa_cblas_scal(n, s, v, 1); }
    static inline void ssa_opt_axpy(ssa_real *y, const ssa_real *x, ssa_real a, int n) { ssa_cblas_axpy(n, a, x, 1, y, 1); }
    static inline void ssa_opt_copy(const ssa_real *src, ssa_real *dst, int n) { ssa_cblas_copy(n, src, 1, dst, 1); }
    static inline ssa_real ssa_opt_normalize(ssa_real *v, int n)
    {
        ssa_real norm = ssa_cblas_nrm2(n, v, 1);
        if (norm > (ssa_real)SSA_CONVERGENCE_TOL)
            ssa_cblas_scal(n, (ssa_real)1.0 / norm, v, 1);
        return norm;
    }
    static inline void ssa_opt_zero(ssa_real *v, int n) { memset(v, 0, n * sizeof(ssa_real)); }
    static inline void ssa_opt_complex_mul_r2c(const ssa_real *a, const ssa_real *b, ssa_real *c, int r2c_len) { ssa_vMul(r2c_len, (const SSA_MKL_COMPLEX *)a, (const SSA_MKL_COMPLEX *)b, (SSA_MKL_COMPLEX *)c); }
    
    // Conjugate multiply: C = A · conj(B)
    // Used for correlation via FFT: corr(x,v) = IFFT(FFT(x) · conj(FFT(v)))
    // This eliminates the need for reverse_copy before FFT
    static inline void ssa_opt_complex_mul_conj_r2c(const ssa_real *a, const ssa_real *b, ssa_real *c, int r2c_len) { ssa_vMulByConj(r2c_len, (const SSA_MKL_COMPLEX *)a, (const SSA_MKL_COMPLEX *)b, (SSA_MKL_COMPLEX *)c); }

    // ========================================================================
    // SIMD-optimized helper functions (AVX2)
    // ========================================================================

    // AVX2 reverse copy: dst[i] = src[n-1-i]
    static inline void ssa_opt_reverse_copy(const ssa_real *src, ssa_real *dst, int n)
    {
        int i = 0;
#ifdef SSA_USE_FLOAT
        // Float version: process 8 floats at a time with AVX
#if defined(__AVX2__) || defined(__AVX__)
        const int simd_width = 8;
        for (; i + simd_width <= n; i += simd_width)
        {
            __m256 v = _mm256_loadu_ps(&src[n - i - simd_width]);
            v = _mm256_permute2f128_ps(v, v, 0x01);
            v = _mm256_shuffle_ps(v, v, _MM_SHUFFLE(0, 1, 2, 3));
            _mm256_storeu_ps(&dst[i], v);
        }
#endif
        for (; i < n; i++)
            dst[i] = src[n - 1 - i];
#else
        // Double version: process 4 doubles at a time with AVX
#if defined(__AVX2__) || defined(__AVX__)
        const int simd_width = 4;
        for (; i + simd_width <= n; i += simd_width)
        {
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
#endif
    }

    // AVX2 weighted squared norm: sum(h[t]^2 * w[t])
    static inline ssa_real ssa_opt_weighted_norm_sq(const ssa_real *h, const ssa_real *w, int n)
    {
        ssa_real result = 0;
        int i = 0;

#ifdef SSA_USE_FLOAT
#if defined(__AVX2__) || defined(__AVX__)
        __m256 sum_vec = _mm256_setzero_ps();
        for (; i + 8 <= n; i += 8)
        {
            __m256 h_vec = _mm256_loadu_ps(&h[i]);
            __m256 w_vec = _mm256_loadu_ps(&w[i]);
            __m256 h_sq = _mm256_mul_ps(h_vec, h_vec);
            __m256 prod = _mm256_mul_ps(h_sq, w_vec);
            sum_vec = _mm256_add_ps(sum_vec, prod);
        }
        // Horizontal sum for float
        __m128 low = _mm256_castps256_ps128(sum_vec);
        __m128 high = _mm256_extractf128_ps(sum_vec, 1);
        __m128 sum128 = _mm_add_ps(low, high);
        sum128 = _mm_hadd_ps(sum128, sum128);
        sum128 = _mm_hadd_ps(sum128, sum128);
        result = _mm_cvtss_f32(sum128);
#endif
#else
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
#endif

        // Scalar cleanup
        for (; i < n; i++)
        {
            result += h[i] * h[i] * w[i];
        }

        return result;
    }

    // AVX2 weighted inner product: sum(a[t] * b[t] * w[t])
    static inline ssa_real ssa_opt_weighted_inner(const ssa_real *a, const ssa_real *b, const ssa_real *w, int n)
    {
        ssa_real result = 0;
        int i = 0;

#ifdef SSA_USE_FLOAT
#if defined(__AVX2__) || defined(__AVX__)
        __m256 sum_vec = _mm256_setzero_ps();
        for (; i + 8 <= n; i += 8)
        {
            __m256 a_vec = _mm256_loadu_ps(&a[i]);
            __m256 b_vec = _mm256_loadu_ps(&b[i]);
            __m256 w_vec = _mm256_loadu_ps(&w[i]);
            __m256 ab = _mm256_mul_ps(a_vec, b_vec);
            __m256 prod = _mm256_mul_ps(ab, w_vec);
            sum_vec = _mm256_add_ps(sum_vec, prod);
        }
        __m128 low = _mm256_castps256_ps128(sum_vec);
        __m128 high = _mm256_extractf128_ps(sum_vec, 1);
        __m128 sum128 = _mm_add_ps(low, high);
        sum128 = _mm_hadd_ps(sum128, sum128);
        sum128 = _mm_hadd_ps(sum128, sum128);
        result = _mm_cvtss_f32(sum128);
#endif
#else
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
    static inline void ssa_opt_weighted_inner3(const ssa_real *a, const ssa_real *b, const ssa_real *w, int n,
                                               ssa_real *inner, ssa_real *norm_a_sq, ssa_real *norm_b_sq)
    {
        ssa_real r_inner = 0, r_norm_a = 0, r_norm_b = 0;
        int i = 0;

#ifdef SSA_USE_FLOAT
#if defined(__AVX2__) || defined(__AVX__)
        __m256 sum_inner = _mm256_setzero_ps();
        __m256 sum_norm_a = _mm256_setzero_ps();
        __m256 sum_norm_b = _mm256_setzero_ps();

        for (; i + 8 <= n; i += 8)
        {
            __m256 a_vec = _mm256_loadu_ps(&a[i]);
            __m256 b_vec = _mm256_loadu_ps(&b[i]);
            __m256 w_vec = _mm256_loadu_ps(&w[i]);

            __m256 ab = _mm256_mul_ps(a_vec, b_vec);
            sum_inner = _mm256_add_ps(sum_inner, _mm256_mul_ps(ab, w_vec));

            __m256 aa = _mm256_mul_ps(a_vec, a_vec);
            sum_norm_a = _mm256_add_ps(sum_norm_a, _mm256_mul_ps(aa, w_vec));

            __m256 bb = _mm256_mul_ps(b_vec, b_vec);
            sum_norm_b = _mm256_add_ps(sum_norm_b, _mm256_mul_ps(bb, w_vec));
        }

        // Horizontal sums for float
        {
            __m128 low = _mm256_castps256_ps128(sum_inner);
            __m128 high = _mm256_extractf128_ps(sum_inner, 1);
            __m128 sum128 = _mm_add_ps(low, high);
            sum128 = _mm_hadd_ps(sum128, sum128);
            sum128 = _mm_hadd_ps(sum128, sum128);
            r_inner = _mm_cvtss_f32(sum128);
        }
        {
            __m128 low = _mm256_castps256_ps128(sum_norm_a);
            __m128 high = _mm256_extractf128_ps(sum_norm_a, 1);
            __m128 sum128 = _mm_add_ps(low, high);
            sum128 = _mm_hadd_ps(sum128, sum128);
            sum128 = _mm_hadd_ps(sum128, sum128);
            r_norm_a = _mm_cvtss_f32(sum128);
        }
        {
            __m128 low = _mm256_castps256_ps128(sum_norm_b);
            __m128 high = _mm256_extractf128_ps(sum_norm_b, 1);
            __m128 sum128 = _mm_add_ps(low, high);
            sum128 = _mm_hadd_ps(sum128, sum128);
            sum128 = _mm_hadd_ps(sum128, sum128);
            r_norm_b = _mm_cvtss_f32(sum128);
        }
#endif
#else
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
#endif

        // Scalar cleanup
        for (; i < n; i++)
        {
            ssa_real wi = w[i];
            r_inner += a[i] * b[i] * wi;
            r_norm_a += a[i] * a[i] * wi;
            r_norm_b += b[i] * b[i] * wi;
        }

        *inner = r_inner;
        *norm_a_sq = r_norm_a;
        *norm_b_sq = r_norm_b;
    }

    // AVX2 scale and weight: dst[t] = scale * src[t] * w[t]
    static inline void ssa_opt_scale_weighted(const ssa_real *src, const ssa_real *w, ssa_real scale, ssa_real *dst, int n)
    {
        int i = 0;

#ifdef SSA_USE_FLOAT
#if defined(__AVX2__) || defined(__AVX__)
        __m256 scale_vec = _mm256_set1_ps(scale);

        for (; i + 8 <= n; i += 8)
        {
            __m256 s_vec = _mm256_loadu_ps(&src[i]);
            __m256 w_vec = _mm256_loadu_ps(&w[i]);
            __m256 prod = _mm256_mul_ps(_mm256_mul_ps(s_vec, w_vec), scale_vec);
            _mm256_storeu_ps(&dst[i], prod);
        }
#endif
#else
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
#endif

        // Scalar cleanup
        for (; i < n; i++)
        {
            dst[i] = scale * src[i] * w[i];
        }
    }

    /**
     * SIMD Complex multiply-accumulate: acc += u * v (fused, no intermediate storage)
     * Replaces: ssa_opt_complex_mul_r2c() + ssa_cblas_axpy() with single fused operation
     */
    static inline void ssa_opt_complex_mul_acc(
        const ssa_real *__restrict u_fft,
        const ssa_real *__restrict v_fft,
        ssa_real *__restrict acc,
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
            ssa_real u_re = u_fft[2 * i];
            ssa_real u_im = u_fft[2 * i + 1];
            ssa_real v_re = v_fft[2 * i];
            ssa_real v_im = v_fft[2 * i + 1];
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
            ssa_real u_re = u_fft[2 * i];
            ssa_real u_im = u_fft[2 * i + 1];
            ssa_real v_re = v_fft[2 * i];
            ssa_real v_im = v_fft[2 * i + 1];
            acc[2 * i] += u_re * v_re - u_im * v_im;
            acc[2 * i + 1] += u_re * v_im + u_im * v_re;
        }
#endif
    }

    // ========================================================================
    // End SIMD helpers
    // ========================================================================

    // Hankel matvec via R2C FFT: y = H·v where H[i,j] = x[i+j]
    // 
    // Mathematical identity:
    //   y[i] = Σⱼ x[i+j]·v[j] = correlation(x, v)[i]
    //   correlation(x, v) = IFFT(FFT(x) · conj(FFT(v)))
    //
    // OPTIMIZATION: Conjugate multiply eliminates reverse_copy entirely
    // Old: flip(v) → FFT → multiply → IFFT → extract [K-1 : K-1+L]
    // New: v → FFT → conj_multiply → IFFT → extract [0 : L]
    static void ssa_opt_hankel_matvec(SSA_Opt *ssa, const ssa_real *v, ssa_real *y)
    {
        int K = ssa->K, L = ssa->L, fft_len = ssa->fft_len, r2c_len = ssa->r2c_len;
        
        // Forward copy (no reverse needed with conjugate multiply)
        memcpy(ssa->ws_real, v, K * sizeof(ssa_real));
        memset(ssa->ws_real + K, 0, (fft_len - K) * sizeof(ssa_real));
        
        DftiComputeForward(ssa->fft_r2c, ssa->ws_real, ssa->ws_complex);                     // FFT(v)
        ssa_opt_complex_mul_conj_r2c(ssa->fft_x, ssa->ws_complex, ssa->ws_complex, r2c_len); // FFT(x) · conj(FFT(v))
        DftiComputeBackward(ssa->fft_c2r, ssa->ws_complex, ssa->ws_real);                    // IFFT → correlation
        memcpy(y, ssa->ws_real, L * sizeof(ssa_real));                                         // extract [0 : L]
    }

    // Adjoint Hankel matvec: z = Hᵀ·u
    //
    // Mathematical identity:
    //   z[j] = Σᵢ x[i+j]·u[i] = correlation(x, u)[j]
    //
    // OPTIMIZATION: Conjugate multiply eliminates reverse_copy entirely
    static void ssa_opt_hankel_matvec_T(SSA_Opt *ssa, const ssa_real *u, ssa_real *y)
    {
        int K = ssa->K, L = ssa->L, fft_len = ssa->fft_len, r2c_len = ssa->r2c_len;
        
        // Forward copy (no reverse needed with conjugate multiply)
        memcpy(ssa->ws_real, u, L * sizeof(ssa_real));
        memset(ssa->ws_real + L, 0, (fft_len - L) * sizeof(ssa_real));
        
        DftiComputeForward(ssa->fft_r2c, ssa->ws_real, ssa->ws_complex);
        ssa_opt_complex_mul_conj_r2c(ssa->fft_x, ssa->ws_complex, ssa->ws_complex, r2c_len);
        DftiComputeBackward(ssa->fft_c2r, ssa->ws_complex, ssa->ws_real);
        memcpy(y, ssa->ws_real, K * sizeof(ssa_real));                                         // extract [0 : K]
    }

    // Block Hankel matvec: Y = H·V where V is K×b, Y is L×b
    // Batches vectors per MKL FFT call for efficiency (batch_size computed at init)
    //
    // OPTIMIZATION: Conjugate multiply eliminates reverse copy in batch packing
    // OPTIMIZATION: OpenMP parallel packing when sufficient work to hide overhead
    static void ssa_opt_hankel_matvec_block(SSA_Opt *ssa, const ssa_real *V_block, ssa_real *Y_block, int b)
    {
        int K = ssa->K, L = ssa->L, fft_len = ssa->fft_len, r2c_len = ssa->r2c_len;
        int batch_size = ssa->batch_size;
        ssa_real *ws_real = ssa->ws_batch_real, *ws_complex = ssa->ws_batch_complex;
        int col = 0;
        while (col < b)
        {
            int batch_count = ssa_opt_min(batch_size, b - col);
            if (batch_count < 2)
            { // Fallback for tiny batches
                for (int i = 0; i < batch_count; i++)
                    ssa_opt_hankel_matvec(ssa, &V_block[(col + i) * K], &Y_block[(col + i) * L]);
                col += batch_count;
                continue;
            }
            // Pack vectors contiguously for batched FFT
            // Parallelize only when enough work to hide OpenMP overhead
            {
                int i;
#ifdef _OPENMP
                #pragma omp parallel for schedule(static) if(batch_count >= 8 && K > 2048)
#endif
                for (i = 0; i < batch_count; i++)
                {
                    const ssa_real *v = &V_block[(col + i) * K];
                    ssa_real *dst = ws_real + i * fft_len;
                    // Forward copy into [0..K)
                    memcpy(dst, v, K * sizeof(ssa_real));
                    // Zero only the tail [K..fft_len)
                    memset(dst + K, 0, (fft_len - K) * sizeof(ssa_real));
                }
            }
            DftiComputeForward(ssa->fft_r2c_batch, ws_real, ws_complex); // Batched FFT
            for (int i = 0; i < batch_count; i++)
            {
                ssa_real *fft_v = ws_complex + i * 2 * r2c_len;
                ssa_opt_complex_mul_conj_r2c(ssa->fft_x, fft_v, fft_v, r2c_len); // Conjugate multiply
            }
            DftiComputeBackward(ssa->fft_c2r_batch, ws_complex, ws_real); // Batched IFFT
            for (int i = 0; i < batch_count; i++)
                memcpy(&Y_block[(col + i) * L], ws_real + i * fft_len, L * sizeof(ssa_real)); // Extract [0 : L]
            col += batch_count;
        }
    }

    // Block adjoint Hankel matvec: Z = Hᵀ·U where U is L×b, Z is K×b
    //
    // OPTIMIZATION: Conjugate multiply eliminates reverse copy in batch packing
    // OPTIMIZATION: OpenMP parallel packing when sufficient work to hide overhead
    static void ssa_opt_hankel_matvec_T_block(SSA_Opt *ssa, const ssa_real *U_block, ssa_real *Y_block, int b)
    {
        int K = ssa->K, L = ssa->L, fft_len = ssa->fft_len, r2c_len = ssa->r2c_len;
        int batch_size = ssa->batch_size;
        ssa_real *ws_real = ssa->ws_batch_real, *ws_complex = ssa->ws_batch_complex;
        int col = 0;
        while (col < b)
        {
            int batch_count = ssa_opt_min(batch_size, b - col);
            if (batch_count < 2)
            {
                for (int i = 0; i < batch_count; i++)
                    ssa_opt_hankel_matvec_T(ssa, &U_block[(col + i) * L], &Y_block[(col + i) * K]);
                col += batch_count;
                continue;
            }
            // Pack vectors contiguously for batched FFT
            // Parallelize only when enough work to hide OpenMP overhead
            {
                int i;
#ifdef _OPENMP
                #pragma omp parallel for schedule(static) if(batch_count >= 8 && L > 2048)
#endif
                for (i = 0; i < batch_count; i++)
                {
                    const ssa_real *u = &U_block[(col + i) * L];
                    ssa_real *dst = ws_real + i * fft_len;
                    // Forward copy into [0..L)
                    memcpy(dst, u, L * sizeof(ssa_real));
                    // Zero only the tail [L..fft_len)
                    memset(dst + L, 0, (fft_len - L) * sizeof(ssa_real));
                }
            }
            DftiComputeForward(ssa->fft_r2c_batch, ws_real, ws_complex);
            for (int i = 0; i < batch_count; i++)
            {
                ssa_real *fft_u = ws_complex + i * 2 * r2c_len;
                ssa_opt_complex_mul_conj_r2c(ssa->fft_x, fft_u, fft_u, r2c_len); // Conjugate multiply
            }
            DftiComputeBackward(ssa->fft_c2r_batch, ws_complex, ws_real);
            for (int i = 0; i < batch_count; i++)
                memcpy(&Y_block[(col + i) * K], ws_real + i * fft_len, K * sizeof(ssa_real)); // Extract [0 : K]
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
    int ssa_opt_init(SSA_Opt *ssa, const ssa_real *x, int N, int L)
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
        ssa->ws_real = (ssa_real *)ssa_opt_alloc(fft_n * sizeof(ssa_real));

        // ws_complex: Complex output from forward FFT / input to inverse FFT
        // Stored as interleaved [re, im, re, im, ...], length 2 * r2c_len doubles
        ssa->ws_complex = (ssa_real *)ssa_opt_alloc(2 * ssa->r2c_len * sizeof(ssa_real));

        // ws_real2: Secondary real workspace for adjoint matvec (Hᵀ·u)
        // Needed because forward and adjoint use different extraction offsets
        ssa->ws_real2 = (ssa_real *)ssa_opt_alloc(fft_n * sizeof(ssa_real));

        // ws_batch_real/complex: Batched FFT workspaces
        // Process multiple vectors in one MKL call for efficiency
        // Reduces function call overhead, improves cache utilization
        
        // === Calculate optimal batch size based on L3 cache ===
        // Target ~8MB working set to fit in L3 cache of most modern CPUs
        // Too large: cache thrashing; too small: MKL thread underutilization
        {
            int vec_bytes = fft_n * sizeof(ssa_real);
            int target_batch = (8 * 1024 * 1024) / vec_bytes;  // Target 8MB
            
            // Clamp to reasonable limits
            if (target_batch < 4) target_batch = 4;      // Minimum for ILP
            if (target_batch > 64) target_batch = 64;    // Diminishing returns
            
            // Round down to power of 2 (better for alignment/MKL heuristics)
            int optimal_batch = 1;
            while (optimal_batch * 2 <= target_batch) optimal_batch *= 2;
            
            ssa->batch_size = optimal_batch;
        }
        
        ssa->ws_batch_real = (ssa_real *)ssa_opt_alloc(ssa->batch_size * fft_n * sizeof(ssa_real));
        ssa->ws_batch_complex = (ssa_real *)ssa_opt_alloc(ssa->batch_size * 2 * ssa->r2c_len * sizeof(ssa_real));

        // fft_x: Pre-computed FFT of the input signal
        // This is the key optimization: signal doesn't change during decomposition,
        // so we compute FFT(x) once here and reuse it in every Hankel matvec
        // Saves one FFT per matvec operation
        ssa->fft_x = (ssa_real *)ssa_opt_alloc(2 * ssa->r2c_len * sizeof(ssa_real));

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
        status = DftiCreateDescriptor(&ssa->fft_r2c, SSA_DFTI_PRECISION, DFTI_REAL, 1, fft_n);
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
        status = DftiCreateDescriptor(&ssa->fft_c2r, SSA_DFTI_PRECISION, DFTI_REAL, 1, fft_n);
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
        // Same as fft_r2c but processes batch_size vectors in one call
        // Used by block power iteration and randomized SVD
        status = DftiCreateDescriptor(&ssa->fft_r2c_batch, SSA_DFTI_PRECISION, DFTI_REAL, 1, fft_n);
        if (status != 0)
        {
            ssa_opt_free(ssa);
            return -1;
        }
        DftiSetValue(ssa->fft_r2c_batch, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
        DftiSetValue(ssa->fft_r2c_batch, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
        // NUMBER_OF_TRANSFORMS: how many FFTs to compute in parallel
        DftiSetValue(ssa->fft_r2c_batch, DFTI_NUMBER_OF_TRANSFORMS, (MKL_LONG)ssa->batch_size);
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
        status = DftiCreateDescriptor(&ssa->fft_c2r_batch, SSA_DFTI_PRECISION, DFTI_REAL, 1, fft_n);
        if (status != 0)
        {
            ssa_opt_free(ssa);
            return -1;
        }
        DftiSetValue(ssa->fft_c2r_batch, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
        DftiSetValue(ssa->fft_c2r_batch, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
        DftiSetValue(ssa->fft_c2r_batch, DFTI_BACKWARD_SCALE, 1.0 / fft_n);
        DftiSetValue(ssa->fft_c2r_batch, DFTI_NUMBER_OF_TRANSFORMS, (MKL_LONG)ssa->batch_size);
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
                status = DftiCreateDescriptor(&ssa->thread_fft_pool[t], SSA_DFTI_PRECISION, DFTI_REAL, 1, fft_n);
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
        memcpy(ssa->ws_real, x, N * sizeof(ssa_real)); // Copy signal
        DftiComputeForward(ssa->fft_r2c, ssa->ws_real, ssa->fft_x);

        // === Pre-compute Diagonal Averaging Weights ===
        // Reconstruction uses diagonal averaging: each signal position t receives
        // contributions from multiple (i,j) pairs where i+j = t
        // The count of such pairs is: min(t+1, L, K, N-t)
        // We store 1/count for fast division during reconstruction
        ssa->inv_diag_count = (ssa_real *)ssa_opt_alloc(N * sizeof(ssa_real));
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
            ssa->inv_diag_count[t] = (count > 0) ? (ssa_real)1.0 / count : (ssa_real)0.0;
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
        ssa->decomp_Omega = (ssa_real *)ssa_opt_alloc(K * kp * sizeof(ssa_real));

        // decomp_Y: Random projection result, L × (k+p)
        // Y = H·Ω captures (with high probability) the top-k column space of H
        // Computed via batched FFT-accelerated Hankel matvec
        ssa->decomp_Y = (ssa_real *)ssa_opt_alloc(L * kp * sizeof(ssa_real));

        // decomp_Q: Orthonormal basis from QR(Y), L × (k+p)
        // Q spans the same subspace as Y but with orthonormal columns
        // Computed via Householder QR (LAPACK DGEQRF + DORGQR)
        ssa->decomp_Q = (ssa_real *)ssa_opt_alloc(L * kp * sizeof(ssa_real));

        // decomp_B: Projected small matrix, K × (k+p)
        // B = Hᵀ·Q projects H onto the subspace Q
        // B has same top singular values as H (key insight of algorithm)
        ssa->decomp_B = (ssa_real *)ssa_opt_alloc(K * kp * sizeof(ssa_real));

        // decomp_tau: Householder reflector coefficients from QR, length (k+p)
        // Internal to LAPACK's QR factorization
        ssa->decomp_tau = (ssa_real *)ssa_opt_alloc(kp * sizeof(ssa_real));

        // decomp_B_left: Left singular vectors of B, K × (k+p)
        // Output from DGESDD (divide-and-conquer SVD)
        ssa->decomp_B_left = (ssa_real *)ssa_opt_alloc(K * kp * sizeof(ssa_real));

        // decomp_B_right_T: Right singular vectors of B (transposed), (k+p) × (k+p)
        // Used to rotate Q back to original coordinates: U = Q · V_B
        ssa->decomp_B_right_T = (ssa_real *)ssa_opt_alloc(kp * kp * sizeof(ssa_real));

        // decomp_S: Singular values of B, length (k+p)
        // These are the final singular values (same as H's top singular values)
        ssa->decomp_S = (ssa_real *)ssa_opt_alloc(kp * sizeof(ssa_real));

        // decomp_iwork: Integer workspace for DGESDD, length 8*(k+p)
        // LAPACK's divide-and-conquer SVD needs integer scratch space
        ssa->decomp_iwork = (int *)ssa_opt_alloc(8 * kp * sizeof(int));

        // === Query LAPACK Workspace Size ===
        // LAPACK routines need workspace whose size depends on problem dimensions
        // We query the optimal size (lwork = -1 triggers query mode)
        ssa_real work_query;
        int lwork = -1;
        ssa_LAPACKE_gesdd_work(LAPACK_COL_MAJOR, 'S',     // 'S' = economy SVD
                            K, kp,                     // Matrix dimensions
                            ssa->decomp_B, K,          // Input matrix (will be overwritten)
                            ssa->decomp_S,             // Singular values output
                            ssa->decomp_B_left, K,     // Left singular vectors
                            ssa->decomp_B_right_T, kp, // Right singular vectors (transposed)
                            &work_query, lwork,        // Query mode: returns optimal size in work_query
                            ssa->decomp_iwork);

        // Allocate optimal workspace (+1 for safety margin)
        lwork = (int)work_query + 1;
        ssa->decomp_work = (ssa_real *)ssa_opt_alloc(lwork * sizeof(ssa_real));
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
        ssa->U = (ssa_real *)ssa_opt_alloc(L * actual_k * sizeof(ssa_real));

        // V: Right singular vectors of H, K × k
        // Each column V[:,i] is the corresponding "principal component"
        ssa->V = (ssa_real *)ssa_opt_alloc(K * actual_k * sizeof(ssa_real));

        // sigma: Singular values, length k
        // σ₀ ≥ σ₁ ≥ ... ≥ σₖ₋₁ (descending order)
        ssa->sigma = (ssa_real *)ssa_opt_alloc(actual_k * sizeof(ssa_real));

        // eigenvalues: λᵢ = σᵢ², length k
        // Eigenvalues of HᵀH (variance captured by each component)
        ssa->eigenvalues = (ssa_real *)ssa_opt_alloc(actual_k * sizeof(ssa_real));

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

    int ssa_opt_update_signal(SSA_Opt *ssa, const ssa_real *new_x)
    {
        if (!ssa || !ssa->initialized || !new_x)
            return -1;
        int N = ssa->N, fft_len = ssa->fft_len;
        // Invalidate cached FFTs and decomposition
        ssa_opt_free_cached_ffts(ssa);
        ssa->decomposed = false;
        // Update FFT(x) - just memcpy + one FFT
        ssa_opt_zero(ssa->ws_real, fft_len);
        memcpy(ssa->ws_real, new_x, N * sizeof(ssa_real));
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
        ssa->U = (ssa_real *)ssa_opt_alloc(L * k * sizeof(ssa_real));
        ssa->V = (ssa_real *)ssa_opt_alloc(K * k * sizeof(ssa_real));
        ssa->sigma = (ssa_real *)ssa_opt_alloc(k * sizeof(ssa_real));
        ssa->eigenvalues = (ssa_real *)ssa_opt_alloc(k * sizeof(ssa_real));
        if (!ssa->U || !ssa->V || !ssa->sigma || !ssa->eigenvalues)
            return -1;
        ssa->n_components = k;
        ssa_real *u = (ssa_real *)ssa_opt_alloc(L * sizeof(ssa_real));
        ssa_real *v = (ssa_real *)ssa_opt_alloc(K * sizeof(ssa_real));
        ssa_real *v_new = (ssa_real *)ssa_opt_alloc(K * sizeof(ssa_real));
        ssa->ws_proj = (ssa_real *)ssa_opt_alloc(k * sizeof(ssa_real));
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
            ssa_vRngUniform(VSL_RNG_METHOD_UNIFORM_STD, ssa->rng, K, v, -0.5, 0.5);
            if (comp > 0)
            {
                ssa_cblas_gemv(CblasColMajor, CblasTrans, K, comp, 1.0, ssa->V, K, v, 1, 0.0, ssa->ws_proj, 1);
                ssa_cblas_gemv(CblasColMajor, CblasNoTrans, K, comp, -1.0, ssa->V, K, ssa->ws_proj, 1, 1.0, v, 1);
            }
            ssa_opt_normalize(v, K);
            for (int iter = 0; iter < max_iter; iter++)
            {
                ssa_opt_hankel_matvec(ssa, v, u);
                if (comp > 0)
                {
                    ssa_cblas_gemv(CblasColMajor, CblasTrans, L, comp, 1.0, ssa->U, L, u, 1, 0.0, ssa->ws_proj, 1);
                    ssa_cblas_gemv(CblasColMajor, CblasNoTrans, L, comp, -1.0, ssa->U, L, ssa->ws_proj, 1, 1.0, u, 1);
                }
                ssa_opt_hankel_matvec_T(ssa, u, v_new);
                if (comp > 0)
                {
                    ssa_cblas_gemv(CblasColMajor, CblasTrans, K, comp, 1.0, ssa->V, K, v_new, 1, 0.0, ssa->ws_proj, 1);
                    ssa_cblas_gemv(CblasColMajor, CblasNoTrans, K, comp, -1.0, ssa->V, K, ssa->ws_proj, 1, 1.0, v_new, 1);
                }
                ssa_opt_normalize(v_new, K);
                ssa_real diff_same = 0.0, diff_flip = 0.0;
                for (int i = 0; i < K; i++)
                {
                    ssa_real d_same = v[i] - v_new[i], d_flip = v[i] + v_new[i];
                    diff_same += d_same * d_same;
                    diff_flip += d_flip * d_flip;
                }
                ssa_real diff = (diff_same < diff_flip) ? diff_same : diff_flip;
                ssa_opt_copy(v_new, v, K);
                if (ssa_sqrt(diff) < SSA_CONVERGENCE_TOL && iter > 10)
                    break;
            }
            if (comp > 0)
            {
                ssa_cblas_gemv(CblasColMajor, CblasTrans, K, comp, 1.0, ssa->V, K, v, 1, 0.0, ssa->ws_proj, 1);
                ssa_cblas_gemv(CblasColMajor, CblasNoTrans, K, comp, -1.0, ssa->V, K, ssa->ws_proj, 1, 1.0, v, 1);
            }
            ssa_opt_normalize(v, K);
            ssa_opt_hankel_matvec(ssa, v, u);
            if (comp > 0)
            {
                ssa_cblas_gemv(CblasColMajor, CblasTrans, L, comp, 1.0, ssa->U, L, u, 1, 0.0, ssa->ws_proj, 1);
                ssa_cblas_gemv(CblasColMajor, CblasNoTrans, L, comp, -1.0, ssa->U, L, ssa->ws_proj, 1, 1.0, u, 1);
            }
            ssa_real sigma = ssa_opt_normalize(u, L);
            ssa_opt_hankel_matvec_T(ssa, u, v);
            if (comp > 0)
            {
                ssa_cblas_gemv(CblasColMajor, CblasTrans, K, comp, 1.0, ssa->V, K, v, 1, 0.0, ssa->ws_proj, 1);
                ssa_cblas_gemv(CblasColMajor, CblasNoTrans, K, comp, -1.0, ssa->V, K, ssa->ws_proj, 1, 1.0, v, 1);
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
                    ssa_real tmp = ssa->sigma[i];
                    ssa->sigma[i] = ssa->sigma[j];
                    ssa->sigma[j] = tmp;
                    tmp = ssa->eigenvalues[i];
                    ssa->eigenvalues[i] = ssa->eigenvalues[j];
                    ssa->eigenvalues[j] = tmp;
                    ssa_cblas_swap(L, &ssa->U[i * L], 1, &ssa->U[j * L], 1);
                    ssa_cblas_swap(K, &ssa->V[i * K], 1, &ssa->V[j * K], 1);
                }
        for (int i = 0; i < k; i++)
        {
            ssa_real sum = 0;
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
        ssa_real *Omega = ssa->decomp_Omega;
        ssa_real *Y = ssa->decomp_Y;
        ssa_real *Q = ssa->decomp_Q;
        ssa_real *B = ssa->decomp_B;
        ssa_real *tau = ssa->decomp_tau;
        ssa_real *B_left = ssa->decomp_B_left;
        ssa_real *B_right_T = ssa->decomp_B_right_T;
        ssa_real *S_svd = ssa->decomp_S;
        ssa_real *work = ssa->decomp_work;
        int *iwork = ssa->decomp_iwork;
        int lwork = ssa->prepared_lwork;

        // U, V, sigma, eigenvalues already allocated by prepare()
        // Just update count to reflect actual k being used
        ssa->n_components = k;
        ssa->total_variance = 0.0;

        // Step 1: Random projection Y = H·Ω (captures column space of H)
        ssa_vRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER, ssa->rng, K * kp, Omega, 0.0, 1.0);
        ssa_opt_hankel_matvec_block(ssa, Omega, Y, kp);

        // Step 2: QR factorization Q = orth(Y)
        ssa_cblas_copy(L * kp, Y, 1, Q, 1);
        ssa_LAPACKE_geqrf(LAPACK_COL_MAJOR, L, kp, Q, L, tau);
        ssa_LAPACKE_orgqr(LAPACK_COL_MAJOR, L, kp, kp, Q, L, tau);

        // Step 3: Project to small matrix B = Hᵀ·Q (kp×K matrix, but stored K×kp)
        ssa_opt_hankel_matvec_T_block(ssa, Q, B, kp);

        // Step 4: SVD of small matrix B = Uᵦ·Σ·Vᵀ (GESDD divide-and-conquer)
        int info = ssa_LAPACKE_gesdd_work(LAPACK_COL_MAJOR, 'S', K, kp, B, K, S_svd, B_left, K, B_right_T, kp, work, lwork, iwork);
        if (info != 0)
            return -1;

        // Step 5: Recover U = Q·Vᵦᵀ (rotate Q by right singular vectors of B)
        ssa_cblas_gemm(CblasColMajor, CblasNoTrans, CblasTrans, L, k, kp, 1.0, Q, L, B_right_T, kp, 0.0, ssa->U, L);
        for (int i = 0; i < k; i++)
            ssa_cblas_copy(K, &B_left[i * K], 1, &ssa->V[i * K], 1);

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
            ssa_real sum = 0;
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
            block_size = ssa->batch_size;  // Use dynamically computed batch size
        int b = ssa_opt_min(block_size, ssa_opt_min(k, ssa_opt_min(L, K)));
        k = ssa_opt_min(k, ssa_opt_min(L, K));
        ssa->U = (ssa_real *)ssa_opt_alloc(L * k * sizeof(ssa_real));
        ssa->V = (ssa_real *)ssa_opt_alloc(K * k * sizeof(ssa_real));
        ssa->sigma = (ssa_real *)ssa_opt_alloc(k * sizeof(ssa_real));
        ssa->eigenvalues = (ssa_real *)ssa_opt_alloc(k * sizeof(ssa_real));
        if (!ssa->U || !ssa->V || !ssa->sigma || !ssa->eigenvalues)
            return -1;
        ssa->n_components = k;
        ssa->total_variance = 0.0;
        ssa_real *V_block = (ssa_real *)ssa_opt_alloc(K * b * sizeof(ssa_real));
        ssa_real *U_block = (ssa_real *)ssa_opt_alloc(L * b * sizeof(ssa_real));
        ssa_real *U_block2 = (ssa_real *)ssa_opt_alloc(L * b * sizeof(ssa_real));
        ssa_real *tau_u = (ssa_real *)ssa_opt_alloc(b * sizeof(ssa_real));
        ssa_real *tau_v = (ssa_real *)ssa_opt_alloc(b * sizeof(ssa_real));
        ssa_real *M = (ssa_real *)ssa_opt_alloc(b * b * sizeof(ssa_real));
        ssa_real *U_small = (ssa_real *)ssa_opt_alloc(b * b * sizeof(ssa_real));
        ssa_real *Vt_small = (ssa_real *)ssa_opt_alloc(b * b * sizeof(ssa_real));
        ssa_real *S_small = (ssa_real *)ssa_opt_alloc(b * sizeof(ssa_real));
        ssa_real *superb = (ssa_real *)ssa_opt_alloc(b * sizeof(ssa_real));
        ssa_real *work = (ssa_real *)ssa_opt_alloc(ssa_opt_max(L, K) * b * sizeof(ssa_real));
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
            ssa_vRngUniform(VSL_RNG_METHOD_UNIFORM_STD, ssa->rng, K * cur_b, V_block, -0.5, 0.5);
            if (comp > 0)
            {
                ssa_cblas_gemm(CblasColMajor, CblasTrans, CblasNoTrans, comp, cur_b, K, 1.0, ssa->V, K, V_block, K, 0.0, work, comp);
                ssa_cblas_gemm(CblasColMajor, CblasNoTrans, CblasNoTrans, K, cur_b, comp, -1.0, ssa->V, K, work, comp, 1.0, V_block, K);
            }
            ssa_LAPACKE_geqrf(LAPACK_COL_MAJOR, K, cur_b, V_block, K, tau_v);
            ssa_LAPACKE_orgqr(LAPACK_COL_MAJOR, K, cur_b, cur_b, V_block, K, tau_v);
            const int QR_INTERVAL = 5;
            for (int iter = 0; iter < max_iter; iter++)
            {
                ssa_opt_hankel_matvec_block(ssa, V_block, U_block, cur_b);
                if (comp > 0)
                {
                    ssa_cblas_gemm(CblasColMajor, CblasTrans, CblasNoTrans, comp, cur_b, L, 1.0, ssa->U, L, U_block, L, 0.0, work, comp);
                    ssa_cblas_gemm(CblasColMajor, CblasNoTrans, CblasNoTrans, L, cur_b, comp, -1.0, ssa->U, L, work, comp, 1.0, U_block, L);
                }
                if ((iter % QR_INTERVAL == 0) || (iter == max_iter - 1))
                {
                    ssa_LAPACKE_geqrf(LAPACK_COL_MAJOR, L, cur_b, U_block, L, tau_u);
                    ssa_LAPACKE_orgqr(LAPACK_COL_MAJOR, L, cur_b, cur_b, U_block, L, tau_u);
                }
                ssa_opt_hankel_matvec_T_block(ssa, U_block, V_block, cur_b);
                if (comp > 0)
                {
                    ssa_cblas_gemm(CblasColMajor, CblasTrans, CblasNoTrans, comp, cur_b, K, 1.0, ssa->V, K, V_block, K, 0.0, work, comp);
                    ssa_cblas_gemm(CblasColMajor, CblasNoTrans, CblasNoTrans, K, cur_b, comp, -1.0, ssa->V, K, work, comp, 1.0, V_block, K);
                }
                if ((iter > 0 && iter % QR_INTERVAL == 0) || (iter == max_iter - 1))
                {
                    ssa_LAPACKE_geqrf(LAPACK_COL_MAJOR, K, cur_b, V_block, K, tau_v);
                    ssa_LAPACKE_orgqr(LAPACK_COL_MAJOR, K, cur_b, cur_b, V_block, K, tau_v);
                }
            }
            ssa_opt_hankel_matvec_block(ssa, V_block, U_block2, cur_b);
            if (comp > 0)
            {
                ssa_cblas_gemm(CblasColMajor, CblasTrans, CblasNoTrans, comp, cur_b, L, 1.0, ssa->U, L, U_block2, L, 0.0, work, comp);
                ssa_cblas_gemm(CblasColMajor, CblasNoTrans, CblasNoTrans, L, cur_b, comp, -1.0, ssa->U, L, work, comp, 1.0, U_block2, L);
            }
            ssa_cblas_gemm(CblasColMajor, CblasTrans, CblasNoTrans, cur_b, cur_b, L, 1.0, U_block, L, U_block2, L, 0.0, M, cur_b);
            int svd_info = ssa_LAPACKE_gesvd(LAPACK_COL_MAJOR, 'A', 'A', cur_b, cur_b, M, cur_b, S_small, U_small, cur_b, Vt_small, cur_b, superb);
            if (svd_info != 0)
            {
                for (int i = 0; i < cur_b; i++)
                    S_small[i] = ssa_cblas_nrm2(L, &U_block2[i * L], 1);
                memset(U_small, 0, cur_b * cur_b * sizeof(ssa_real));
                memset(Vt_small, 0, cur_b * cur_b * sizeof(ssa_real));
                for (int i = 0; i < cur_b; i++)
                {
                    U_small[i + i * cur_b] = 1.0;
                    Vt_small[i + i * cur_b] = 1.0;
                }
            }
            ssa_cblas_gemm(CblasColMajor, CblasNoTrans, CblasNoTrans, L, cur_b, cur_b, 1.0, U_block, L, U_small, cur_b, 0.0, U_block2, L);
            ssa_cblas_gemm(CblasColMajor, CblasNoTrans, CblasTrans, K, cur_b, cur_b, 1.0, V_block, K, Vt_small, cur_b, 0.0, work, K);
            for (int i = 0; i < cur_b; i++)
            {
                ssa_real sigma = S_small[i];
                ssa->sigma[comp + i] = sigma;
                ssa->eigenvalues[comp + i] = sigma * sigma;
                ssa->total_variance += sigma * sigma;
                ssa_cblas_copy(L, &U_block2[i * L], 1, &ssa->U[(comp + i) * L], 1);
                ssa_cblas_copy(K, &work[i * K], 1, &ssa->V[(comp + i) * K], 1);
            }
            for (int i = 0; i < cur_b; i++)
                ssa_cblas_copy(L, &ssa->U[(comp + i) * L], 1, &U_block[i * L], 1);
            ssa_opt_hankel_matvec_T_block(ssa, U_block, V_block, cur_b);
            if (comp > 0)
            {
                ssa_cblas_gemm(CblasColMajor, CblasTrans, CblasNoTrans, comp, cur_b, K, 1.0, ssa->V, K, V_block, K, 0.0, work, comp);
                ssa_cblas_gemm(CblasColMajor, CblasNoTrans, CblasNoTrans, K, cur_b, comp, -1.0, ssa->V, K, work, comp, 1.0, V_block, K);
            }
            for (int i = 0; i < cur_b; i++)
            {
                ssa_real sigma = ssa->sigma[comp + i];
                ssa_real *v_col = &V_block[i * K];
                if (sigma > 1e-12)
                    ssa_cblas_scal(K, 1.0 / sigma, v_col, 1);
                ssa_cblas_copy(K, v_col, 1, &ssa->V[(comp + i) * K], 1);
            }
            comp += cur_b;
        }
        for (int i = 0; i < k - 1; i++)
            for (int j = i + 1; j < k; j++)
                if (ssa->sigma[j] > ssa->sigma[i])
                {
                    ssa_real tmp = ssa->sigma[i];
                    ssa->sigma[i] = ssa->sigma[j];
                    ssa->sigma[j] = tmp;
                    tmp = ssa->eigenvalues[i];
                    ssa->eigenvalues[i] = ssa->eigenvalues[j];
                    ssa->eigenvalues[j] = tmp;
                    ssa_cblas_swap(L, &ssa->U[i * L], 1, &ssa->U[j * L], 1);
                    ssa_cblas_swap(K, &ssa->V[i * K], 1, &ssa->V[j * K], 1);
                }
        for (int i = 0; i < k; i++)
        {
            ssa_real sum = 0;
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
        ssa_real *U_new = (ssa_real *)ssa_opt_alloc(L * new_k * sizeof(ssa_real));
        ssa_real *V_new = (ssa_real *)ssa_opt_alloc(K * new_k * sizeof(ssa_real));
        ssa_real *sigma_new = (ssa_real *)ssa_opt_alloc(new_k * sizeof(ssa_real));
        ssa_real *eigen_new = (ssa_real *)ssa_opt_alloc(new_k * sizeof(ssa_real));
        if (!U_new || !V_new || !sigma_new || !eigen_new)
        {
            ssa_opt_free_ptr(U_new);
            ssa_opt_free_ptr(V_new);
            ssa_opt_free_ptr(sigma_new);
            ssa_opt_free_ptr(eigen_new);
            return -1;
        }
        memcpy(U_new, ssa->U, L * old_k * sizeof(ssa_real));
        memcpy(V_new, ssa->V, K * old_k * sizeof(ssa_real));
        memcpy(sigma_new, ssa->sigma, old_k * sizeof(ssa_real));
        memcpy(eigen_new, ssa->eigenvalues, old_k * sizeof(ssa_real));
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
        ssa->ws_proj = (ssa_real *)ssa_opt_alloc(new_k * sizeof(ssa_real));
        if (!ssa->ws_proj)
            return -1;
        ssa_real *u = (ssa_real *)ssa_opt_alloc(L * sizeof(ssa_real)), *v = (ssa_real *)ssa_opt_alloc(K * sizeof(ssa_real)), *v_new = (ssa_real *)ssa_opt_alloc(K * sizeof(ssa_real));
        if (!u || !v || !v_new)
        {
            ssa_opt_free_ptr(u);
            ssa_opt_free_ptr(v);
            ssa_opt_free_ptr(v_new);
            return -1;
        }
        for (int comp = old_k; comp < new_k; comp++)
        {
            ssa_vRngUniform(VSL_RNG_METHOD_UNIFORM_STD, ssa->rng, K, v, -0.5, 0.5);
            ssa_cblas_gemv(CblasColMajor, CblasTrans, K, comp, 1.0, ssa->V, K, v, 1, 0.0, ssa->ws_proj, 1);
            ssa_cblas_gemv(CblasColMajor, CblasNoTrans, K, comp, -1.0, ssa->V, K, ssa->ws_proj, 1, 1.0, v, 1);
            ssa_opt_normalize(v, K);
            for (int iter = 0; iter < max_iter; iter++)
            {
                ssa_opt_hankel_matvec(ssa, v, u);
                ssa_cblas_gemv(CblasColMajor, CblasTrans, L, comp, 1.0, ssa->U, L, u, 1, 0.0, ssa->ws_proj, 1);
                ssa_cblas_gemv(CblasColMajor, CblasNoTrans, L, comp, -1.0, ssa->U, L, ssa->ws_proj, 1, 1.0, u, 1);
                ssa_opt_normalize(u, L);
                ssa_opt_hankel_matvec_T(ssa, u, v_new);
                ssa_cblas_gemv(CblasColMajor, CblasTrans, K, comp, 1.0, ssa->V, K, v_new, 1, 0.0, ssa->ws_proj, 1);
                ssa_cblas_gemv(CblasColMajor, CblasNoTrans, K, comp, -1.0, ssa->V, K, ssa->ws_proj, 1, 1.0, v_new, 1);
                ssa_opt_normalize(v_new, K);
                ssa_real diff_same = 0.0, diff_flip = 0.0;
                for (int i = 0; i < K; i++)
                {
                    ssa_real d_same = v[i] - v_new[i], d_flip = v[i] + v_new[i];
                    diff_same += d_same * d_same;
                    diff_flip += d_flip * d_flip;
                }
                ssa_real diff = (diff_same < diff_flip) ? diff_same : diff_flip;
                ssa_opt_copy(v_new, v, K);
                if (ssa_sqrt(diff) < SSA_CONVERGENCE_TOL && iter > 10)
                    break;
            }
            ssa_cblas_gemv(CblasColMajor, CblasTrans, K, comp, 1.0, ssa->V, K, v, 1, 0.0, ssa->ws_proj, 1);
            ssa_cblas_gemv(CblasColMajor, CblasNoTrans, K, comp, -1.0, ssa->V, K, ssa->ws_proj, 1, 1.0, v, 1);
            ssa_opt_normalize(v, K);
            ssa_opt_hankel_matvec(ssa, v, u);
            ssa_cblas_gemv(CblasColMajor, CblasTrans, L, comp, 1.0, ssa->U, L, u, 1, 0.0, ssa->ws_proj, 1);
            ssa_cblas_gemv(CblasColMajor, CblasNoTrans, L, comp, -1.0, ssa->U, L, ssa->ws_proj, 1, 1.0, u, 1);
            ssa_real sigma = ssa_opt_normalize(u, L);
            ssa_opt_hankel_matvec_T(ssa, u, v);
            ssa_cblas_gemv(CblasColMajor, CblasTrans, K, comp, 1.0, ssa->V, K, v, 1, 0.0, ssa->ws_proj, 1);
            ssa_cblas_gemv(CblasColMajor, CblasNoTrans, K, comp, -1.0, ssa->V, K, ssa->ws_proj, 1, 1.0, v, 1);
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
                    ssa_real tmp = ssa->sigma[i];
                    ssa->sigma[i] = ssa->sigma[j];
                    ssa->sigma[j] = tmp;
                    tmp = ssa->eigenvalues[i];
                    ssa->eigenvalues[i] = ssa->eigenvalues[j];
                    ssa->eigenvalues[j] = tmp;
                    ssa_cblas_swap(L, &ssa->U[i * L], 1, &ssa->U[j * L], 1);
                    ssa_cblas_swap(K, &ssa->V[i * K], 1, &ssa->V[j * K], 1);
                }
        for (int i = old_k; i < new_k; i++)
        {
            ssa_real sum = 0;
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
        size_t cache_size = 2 * r2c_len * k * sizeof(ssa_real);

        // U_fft[i] = FFT(σᵢ · U[:,i]) - scaled left singular vectors
        // Pre-multiplying by σᵢ saves one multiply per component during reconstruction
        ssa->U_fft = (ssa_real *)ssa_opt_alloc(cache_size);

        // V_fft[i] = FFT(V[:,i]) - right singular vectors
        ssa->V_fft = (ssa_real *)ssa_opt_alloc(cache_size);

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
        ssa_real *ws_pool = (ssa_real *)ssa_opt_alloc(n_threads * fft_len * sizeof(ssa_real));
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
            ssa_real *ws_real = ws_pool + tid * fft_len;
            DFTI_DESCRIPTOR_HANDLE my_fft = ssa->thread_fft_pool[tid]; // Pre-allocated!
            int i; // Declare BEFORE pragma for MSVC OpenMP 2.0 compatibility

            #pragma omp for schedule(static)
            for (i = 0; i < k; i++)
            {
                ssa_real sigma = ssa->sigma[i];
                const ssa_real *u_vec = &ssa->U[i * L];
                ssa_real *dst = &ssa->U_fft[i * 2 * r2c_len];

                memset(ws_real, 0, fft_len * sizeof(ssa_real));
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
            ssa_real *ws_real = ws_pool + tid * fft_len;
            DFTI_DESCRIPTOR_HANDLE my_fft = ssa->thread_fft_pool[tid]; // Pre-allocated!
            int i; // Declare BEFORE pragma for MSVC OpenMP 2.0 compatibility

            #pragma omp for schedule(static)
            for (i = 0; i < k; i++)
            {
                const ssa_real *v_vec = &ssa->V[i * K];
                ssa_real *dst = &ssa->V_fft[i * 2 * r2c_len];

                memset(ws_real, 0, fft_len * sizeof(ssa_real));
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
            ssa_real sigma = ssa->sigma[i];
            const ssa_real *u_vec = &ssa->U[i * L];       // u_i has length L
            ssa_real *dst = &ssa->U_fft[i * 2 * r2c_len]; // Destination in cache

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
            const ssa_real *v_vec = &ssa->V[i * K];       // v_i has length K
            ssa_real *dst = &ssa->V_fft[i * 2 * r2c_len]; // Destination in cache

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
            ssa_real scale = 1.0 / fft_len; // IFFT normalization

            // wcorr_ws_complex: workspace for batched FFT output
            // Holds k complex vectors of length r2c_len
            ssa->wcorr_ws_complex = (ssa_real *)ssa_opt_alloc(k * 2 * r2c_len * sizeof(ssa_real));

            // wcorr_h: reconstructed components before weighting, k × fft_len
            // Each row is IFFT(U_fft[i] ⊙ V_fft[i]) = diagonal sums for component i
            ssa->wcorr_h = (ssa_real *)ssa_opt_alloc(k * fft_len * sizeof(ssa_real));

            // wcorr_G: weighted/normalized matrix for DSYRK, k × N
            // G[i,t] = (hᵢ[t] / ||hᵢ||_w) · √w[t]
            // Then W = G · Gᵀ gives the W-correlation matrix directly
            ssa->wcorr_G = (ssa_real *)ssa_opt_alloc(k * N * sizeof(ssa_real));

            // wcorr_sqrt_inv_c: precomputed √(1/count[t]) for fast weighting
            // Used to convert raw diagonal sums to weighted values
            ssa->wcorr_sqrt_inv_c = (ssa_real *)ssa_opt_alloc(N * sizeof(ssa_real));

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
                ssa->wcorr_sqrt_inv_c[t] = ssa_sqrt(ssa->inv_diag_count[t]);

            // === Create Batched IFFT Descriptor for W-correlation ===
            // This descriptor computes k IFFTs in a single MKL call
            // Used to transform all k component products simultaneously
            status = DftiCreateDescriptor(&ssa->fft_c2r_wcorr, SSA_DFTI_PRECISION, DFTI_REAL, 1, fft_len);
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

    int ssa_opt_reconstruct(const SSA_Opt *ssa, const int *group, int n_group, ssa_real *output)
    {
        if (!ssa || !ssa->decomposed || !group || !output || n_group < 1)
            return -1;

        int N = ssa->N, L = ssa->L, K = ssa->K;
        int fft_len = ssa->fft_len, r2c_len = ssa->r2c_len;
        SSA_Opt *ssa_mut = (SSA_Opt *)ssa;

        // Use batch workspace as accumulator
        ssa_real *freq_accum = ssa_mut->ws_batch_complex;
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

                const ssa_real *u_fft_cached = &ssa->U_fft[idx * 2 * r2c_len];
                const ssa_real *v_fft_cached = &ssa->V_fft[idx * 2 * r2c_len];

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
            ssa_real *temp_fft2 = ssa_mut->ws_batch_complex + 2 * r2c_len;

            for (int g = 0; g < n_group; g++)
            {
                int idx = group[g];
                if (idx < 0 || idx >= ssa->n_components)
                    continue;

                ssa_real sigma = ssa->sigma[idx];
                const ssa_real *u_vec = &ssa->U[idx * L];
                const ssa_real *v_vec = &ssa->V[idx * K];

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

        // Apply diagonal averaging weights directly to output (fused copy+multiply)
        // This eliminates an extra memory pass vs memcpy followed by in-place vMul
        ssa_vMul_real(N, ssa_mut->ws_real, ssa->inv_diag_count, output);

        return 0;
    }

#endif // SSA_OPT_IMPLEMENTATION

#ifdef __cplusplus
}
#endif

#endif // SSA_OPT_R2C_H
