/*
 * ============================================================================
 * SSA-OPT: High-Performance Singular Spectrum Analysis (MKL R2C Version)
 * ============================================================================
 *
 * PURPOSE:
 *   Production-grade SSA implementation using Intel MKL for maximum performance.
 *   This version uses Real-to-Complex FFTs for ~2x speedup on FFT operations.
 *
 * R2C FFT OPTIMIZATION:
 *   For real input of length N, R2C FFT exploits conjugate symmetry:
 *   - Output is only N/2+1 complex values (vs N for C2C)
 *   - Complex multiply is half the work
 *   - C2R IFFT outputs real directly (no stride-2 extraction)
 *   - ~50% memory reduction for FFT buffers
 *
 * WHAT IS SSA?
 *   Singular Spectrum Analysis decomposes a time series into trend, periodic
 *   components, and noise by embedding it into a Hankel matrix and computing
 *   its SVD. This implementation avoids forming the explicit matrix.
 *
 * WHAT THIS IMPLEMENTATION PROVIDES:
 *   - O(N log N) Hankel matvec via FFT convolution
 *   - Three decomposition methods:
 *       1. Sequential power iteration (baseline)
 *       2. Block power iteration with Rayleigh-Ritz (~3x faster)
 *       3. Randomized SVD (~15-25x faster)
 *   - Batched FFT operations for throughput
 *   - MKL-optimized BLAS/LAPACK throughout
 *
 * USAGE:
 *   #define SSA_OPT_IMPLEMENTATION
 *   #include "ssa_opt_r2c.h"
 *
 *   SSA_Opt ssa = {0};
 *   ssa_opt_init(&ssa, signal, N, L);
 *   ssa_opt_decompose_randomized(&ssa, k, 8);
 *   ssa_opt_reconstruct(&ssa, components, n_components, output);
 *   ssa_opt_free(&ssa);
 *
 * BUILD (Windows + MKL):
 *   cl /O2 your_code.c /I"%MKLROOT%\include" /link /LIBPATH:"%MKLROOT%\lib"
 *      mkl_intel_lp64.lib mkl_intel_thread.lib mkl_core.lib
 *
 * BUILD (Linux + MKL):
 *   gcc -O2 your_code.c -I${MKLROOT}/include -L${MKLROOT}/lib/intel64
 *       -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm
 *
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

#ifdef __cplusplus
extern "C"
{
#endif

    // ============================================================================
    // Configuration
    // ============================================================================

#ifndef SSA_CONVERGENCE_TOL
#define SSA_CONVERGENCE_TOL 1e-12
#endif

#define SSA_ALIGN 64

#ifndef SSA_BATCH_SIZE
#define SSA_BATCH_SIZE 32
#endif

    // ============================================================================
    // Data Structures
    // ============================================================================

    /**
     * @brief Main SSA context holding all state, workspace, and results.
     *
     * R2C CHANGES FROM C2C VERSION:
     *   - fft_x is now r2c_len complex values (fft_len/2+1), not fft_len
     *   - Separate R2C (forward) and C2R (backward) FFT descriptors
     *   - Workspace buffers split into real and complex parts
     *   - ~50% less memory for FFT buffers
     *
     * MEMORY LAYOUT:
     *   - All matrices are column-major (BLAS/LAPACK convention)
     *   - U is L × k: column i is the i-th left singular vector
     *   - V is K × k: column i is the i-th right singular vector
     *   - Complex arrays are interleaved: [re₀, im₀, re₁, im₁, ...]
     */
    typedef struct
    {
        // Dimensions
        int N;       ///< Original series length
        int L;       ///< Window length (embedding dimension)
        int K;       ///< K = N - L + 1
        int fft_len; ///< FFT length (next power of 2)
        int r2c_len; ///< R2C output length = fft_len/2 + 1 complex values

        // MKL FFT Descriptors (R2C/C2R - separate for forward/backward)
        DFTI_DESCRIPTOR_HANDLE fft_r2c;       ///< Single R2C (real → complex)
        DFTI_DESCRIPTOR_HANDLE fft_c2r;       ///< Single C2R (complex → real)
        DFTI_DESCRIPTOR_HANDLE fft_r2c_batch; ///< Batched R2C
        DFTI_DESCRIPTOR_HANDLE fft_c2r_batch; ///< Batched C2R

        // Random Number Generator
        VSLStreamStatePtr rng;

        // Precomputed Data (R2C format - half the size of C2C!)
        double *fft_x; ///< FFT of input signal, r2c_len complex values (interleaved)

        // Workspace Buffers (split real/complex for R2C)
        double *ws_real;    ///< Real workspace, length fft_len
        double *ws_complex; ///< Complex workspace, 2 * r2c_len doubles
        double *ws_real2;   ///< Second real workspace
        double *ws_proj;    ///< Projection coefficients for orthogonalization

        // Batch workspace (for block methods)
        double *ws_batch_real;    ///< SSA_BATCH_SIZE * fft_len reals
        double *ws_batch_complex; ///< SSA_BATCH_SIZE * 2 * r2c_len doubles

        // Results
        double *U;           ///< Left singular vectors, L × k, column-major
        double *V;           ///< Right singular vectors, K × k, column-major
        double *sigma;       ///< Singular values
        double *eigenvalues; ///< Squared singular values
        int n_components;    ///< Number of computed components

        // Reconstruction Optimization
        double *inv_diag_count; ///< Precomputed 1/count for diagonal averaging
        double *U_fft;          ///< Cached FFT of scaled U vectors (R2C format)
        double *V_fft;          ///< Cached FFT of V vectors (R2C format)
        bool fft_cached;        ///< True if U_fft/V_fft are populated

        // State
        bool initialized;
        bool decomposed;
        double total_variance;
    } SSA_Opt;

    /**
     * @brief Statistics for automatic component selection and analysis.
     */
    typedef struct
    {
        int n;                   ///< Number of components analyzed
        double *singular_values; ///< Copy of singular values σ₁ ≥ σ₂ ≥ ... ≥ σₙ
        double *log_sv;          ///< log(σᵢ) for scree plot visualization
        double *gaps;            ///< Gap ratios: σᵢ/σᵢ₊₁ for i=0..n-2
        double *cumulative_var;  ///< Cumulative explained variance ratio
        double *second_diff;     ///< Second difference of log(σ) for elbow detection
        int suggested_signal;    ///< Suggested signal component count
        double gap_threshold;    ///< Gap ratio at the suggested cutoff point
    } SSA_ComponentStats;

    /**
     * @brief Linear Recurrence Formula coefficients for SSA forecasting.
     */
    typedef struct
    {
        double *R;          ///< Recurrence coefficients, length L-1
        int L;              ///< Window length
        double verticality; ///< ν² = ‖π‖². Must be < 1 for valid forecast.
        bool valid;         ///< True if verticality < 1
    } SSA_LRF;

    /**
     * @brief Multivariate SSA context for joint analysis of M correlated time series.
     */
    typedef struct
    {
        // Dimensions
        int M;       ///< Number of time series
        int N;       ///< Length of each series
        int L;       ///< Window length (embedding dimension)
        int K;       ///< K = N - L + 1
        int fft_len; ///< FFT length (next power of 2)
        int r2c_len; ///< R2C output length = fft_len/2 + 1

        // MKL FFT Descriptors (R2C/C2R)
        DFTI_DESCRIPTOR_HANDLE fft_r2c;       ///< Single R2C
        DFTI_DESCRIPTOR_HANDLE fft_c2r;       ///< Single C2R
        DFTI_DESCRIPTOR_HANDLE fft_r2c_batch; ///< Batched R2C for M series
        DFTI_DESCRIPTOR_HANDLE fft_c2r_batch; ///< Batched C2R for M series

        // Random Number Generator
        VSLStreamStatePtr rng;

        // Precomputed FFT(x) for each series (R2C format)
        double *fft_x; ///< M × r2c_len × 2 doubles (M r2c_len-complex arrays)

        // Workspace Buffers
        double *ws_real;          ///< fft_len reals
        double *ws_complex;       ///< 2 * r2c_len doubles
        double *ws_batch_real;    ///< M * fft_len reals
        double *ws_batch_complex; ///< M * 2 * r2c_len doubles

        // Results
        double *U;           ///< Left singular vectors, (M*L) × k, column-major
        double *V;           ///< Right singular vectors, K × k, column-major
        double *sigma;       ///< Singular values
        double *eigenvalues; ///< Squared singular values
        int n_components;    ///< Number of computed components

        // Reconstruction Optimization
        double *inv_diag_count; ///< Precomputed 1/count for diagonal averaging

        // State
        bool initialized;
        bool decomposed;
        double total_variance;
    } MSSA_Opt;

    // ============================================================================
    // Public API
    // ============================================================================

    int ssa_opt_init(SSA_Opt *ssa, const double *x, int N, int L);
    void ssa_opt_free(SSA_Opt *ssa);

    /**
     * @brief Decompose time series via sequential power iteration.
     *
     * Computes the first k singular triplets (Uᵢ, σᵢ, Vᵢ) of the Hankel matrix.
     * Uses deflated power iteration: each component is found by iterating
     * u = H·v, v = Hᵀ·u until convergence, then deflating.
     *
     * @param ssa       Initialized SSA context
     * @param k         Number of components to compute
     * @param max_iter  Maximum iterations per component (100-200 typical)
     * @return          0 on success, -1 on error
     *
     * @note For k > 10, consider ssa_opt_decompose_randomized() which is faster.
     */
    int ssa_opt_decompose(SSA_Opt *ssa, int k, int max_iter);

    /**
     * @brief Decompose via block power iteration (faster for k > 5).
     *
     * Instead of computing one component at a time, processes block_size
     * components simultaneously. Uses QR factorization for orthogonalization.
     *
     * @param ssa        Initialized SSA context
     * @param k          Number of components to compute
     * @param block_size Components per block (8-32 typical)
     * @param max_iter   Maximum iterations per block
     * @return           0 on success, -1 on error
     */
    int ssa_opt_decompose_block(SSA_Opt *ssa, int k, int block_size, int max_iter);

    /**
     * @brief Decompose via randomized SVD (fastest for k << min(L,K)).
     *
     * Algorithm: Halko-Martinsson-Tropp randomized range finder
     *   1. Generate random test matrix Ω (K × (k+p))
     *   2. Y = H @ Ω, then QR factorize Y = Q @ R
     *   3. B = Hᵀ @ Q, then SVD of B
     *   4. Recover U, σ, V from Q and B's SVD
     *
     * Complexity: O((k+p) × N log N) vs O(k × max_iter × N log N) for power iteration.
     * Recommended for k > 10 when you don't need extreme accuracy.
     *
     * @param ssa          Initialized SSA context
     * @param k            Number of components to compute
     * @param oversampling Extra random vectors for accuracy (5-10 typical)
     * @return             0 on success, -1 on error
     */
    int ssa_opt_decompose_randomized(SSA_Opt *ssa, int k, int oversampling);

    /**
     * @brief Extend existing decomposition with additional components.
     *
     * Adds more singular triplets to an already decomposed SSA context without
     * recomputing existing components. Uses deflated power iteration starting
     * from random vectors orthogonal to existing V columns.
     *
     * Use case: You computed 10 components, analyzed them, and now want 20 more.
     * This is faster than decomposing from scratch with k=30.
     *
     * @param ssa          Already decomposed SSA context
     * @param additional_k Number of new components to add
     * @param max_iter     Maximum iterations per new component
     * @return             0 on success, -1 on error
     *
     * @note Invalidates cached FFTs. Results are re-sorted by σ, so component
     *       indices may change if new components have larger singular values.
     */
    int ssa_opt_extend(SSA_Opt *ssa, int additional_k, int max_iter);

    /**
     * @brief Reconstruct signal from selected components.
     *
     * Computes: output[t] = Σᵢ∈group Xᵢ[t] where Xᵢ is the reconstruction
     * from component i using diagonal averaging.
     *
     * Uses frequency-domain accumulation: all components are summed in frequency
     * domain before a single IFFT, giving O(n_group × N log N + N log N) complexity
     * instead of O(n_group × N log N) naive.
     *
     * @param ssa      Decomposed SSA context
     * @param group    Array of component indices (0-based)
     * @param n_group  Number of components in group
     * @param output   Output buffer, length N (caller allocates)
     * @return         0 on success, -1 on error
     */
    int ssa_opt_reconstruct(const SSA_Opt *ssa, const int *group, int n_group, double *output);

    /**
     * @brief Pre-cache FFTs of all U and V vectors for faster reconstruction.
     *
     * When reconstructing multiple groupings, the FFTs of U and V vectors are
     * recomputed each time. This function pre-computes and caches them.
     *
     * @param ssa  Decomposed SSA context
     * @return     0 on success, -1 on error
     *
     * @note Increases memory usage by O(k × N) complex values.
     * @note Call ssa_opt_free_cached_ffts() or ssa_opt_free() when done.
     */
    int ssa_opt_cache_ffts(SSA_Opt *ssa);

    /**
     * @brief Free cached FFT buffers.
     */
    void ssa_opt_free_cached_ffts(SSA_Opt *ssa);

    /**
     * @brief Compute the full W-correlation matrix between all SSA components.
     *
     * W-correlation measures separability between reconstructed components.
     * Values close to 0 indicate well-separated components; values close to ±1
     * indicate components that should be grouped together.
     *
     * Uses optimized direct formula: O(n × N log N + n² × N) instead of
     * reconstructing each component pair.
     *
     * @param ssa  Decomposed SSA context
     * @param W    Output matrix, n_components × n_components, row-major
     *             W[i*n + j] = W-correlation between components i and j
     * @return     0 on success, -1 on error
     */
    int ssa_opt_wcorr_matrix(const SSA_Opt *ssa, double *W);

    /**
     * @brief Compute W-correlation between two specific components.
     *
     * W-correlation is defined as:
     *   W(X_i, X_j) = <X_i, X_j>_W / (||X_i||_W × ||X_j||_W)
     *
     * where <·,·>_W is the weighted inner product with weights c[t] = min(t+1, L, K, N-t).
     * These weights account for the number of terms in diagonal averaging.
     *
     * Interpretation:
     *   - |W| ≈ 0:    Components are well-separated (orthogonal in weighted sense)
     *   - |W| ≈ 1:    Components are highly correlated (should be grouped)
     *   - Periodic components often come in pairs with |W| ≈ 1
     *
     * Uses direct formula via FFT convolution - does not require full reconstruction.
     *
     * @param ssa  Decomposed SSA context
     * @param i    First component index (0-based)
     * @param j    Second component index (0-based)
     * @return     W-correlation value in [-1, 1], or 0.0 on error
     */
    double ssa_opt_wcorr_pair(const SSA_Opt *ssa, int i, int j);

    /**
     * @brief Compute statistics for automatic component selection.
     *
     * Analyzes the singular value spectrum to help identify signal vs. noise
     * components. Several heuristics are provided:
     *
     * 1. Scree plot data: log(σᵢ) shows "elbow" at signal-noise boundary
     * 2. Gap ratios: σᵢ/σᵢ₊₁ - large gaps indicate boundaries between groups
     * 3. Cumulative variance: fraction of variance explained by first k components
     *
     * Suggested usage:
     *   - Signal components: indices 0 to stats->suggested_signal - 1
     *   - Noise components: indices stats->suggested_signal to n-1
     *
     * @param ssa    Decomposed SSA context
     * @param stats  Output structure (caller allocates, function fills arrays)
     *               Must call ssa_opt_component_stats_free() when done
     * @return       0 on success, -1 on error
     *
     * @code
     * SSA_ComponentStats stats;
     * ssa_opt_component_stats(&ssa, &stats);
     * printf("Suggested signal components: %d\n", stats.suggested_signal);
     * printf("Variance explained by signal: %.2f%%\n",
     *        stats.cumulative_var[stats.suggested_signal - 1] * 100);
     * ssa_opt_component_stats_free(&stats);
     * @endcode
     */
    int ssa_opt_component_stats(const SSA_Opt *ssa, SSA_ComponentStats *stats);

    /**
     * @brief Free memory allocated by ssa_opt_component_stats().
     */
    void ssa_opt_component_stats_free(SSA_ComponentStats *stats);

    /**
     * @brief Automatically find periodic component pairs.
     *
     * Periodic signals (e.g., sine waves) appear as pairs of consecutive
     * components with similar singular values and high W-correlation.
     * This function identifies such pairs automatically.
     *
     * @param ssa           Decomposed SSA context
     * @param pairs         Output: array of pair start indices (length max_pairs)
     *                      pairs[k] = i means components i and i+1 form a pair
     * @param max_pairs     Maximum number of pairs to find
     * @param sv_tol        Singular value similarity tolerance (e.g., 0.1 = 10%)
     *                      Pair requires |σᵢ - σᵢ₊₁| / σᵢ < sv_tol
     * @param wcorr_thresh  Minimum |W-correlation| for pair (e.g., 0.7)
     * @return              Number of pairs found, or -1 on error
     */
    int ssa_opt_find_periodic_pairs(const SSA_Opt *ssa, int *pairs, int max_pairs,
                                    double sv_tol, double wcorr_thresh);

    /**
     * @brief Compute Linear Recurrence Formula (LRF) coefficients.
     *
     * SSA-based forecasting relies on the fact that signal components satisfy
     * a linear recurrence: x[t] = Σⱼ Rⱼ × x[t-L+1+j] for j = 0..L-2.
     *
     * The LRF coefficients R are computed from the last row of U vectors.
     * The verticality coefficient ν² measures forecast stability:
     *   - ν² close to 0: stable forecast
     *   - ν² close to 1: unstable (LRF is near-singular)
     *
     * @param ssa      Decomposed SSA context
     * @param group    Components to include in LRF
     * @param n_group  Number of components
     * @param lrf      Output: LRF structure (caller allocates, function fills)
     *                 Must call ssa_opt_lrf_free() when done
     * @return         0 on success, -1 on error (including ν² ≥ 1)
     */
    int ssa_opt_compute_lrf(const SSA_Opt *ssa, const int *group, int n_group, SSA_LRF *lrf);

    /**
     * @brief Free LRF coefficients.
     */
    void ssa_opt_lrf_free(SSA_LRF *lrf);

    /**
     * @brief Forecast using LRF recurrence.
     *
     * Reconstructs signal from group, then extends using LRF:
     *   x̃[t] = Σⱼ Rⱼ × x̃[t-L+1+j] for t = N, N+1, ..., N+n_forecast-1
     *
     * @param ssa         Decomposed SSA context
     * @param group       Components to use for forecasting
     * @param n_group     Number of components
     * @param n_forecast  Number of steps to forecast
     * @param output      Output buffer, length N + n_forecast (caller allocates)
     * @return            0 on success, -1 on error
     */
    int ssa_opt_forecast(const SSA_Opt *ssa, const int *group, int n_group,
                         int n_forecast, double *output);

    /**
     * @brief Forecast with full vector forecasting (more accurate).
     *
     * Instead of scalar LRF, uses vector forecasting which extends the
     * entire Hankel structure. More accurate for multivariate patterns.
     *
     * @param ssa         Decomposed SSA context
     * @param group       Components to use for forecasting
     * @param n_group     Number of components
     * @param n_forecast  Number of steps to forecast
     * @param output      Output buffer, length N + n_forecast
     * @return            0 on success, -1 on error
     */
    int ssa_opt_forecast_full(const SSA_Opt *ssa, const int *group, int n_group,
                              int n_forecast, double *output);

    /**
     * @brief Forecast using pre-computed LRF.
     *
     * For repeated forecasting with different base signals but same LRF.
     *
     * @param lrf         Pre-computed LRF from ssa_opt_compute_lrf()
     * @param base_signal Signal to extend (length base_len, must have base_len >= L-1)
     * @param base_len    Length of base signal
     * @param n_forecast  Number of steps to forecast
     * @param output      Output buffer, length base_len + n_forecast
     * @return            0 on success, -1 on error
     */
    int ssa_opt_forecast_with_lrf(const SSA_LRF *lrf, const double *base_signal, int base_len,
                                  int n_forecast, double *output);

    // ============================================================================
    // MSSA (Multivariate SSA) API
    // ============================================================================

    /**
     * @brief Initialize MSSA context for multiple time series.
     *
     * Constructs block-Hankel matrix from M series of length N.
     *
     * @param mssa  Zero-initialized MSSA context
     * @param X     Input: M × N matrix, row-major (series i is X[i*N : (i+1)*N])
     * @param M     Number of series
     * @param N     Length of each series
     * @param L     Window length
     * @return      0 on success, -1 on error
     */
    int mssa_opt_init(MSSA_Opt *mssa, const double *X, int M, int N, int L);

    /**
     * @brief Decompose MSSA via randomized SVD.
     */
    int mssa_opt_decompose(MSSA_Opt *mssa, int k, int oversampling);

    /**
     * @brief Reconstruct single series from MSSA components.
     */
    int mssa_opt_reconstruct(const MSSA_Opt *mssa, int series_idx,
                             const int *group, int n_group, double *output);

    /**
     * @brief Reconstruct all series from MSSA components.
     */
    int mssa_opt_reconstruct_all(const MSSA_Opt *mssa,
                                 const int *group, int n_group, double *output);

    /**
     * @brief Compute contribution of each series to total variance.
     */
    int mssa_opt_series_contributions(const MSSA_Opt *mssa, double *contributions);

    /**
     * @brief Get explained variance for component range.
     */
    double mssa_opt_variance_explained(const MSSA_Opt *mssa, int start, int end);

    /**
     * @brief Free MSSA context.
     */
    void mssa_opt_free(MSSA_Opt *mssa);

    // ============================================================================
    // Convenience Functions
    // ============================================================================

    /**
     * @brief Reconstruct trend (component 0 only).
     */
    int ssa_opt_get_trend(const SSA_Opt *ssa, double *output);

    /**
     * @brief Reconstruct noise (components noise_start to end).
     */
    int ssa_opt_get_noise(const SSA_Opt *ssa, int noise_start, double *output);

    /**
     * @brief Get explained variance ratio for component range.
     *
     * @param ssa    Decomposed SSA context
     * @param start  First component index (inclusive)
     * @param end    Last component index (inclusive), or -1 for all remaining
     * @return       Fraction of total variance explained (0 to 1)
     */
    double ssa_opt_variance_explained(const SSA_Opt *ssa, int start, int end);

    // ============================================================================
    // Implementation
    // ============================================================================

#ifdef SSA_OPT_IMPLEMENTATION

    // ----------------------------------------------------------------------------
    // Internal Helpers
    // ----------------------------------------------------------------------------

    static inline int ssa_opt_next_pow2(int n)
    {
        int p = 1;
        while (p < n)
            p <<= 1;
        return p;
    }

    static inline int ssa_opt_min(int a, int b) { return a < b ? a : b; }
    static inline int ssa_opt_max(int a, int b) { return a > b ? a : b; }

    static inline void *ssa_opt_alloc(size_t size)
    {
        return mkl_malloc(size, SSA_ALIGN);
    }

    static inline void ssa_opt_free_ptr(void *ptr)
    {
        mkl_free(ptr);
    }

    // ----------------------------------------------------------------------------
    // BLAS Wrappers
    // ----------------------------------------------------------------------------

    static inline double ssa_opt_dot(const double *a, const double *b, int n)
    {
        return cblas_ddot(n, a, 1, b, 1);
    }

    static inline double ssa_opt_nrm2(const double *v, int n)
    {
        return cblas_dnrm2(n, v, 1);
    }

    static inline void ssa_opt_scal(double *v, int n, double s)
    {
        cblas_dscal(n, s, v, 1);
    }

    static inline void ssa_opt_axpy(double *y, const double *x, double a, int n)
    {
        cblas_daxpy(n, a, x, 1, y, 1);
    }

    static inline void ssa_opt_copy(const double *src, double *dst, int n)
    {
        cblas_dcopy(n, src, 1, dst, 1);
    }

    static inline double ssa_opt_normalize(double *v, int n)
    {
        double norm = cblas_dnrm2(n, v, 1);
        if (norm > 1e-12) // More stable for large vectors
        {
            cblas_dscal(n, 1.0 / norm, v, 1);
        }
        return norm;
    }

    static inline void ssa_opt_zero(double *v, int n)
    {
        memset(v, 0, n * sizeof(double));
    }

    // ----------------------------------------------------------------------------
    // R2C Complex Multiply Helper
    //
    // For R2C output, we have r2c_len = fft_len/2+1 complex values.
    // This is HALF the complex multiplies compared to C2C!
    // ----------------------------------------------------------------------------

    static inline void ssa_opt_complex_mul_r2c(const double *a, const double *b,
                                               double *c, int r2c_len)
    {
        vzMul(r2c_len, (const MKL_Complex16 *)a, (const MKL_Complex16 *)b,
              (MKL_Complex16 *)c);
    }

    // ============================================================================
    // Reverse copy helper - copies src[n-1..0] to dst[0..n-1]
    // ============================================================================
    static inline void ssa_opt_reverse_copy(const double *src, double *dst, int n)
    {
        // Simple scalar loop - reliable across all platforms
        for (int i = 0; i < n; i++)
        {
            dst[i] = src[n - 1 - i];
        }
    }

    // ----------------------------------------------------------------------------
    // Hankel Matrix-Vector Products via R2C FFT Convolution
    //
    // KEY OPTIMIZATION: R2C FFT exploits conjugate symmetry of real signals.
    // For real input of length N:
    //   - FFT output is only N/2+1 complex values (not N)
    //   - Complex multiply is HALF the work
    //   - C2R IFFT outputs real directly (no stride-2 extraction!)
    // ----------------------------------------------------------------------------

    /**
     * @brief Compute y = H @ v via R2C FFT convolution.
     *
     * R2C Algorithm:
     *   1. Reverse v, zero-pad to fft_len (real array, no interleaving!)
     *   2. R2C FFT → r2c_len complex values (half of C2C)
     *   3. Pointwise multiply with precomputed FFT(x) (half the work!)
     *   4. C2R IFFT → fft_len real values (direct real output!)
     *   5. Extract y = result[K-1 : K-1+L] via contiguous memcpy
     */
    static void ssa_opt_hankel_matvec(SSA_Opt *ssa, const double *v, double *y)
    {
        int K = ssa->K;
        int L = ssa->L;
        int fft_len = ssa->fft_len;
        int r2c_len = ssa->r2c_len;

        double *ws_real = ssa->ws_real;
        double *ws_complex = ssa->ws_complex;

        // Pack reversed v using SIMD-optimized BLAS copy with negative stride
        ssa_opt_zero(ws_real, fft_len);
        ssa_opt_reverse_copy(v, ws_real, K);

        // R2C forward FFT: real → complex (only r2c_len output values)
        DftiComputeForward(ssa->fft_r2c, ws_real, ws_complex);

        // Complex multiply - HALF THE WORK compared to C2C!
        ssa_opt_complex_mul_r2c(ssa->fft_x, ws_complex, ws_complex, r2c_len);

        // C2R inverse FFT: complex → real (direct real output!)
        DftiComputeBackward(ssa->fft_c2r, ws_complex, ws_real);

        // Extract result - CONTIGUOUS MEMCPY instead of stride-2 cblas_dcopy!
        memcpy(y, ws_real + (K - 1), L * sizeof(double));
    }

    /**
     * @brief Compute y = Hᵀ @ u via R2C FFT convolution.
     */
    static void ssa_opt_hankel_matvec_T(SSA_Opt *ssa, const double *u, double *y)
    {
        int K = ssa->K;
        int L = ssa->L;
        int fft_len = ssa->fft_len;
        int r2c_len = ssa->r2c_len;

        double *ws_real = ssa->ws_real;
        double *ws_complex = ssa->ws_complex;

        // Pack reversed u using SIMD-optimized BLAS copy
        ssa_opt_zero(ws_real, fft_len);
        ssa_opt_reverse_copy(u, ws_real, L);

        // R2C forward FFT
        DftiComputeForward(ssa->fft_r2c, ws_real, ws_complex);

        // Complex multiply (half the work!)
        ssa_opt_complex_mul_r2c(ssa->fft_x, ws_complex, ws_complex, r2c_len);

        // C2R inverse FFT (direct real output)
        DftiComputeBackward(ssa->fft_c2r, ws_complex, ws_real);

        // Extract result via contiguous memcpy
        memcpy(y, ws_real + (L - 1), K * sizeof(double));
    }

    // ----------------------------------------------------------------------------
    // Batched Hankel Matrix-Vector Products (R2C Version)
    // ----------------------------------------------------------------------------

    static void ssa_opt_hankel_matvec_block(SSA_Opt *ssa, const double *V_block,
                                            double *Y_block, int b)
    {
        int K = ssa->K;
        int L = ssa->L;
        int fft_len = ssa->fft_len;
        int r2c_len = ssa->r2c_len;

        double *ws_real = ssa->ws_batch_real;
        double *ws_complex = ssa->ws_batch_complex;

        const int BATCH_THRESHOLD = 2; // Below 2, batching overhead hurts performance

        int col = 0;
        while (col < b)
        {
            int batch_count = ssa_opt_min(SSA_BATCH_SIZE, b - col);

            // Fall back to sequential for small batches
            if (batch_count < BATCH_THRESHOLD)
            {
                for (int i = 0; i < batch_count; i++)
                {
                    ssa_opt_hankel_matvec(ssa, &V_block[(col + i) * K], &Y_block[(col + i) * L]);
                }
                col += batch_count;
                continue;
            }

            // Zero real workspace and pack reversed vectors
            memset(ws_real, 0, SSA_BATCH_SIZE * fft_len * sizeof(double));

            for (int i = 0; i < batch_count; i++)
            {
                const double *v = &V_block[(col + i) * K];
                double *dst = ws_real + i * fft_len;
                for (int j = 0; j < K; j++)
                {
                    dst[j] = v[K - 1 - j];
                }
            }

            // Batched R2C forward FFT
            DftiComputeForward(ssa->fft_r2c_batch, ws_real, ws_complex);

            // Complex multiply for each vector in batch (half the work per vector!)
            for (int i = 0; i < batch_count; i++)
            {
                double *fft_v = ws_complex + i * 2 * r2c_len;
                ssa_opt_complex_mul_r2c(ssa->fft_x, fft_v, fft_v, r2c_len);
            }

            // Batched C2R inverse FFT
            DftiComputeBackward(ssa->fft_c2r_batch, ws_complex, ws_real);

            // Extract results - contiguous memcpy instead of stride-2!
            for (int i = 0; i < batch_count; i++)
            {
                double *conv = ws_real + i * fft_len;
                memcpy(&Y_block[(col + i) * L], conv + (K - 1), L * sizeof(double));
            }

            col += batch_count;
        }
    }

    static void ssa_opt_hankel_matvec_T_block(SSA_Opt *ssa, const double *U_block,
                                              double *Y_block, int b)
    {
        int K = ssa->K;
        int L = ssa->L;
        int fft_len = ssa->fft_len;
        int r2c_len = ssa->r2c_len;

        double *ws_real = ssa->ws_batch_real;
        double *ws_complex = ssa->ws_batch_complex;

        const int BATCH_THRESHOLD = 2; // Below 2, batching overhead hurts performance

        int col = 0;
        while (col < b)
        {
            int batch_count = ssa_opt_min(SSA_BATCH_SIZE, b - col);

            if (batch_count < BATCH_THRESHOLD)
            {
                for (int i = 0; i < batch_count; i++)
                {
                    ssa_opt_hankel_matvec_T(ssa, &U_block[(col + i) * L], &Y_block[(col + i) * K]);
                }
                col += batch_count;
                continue;
            }

            memset(ws_real, 0, SSA_BATCH_SIZE * fft_len * sizeof(double));

            for (int i = 0; i < batch_count; i++)
            {
                const double *u = &U_block[(col + i) * L];
                double *dst = ws_real + i * fft_len;
                for (int j = 0; j < L; j++)
                {
                    dst[j] = u[L - 1 - j];
                }
            }

            DftiComputeForward(ssa->fft_r2c_batch, ws_real, ws_complex);

            for (int i = 0; i < batch_count; i++)
            {
                double *fft_u = ws_complex + i * 2 * r2c_len;
                ssa_opt_complex_mul_r2c(ssa->fft_x, fft_u, fft_u, r2c_len);
            }

            DftiComputeBackward(ssa->fft_c2r_batch, ws_complex, ws_real);

            for (int i = 0; i < batch_count; i++)
            {
                double *conv = ws_real + i * fft_len;
                memcpy(&Y_block[(col + i) * K], conv + (L - 1), K * sizeof(double));
            }

            col += batch_count;
        }
    }

    // ============================================================================
    // INITIALIZATION (R2C Version)
    // ============================================================================

    int ssa_opt_init(SSA_Opt *ssa, const double *x, int N, int L)
    {
        if (!ssa || !x || N < 4 || L < 2 || L > N - 1)
        {
            return -1;
        }

        memset(ssa, 0, sizeof(SSA_Opt));

        ssa->N = N;
        ssa->L = L;
        ssa->K = N - L + 1;

        int conv_len = N + ssa->K - 1;
        int fft_n = ssa_opt_next_pow2(conv_len);
        ssa->fft_len = fft_n;

        // R2C output length: fft_len/2 + 1 complex values (KEY TO R2C SAVINGS!)
        ssa->r2c_len = fft_n / 2 + 1;

        // -------------------------------------------------------------------------
        // Allocate workspace buffers (R2C sized - ~50% less than C2C!)
        // -------------------------------------------------------------------------
        ssa->ws_real = (double *)ssa_opt_alloc(fft_n * sizeof(double));
        ssa->ws_complex = (double *)ssa_opt_alloc(2 * ssa->r2c_len * sizeof(double));
        ssa->ws_real2 = (double *)ssa_opt_alloc(fft_n * sizeof(double));

        size_t batch_real_size = SSA_BATCH_SIZE * fft_n * sizeof(double);
        size_t batch_complex_size = SSA_BATCH_SIZE * 2 * ssa->r2c_len * sizeof(double);

        ssa->ws_batch_real = (double *)ssa_opt_alloc(batch_real_size);
        ssa->ws_batch_complex = (double *)ssa_opt_alloc(batch_complex_size);

        // Precomputed FFT(x) - now only r2c_len complex values (50% less!)
        ssa->fft_x = (double *)ssa_opt_alloc(2 * ssa->r2c_len * sizeof(double));

        if (!ssa->ws_real || !ssa->ws_complex || !ssa->ws_real2 ||
            !ssa->ws_batch_real || !ssa->ws_batch_complex || !ssa->fft_x)
        {
            ssa_opt_free(ssa);
            return -1;
        }

        // -------------------------------------------------------------------------
        // Create MKL R2C/C2R FFT Descriptors
        // -------------------------------------------------------------------------
        MKL_LONG status;

        // Single R2C FFT (forward: real → complex)
        status = DftiCreateDescriptor(&ssa->fft_r2c, DFTI_DOUBLE, DFTI_REAL, 1, fft_n);
        if (status != 0)
        {
            ssa_opt_free(ssa);
            return -1;
        }
        DftiSetValue(ssa->fft_r2c, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
        DftiSetValue(ssa->fft_r2c, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
        status = DftiCommitDescriptor(ssa->fft_r2c);
        if (status != 0)
        {
            ssa_opt_free(ssa);
            return -1;
        }

        // Single C2R FFT (backward: complex → real)
        status = DftiCreateDescriptor(&ssa->fft_c2r, DFTI_DOUBLE, DFTI_REAL, 1, fft_n);
        if (status != 0)
        {
            ssa_opt_free(ssa);
            return -1;
        }
        DftiSetValue(ssa->fft_c2r, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
        DftiSetValue(ssa->fft_c2r, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
        DftiSetValue(ssa->fft_c2r, DFTI_BACKWARD_SCALE, 1.0 / fft_n);
        status = DftiCommitDescriptor(ssa->fft_c2r);
        if (status != 0)
        {
            ssa_opt_free(ssa);
            return -1;
        }

        // Batched R2C FFT (forward)
        status = DftiCreateDescriptor(&ssa->fft_r2c_batch, DFTI_DOUBLE, DFTI_REAL, 1, fft_n);
        if (status != 0)
        {
            ssa_opt_free(ssa);
            return -1;
        }
        DftiSetValue(ssa->fft_r2c_batch, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
        DftiSetValue(ssa->fft_r2c_batch, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
        DftiSetValue(ssa->fft_r2c_batch, DFTI_NUMBER_OF_TRANSFORMS, (MKL_LONG)SSA_BATCH_SIZE);
        DftiSetValue(ssa->fft_r2c_batch, DFTI_INPUT_DISTANCE, (MKL_LONG)fft_n);
        DftiSetValue(ssa->fft_r2c_batch, DFTI_OUTPUT_DISTANCE, (MKL_LONG)ssa->r2c_len);
        status = DftiCommitDescriptor(ssa->fft_r2c_batch);
        if (status != 0)
        {
            ssa_opt_free(ssa);
            return -1;
        }

        // Batched C2R FFT (backward)
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
        DftiSetValue(ssa->fft_c2r_batch, DFTI_INPUT_DISTANCE, (MKL_LONG)ssa->r2c_len);
        DftiSetValue(ssa->fft_c2r_batch, DFTI_OUTPUT_DISTANCE, (MKL_LONG)fft_n);
        status = DftiCommitDescriptor(ssa->fft_c2r_batch);
        if (status != 0)
        {
            ssa_opt_free(ssa);
            return -1;
        }

        // RNG
        status = vslNewStream(&ssa->rng, VSL_BRNG_MT2203, 42);
        if (status != VSL_STATUS_OK)
        {
            ssa_opt_free(ssa);
            return -1;
        }

        // Precompute FFT(x) using R2C
        ssa_opt_zero(ssa->ws_real, fft_n);
        memcpy(ssa->ws_real, x, N * sizeof(double));
        DftiComputeForward(ssa->fft_r2c, ssa->ws_real, ssa->fft_x);

        // Precompute inverse diagonal counts
        ssa->inv_diag_count = (double *)ssa_opt_alloc(N * sizeof(double));
        if (!ssa->inv_diag_count)
        {
            ssa_opt_free(ssa);
            return -1;
        }
        for (int t = 0; t < N; t++)
        {
            int count = ssa_opt_min(ssa_opt_min(t + 1, L), ssa_opt_min(ssa->K, N - t));
            ssa->inv_diag_count[t] = (count > 0) ? 1.0 / count : 0.0;
        }

        ssa->initialized = true;
        return 0;
    }

    // ============================================================================
    // SEQUENTIAL POWER ITERATION DECOMPOSITION
    // ============================================================================

    int ssa_opt_decompose(SSA_Opt *ssa, int k, int max_iter)
    {
        if (!ssa || !ssa->initialized || k < 1)
            return -1;

        ssa_opt_free_cached_ffts(ssa);

        int L = ssa->L;
        int K = ssa->K;

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
                cblas_dgemv(CblasColMajor, CblasTrans, K, comp,
                            1.0, ssa->V, K, v, 1, 0.0, ssa->ws_proj, 1);
                cblas_dgemv(CblasColMajor, CblasNoTrans, K, comp,
                            -1.0, ssa->V, K, ssa->ws_proj, 1, 1.0, v, 1);
            }
            ssa_opt_normalize(v, K);

            for (int iter = 0; iter < max_iter; iter++)
            {
                ssa_opt_hankel_matvec(ssa, v, u);

                if (comp > 0)
                {
                    cblas_dgemv(CblasColMajor, CblasTrans, L, comp,
                                1.0, ssa->U, L, u, 1, 0.0, ssa->ws_proj, 1);
                    cblas_dgemv(CblasColMajor, CblasNoTrans, L, comp,
                                -1.0, ssa->U, L, ssa->ws_proj, 1, 1.0, u, 1);
                }

                ssa_opt_hankel_matvec_T(ssa, u, v_new);

                if (comp > 0)
                {
                    cblas_dgemv(CblasColMajor, CblasTrans, K, comp,
                                1.0, ssa->V, K, v_new, 1, 0.0, ssa->ws_proj, 1);
                    cblas_dgemv(CblasColMajor, CblasNoTrans, K, comp,
                                -1.0, ssa->V, K, ssa->ws_proj, 1, 1.0, v_new, 1);
                }

                ssa_opt_normalize(v_new, K);

                // Convergence check: eigenvectors can flip sign between iterations
                // Check both ||v - v_new|| and ||v + v_new||, take the minimum
                double diff_same = 0.0, diff_flip = 0.0;
                for (int i = 0; i < K; i++)
                {
                    double d_same = v[i] - v_new[i];
                    double d_flip = v[i] + v_new[i];
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
                cblas_dgemv(CblasColMajor, CblasTrans, K, comp,
                            1.0, ssa->V, K, v, 1, 0.0, ssa->ws_proj, 1);
                cblas_dgemv(CblasColMajor, CblasNoTrans, K, comp,
                            -1.0, ssa->V, K, ssa->ws_proj, 1, 1.0, v, 1);
            }
            ssa_opt_normalize(v, K);

            ssa_opt_hankel_matvec(ssa, v, u);

            if (comp > 0)
            {
                cblas_dgemv(CblasColMajor, CblasTrans, L, comp,
                            1.0, ssa->U, L, u, 1, 0.0, ssa->ws_proj, 1);
                cblas_dgemv(CblasColMajor, CblasNoTrans, L, comp,
                            -1.0, ssa->U, L, ssa->ws_proj, 1, 1.0, u, 1);
            }

            double sigma = ssa_opt_normalize(u, L);

            ssa_opt_hankel_matvec_T(ssa, u, v);

            if (comp > 0)
            {
                cblas_dgemv(CblasColMajor, CblasTrans, K, comp,
                            1.0, ssa->V, K, v, 1, 0.0, ssa->ws_proj, 1);
                cblas_dgemv(CblasColMajor, CblasNoTrans, K, comp,
                            -1.0, ssa->V, K, ssa->ws_proj, 1, 1.0, v, 1);
            }

            if (sigma > 1e-12)
                ssa_opt_scal(v, K, 1.0 / sigma);

            ssa_opt_copy(u, &ssa->U[comp * L], L);
            ssa_opt_copy(v, &ssa->V[comp * K], K);
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
                    double tmp = ssa->sigma[i];
                    ssa->sigma[i] = ssa->sigma[j];
                    ssa->sigma[j] = tmp;

                    tmp = ssa->eigenvalues[i];
                    ssa->eigenvalues[i] = ssa->eigenvalues[j];
                    ssa->eigenvalues[j] = tmp;

                    cblas_dswap(L, &ssa->U[i * L], 1, &ssa->U[j * L], 1);
                    cblas_dswap(K, &ssa->V[i * K], 1, &ssa->V[j * K], 1);
                }
            }
        }

        // Fix sign convention
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

    // ============================================================================
    // BLOCK POWER METHOD DECOMPOSITION (R2C Version)
    // ============================================================================

    int ssa_opt_decompose_block(SSA_Opt *ssa, int k, int block_size, int max_iter)
    {
        if (!ssa || !ssa->initialized || k < 1 || block_size < 1)
            return -1;

        ssa_opt_free_cached_ffts(ssa);

        int L = ssa->L;
        int K = ssa->K;

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

        if (!V_block || !U_block || !U_block2 || !tau_u || !tau_v ||
            !M || !U_small || !Vt_small || !S_small || !superb || !work)
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
                cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                            comp, cur_b, K, 1.0, ssa->V, K, V_block, K, 0.0, work, comp);
                cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                            K, cur_b, comp, -1.0, ssa->V, K, work, comp, 1.0, V_block, K);
            }

            LAPACKE_dgeqrf(LAPACK_COL_MAJOR, K, cur_b, V_block, K, tau_v);
            LAPACKE_dorgqr(LAPACK_COL_MAJOR, K, cur_b, cur_b, V_block, K, tau_v);

            const int QR_INTERVAL = 5;

            for (int iter = 0; iter < max_iter; iter++)
            {
                ssa_opt_hankel_matvec_block(ssa, V_block, U_block, cur_b);

                if (comp > 0)
                {
                    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                                comp, cur_b, L, 1.0, ssa->U, L, U_block, L, 0.0, work, comp);
                    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                L, cur_b, comp, -1.0, ssa->U, L, work, comp, 1.0, U_block, L);
                }

                if ((iter % QR_INTERVAL == 0) || (iter == max_iter - 1))
                {
                    LAPACKE_dgeqrf(LAPACK_COL_MAJOR, L, cur_b, U_block, L, tau_u);
                    LAPACKE_dorgqr(LAPACK_COL_MAJOR, L, cur_b, cur_b, U_block, L, tau_u);
                }

                ssa_opt_hankel_matvec_T_block(ssa, U_block, V_block, cur_b);

                if (comp > 0)
                {
                    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                                comp, cur_b, K, 1.0, ssa->V, K, V_block, K, 0.0, work, comp);
                    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                K, cur_b, comp, -1.0, ssa->V, K, work, comp, 1.0, V_block, K);
                }

                if ((iter > 0 && iter % QR_INTERVAL == 0) || (iter == max_iter - 1))
                {
                    LAPACKE_dgeqrf(LAPACK_COL_MAJOR, K, cur_b, V_block, K, tau_v);
                    LAPACKE_dorgqr(LAPACK_COL_MAJOR, K, cur_b, cur_b, V_block, K, tau_v);
                }
            }

            // Rayleigh-Ritz refinement
            ssa_opt_hankel_matvec_block(ssa, V_block, U_block2, cur_b);

            if (comp > 0)
            {
                cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                            comp, cur_b, L, 1.0, ssa->U, L, U_block2, L, 0.0, work, comp);
                cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                            L, cur_b, comp, -1.0, ssa->U, L, work, comp, 1.0, U_block2, L);
            }

            cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                        cur_b, cur_b, L, 1.0, U_block, L, U_block2, L, 0.0, M, cur_b);

            int svd_info = LAPACKE_dgesvd(LAPACK_COL_MAJOR, 'A', 'A',
                                          cur_b, cur_b, M, cur_b,
                                          S_small, U_small, cur_b, Vt_small, cur_b, superb);
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

            cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                        L, cur_b, cur_b, 1.0, U_block, L, U_small, cur_b, 0.0, U_block2, L);

            cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                        K, cur_b, cur_b, 1.0, V_block, K, Vt_small, cur_b, 0.0, work, K);

            for (int i = 0; i < cur_b; i++)
            {
                double sigma = S_small[i];
                ssa->sigma[comp + i] = sigma;
                ssa->eigenvalues[comp + i] = sigma * sigma;
                ssa->total_variance += sigma * sigma;

                cblas_dcopy(L, &U_block2[i * L], 1, &ssa->U[(comp + i) * L], 1);
                cblas_dcopy(K, &work[i * K], 1, &ssa->V[(comp + i) * K], 1);
            }

            // Final V refinement
            for (int i = 0; i < cur_b; i++)
                cblas_dcopy(L, &ssa->U[(comp + i) * L], 1, &U_block[i * L], 1);

            ssa_opt_hankel_matvec_T_block(ssa, U_block, V_block, cur_b);

            if (comp > 0)
            {
                cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                            comp, cur_b, K, 1.0, ssa->V, K, V_block, K, 0.0, work, comp);
                cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                            K, cur_b, comp, -1.0, ssa->V, K, work, comp, 1.0, V_block, K);
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

        // Sort by descending singular value
        for (int i = 0; i < k - 1; i++)
        {
            for (int j = i + 1; j < k; j++)
            {
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
            }
        }

        // Fix sign convention
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

    // ============================================================================
    // RANDOMIZED SVD DECOMPOSITION (R2C Version)
    // ============================================================================

    int ssa_opt_decompose_randomized(SSA_Opt *ssa, int k, int oversampling)
    {
        if (!ssa || !ssa->initialized || k < 1)
            return -1;

        ssa_opt_free_cached_ffts(ssa);

        int L = ssa->L;
        int K = ssa->K;

        int p = (oversampling <= 0) ? 8 : oversampling;
        int kp = k + p;

        kp = ssa_opt_min(kp, ssa_opt_min(L, K));
        k = ssa_opt_min(k, kp);

        ssa->U = (double *)ssa_opt_alloc(L * k * sizeof(double));
        ssa->V = (double *)ssa_opt_alloc(K * k * sizeof(double));
        ssa->sigma = (double *)ssa_opt_alloc(k * sizeof(double));
        ssa->eigenvalues = (double *)ssa_opt_alloc(k * sizeof(double));

        if (!ssa->U || !ssa->V || !ssa->sigma || !ssa->eigenvalues)
            return -1;

        ssa->n_components = k;
        ssa->total_variance = 0.0;

        double *Omega = (double *)ssa_opt_alloc(K * kp * sizeof(double));
        double *Y = (double *)ssa_opt_alloc(L * kp * sizeof(double));
        double *Q = (double *)ssa_opt_alloc(L * kp * sizeof(double));
        double *B = (double *)ssa_opt_alloc(K * kp * sizeof(double));
        double *tau = (double *)ssa_opt_alloc(kp * sizeof(double));

        // =========================================================================
        // SVD workspace allocation
        //
        // For B = K × kp, dgesdd('S') computes B = B_left × S × B_right_T where:
        //   B_left:    K × kp   (left singular vectors of B)
        //   B_right_T: kp × kp  (right singular vectors of B, stored as Vᵀ)
        //
        // The relationship to H's SVD is:
        //   H ≈ Q @ B_right_Tᵀ × S × B_leftᵀ
        // So: U_H = Q @ B_right_Tᵀ  and  V_H = B_left
        // =========================================================================
        double *B_left = (double *)ssa_opt_alloc(K * kp * sizeof(double));     // Left singular vectors of B → becomes V of H
        double *B_right_T = (double *)ssa_opt_alloc(kp * kp * sizeof(double)); // Vᵀ of B → used to compute U of H
        double *S_svd = (double *)ssa_opt_alloc(kp * sizeof(double));

        double work_query;
        int *iwork = (int *)ssa_opt_alloc(8 * kp * sizeof(int));
        int lwork = -1;
        int info;

        LAPACKE_dgesdd_work(LAPACK_COL_MAJOR, 'S', K, kp, B, K, S_svd,
                            B_left, K, B_right_T, kp, &work_query, lwork, iwork);
        lwork = (int)work_query + 1;
        double *work = (double *)ssa_opt_alloc(lwork * sizeof(double));

        if (!Omega || !Y || !Q || !B || !tau || !B_left || !B_right_T || !S_svd || !iwork || !work)
        {
            ssa_opt_free_ptr(Omega);
            ssa_opt_free_ptr(Y);
            ssa_opt_free_ptr(Q);
            ssa_opt_free_ptr(B);
            ssa_opt_free_ptr(tau);
            ssa_opt_free_ptr(B_left);
            ssa_opt_free_ptr(B_right_T);
            ssa_opt_free_ptr(S_svd);
            ssa_opt_free_ptr(iwork);
            ssa_opt_free_ptr(work);
            return -1;
        }

        // Step 1: Random Gaussian test matrix
        vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER, ssa->rng, K * kp, Omega, 0.0, 1.0);

        // Step 2: Range sampling Y = H @ Ω (batched R2C FFT)
        ssa_opt_hankel_matvec_block(ssa, Omega, Y, kp);

        // Step 3: QR factorization
        cblas_dcopy(L * kp, Y, 1, Q, 1);
        LAPACKE_dgeqrf(LAPACK_COL_MAJOR, L, kp, Q, L, tau);
        LAPACKE_dorgqr(LAPACK_COL_MAJOR, L, kp, kp, Q, L, tau);

        // Step 4: B = H^T @ Q (batched R2C FFT)
        ssa_opt_hankel_matvec_T_block(ssa, Q, B, kp);

        // Step 5: SVD of B → B = B_left × S × B_right_T
        info = LAPACKE_dgesdd_work(LAPACK_COL_MAJOR, 'S', K, kp, B, K, S_svd,
                                   B_left, K, B_right_T, kp, work, lwork, iwork);

        if (info != 0)
        {
            ssa_opt_free_ptr(Omega);
            ssa_opt_free_ptr(Y);
            ssa_opt_free_ptr(Q);
            ssa_opt_free_ptr(B);
            ssa_opt_free_ptr(tau);
            ssa_opt_free_ptr(B_left);
            ssa_opt_free_ptr(B_right_T);
            ssa_opt_free_ptr(S_svd);
            ssa_opt_free_ptr(iwork);
            ssa_opt_free_ptr(work);
            return -1;
        }

        // Step 6: Recover U = Q @ B_right_Tᵀ, V = B_left
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                    L, k, kp, 1.0, Q, L, B_right_T, kp, 0.0, ssa->U, L);

        for (int i = 0; i < k; i++)
            cblas_dcopy(K, &B_left[i * K], 1, &ssa->V[i * K], 1);

        for (int i = 0; i < k; i++)
        {
            ssa->sigma[i] = S_svd[i];
            ssa->eigenvalues[i] = S_svd[i] * S_svd[i];
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
                ssa_opt_scal(&ssa->U[i * L], L, -1.0);
                ssa_opt_scal(&ssa->V[i * K], K, -1.0);
            }
        }

        ssa_opt_free_ptr(Omega);
        ssa_opt_free_ptr(Y);
        ssa_opt_free_ptr(Q);
        ssa_opt_free_ptr(B);
        ssa_opt_free_ptr(tau);
        ssa_opt_free_ptr(B_left);
        ssa_opt_free_ptr(B_right_T);
        ssa_opt_free_ptr(S_svd);
        ssa_opt_free_ptr(iwork);
        ssa_opt_free_ptr(work);

        ssa->decomposed = true;
        return 0;
    }

    // ============================================================================
    // CACHE FFTs for fast reconstruction (R2C Version)
    // ============================================================================

    void ssa_opt_free_cached_ffts(SSA_Opt *ssa)
    {
        if (!ssa)
            return;
        ssa_opt_free_ptr(ssa->U_fft);
        ssa_opt_free_ptr(ssa->V_fft);
        ssa->U_fft = NULL;
        ssa->V_fft = NULL;
        ssa->fft_cached = false;
    }

    int ssa_opt_cache_ffts(SSA_Opt *ssa)
    {
        if (!ssa || !ssa->decomposed)
            return -1;

        ssa_opt_free_cached_ffts(ssa);

        int L = ssa->L;
        int K = ssa->K;
        int fft_len = ssa->fft_len;
        int r2c_len = ssa->r2c_len;
        int k = ssa->n_components;

        // R2C cache: r2c_len complex values per component (50% less than C2C!)
        size_t cache_size = 2 * r2c_len * k * sizeof(double);

        ssa->U_fft = (double *)ssa_opt_alloc(cache_size);
        ssa->V_fft = (double *)ssa_opt_alloc(cache_size);

        if (!ssa->U_fft || !ssa->V_fft)
        {
            ssa_opt_free_cached_ffts(ssa);
            return -1;
        }

        for (int i = 0; i < k; i++)
        {
            double sigma = ssa->sigma[i];
            const double *u_vec = &ssa->U[i * L];
            double *dst = &ssa->U_fft[i * 2 * r2c_len];

            ssa_opt_zero(ssa->ws_real, fft_len);
            for (int j = 0; j < L; j++)
                ssa->ws_real[j] = sigma * u_vec[j];

            DftiComputeForward(ssa->fft_r2c, ssa->ws_real, dst);
        }

        for (int i = 0; i < k; i++)
        {
            const double *v_vec = &ssa->V[i * K];
            double *dst = &ssa->V_fft[i * 2 * r2c_len];

            ssa_opt_zero(ssa->ws_real, fft_len);
            for (int j = 0; j < K; j++)
                ssa->ws_real[j] = v_vec[j];

            DftiComputeForward(ssa->fft_r2c, ssa->ws_real, dst);
        }

        ssa->fft_cached = true;
        return 0;
    }

    int ssa_opt_reconstruct(const SSA_Opt *ssa, const int *group, int n_group, double *output)
    {
        if (!ssa || !ssa->decomposed || !group || !output || n_group < 1)
            return -1;

        int N = ssa->N;
        int L = ssa->L;
        int K = ssa->K;
        int fft_len = ssa->fft_len;
        int r2c_len = ssa->r2c_len;

        SSA_Opt *ssa_mut = (SSA_Opt *)ssa;

        // =========================================================================
        // FREQUENCY-DOMAIN ACCUMULATION OPTIMIZATION
        //
        // Instead of: for each component { complex_mul → IFFT → accumulate real }
        // We do:      for each component { complex_mul → accumulate complex }
        //             single IFFT at end
        //
        // This reduces n_group IFFTs to just 1 IFFT!
        // =========================================================================

        // Use ws_batch_complex as frequency accumulator (zero it first)
        double *freq_accum = ssa_mut->ws_batch_complex;
        ssa_opt_zero(freq_accum, 2 * r2c_len);

        if (ssa->fft_cached && ssa->U_fft && ssa->V_fft)
        {
            // FAST PATH: Use precomputed R2C FFTs
            for (int g = 0; g < n_group; g++)
            {
                int idx = group[g];
                if (idx < 0 || idx >= ssa->n_components)
                    continue;

                const double *u_fft_cached = &ssa->U_fft[idx * 2 * r2c_len];
                const double *v_fft_cached = &ssa->V_fft[idx * 2 * r2c_len];

                // Complex multiply into temp buffer
                ssa_opt_complex_mul_r2c(u_fft_cached, v_fft_cached,
                                        ssa_mut->ws_complex, r2c_len);

                // Accumulate in frequency domain: freq_accum += ws_complex
                cblas_daxpy(2 * r2c_len, 1.0, ssa_mut->ws_complex, 1, freq_accum, 1);
            }
        }
        else
        {
            // STANDARD PATH: Compute FFTs on-the-fly
            // Use second slot of ws_batch_complex for temp FFT output
            double *temp_fft2 = ssa_mut->ws_batch_complex + 2 * r2c_len;

            for (int g = 0; g < n_group; g++)
            {
                int idx = group[g];
                if (idx < 0 || idx >= ssa->n_components)
                    continue;

                double sigma = ssa->sigma[idx];
                const double *u_vec = &ssa->U[idx * L];
                const double *v_vec = &ssa->V[idx * K];

                // FFT(σ × u) → ws_complex
                ssa_opt_zero(ssa_mut->ws_real, fft_len);
                for (int i = 0; i < L; i++)
                    ssa_mut->ws_real[i] = sigma * u_vec[i];
                DftiComputeForward(ssa_mut->fft_r2c, ssa_mut->ws_real, ssa_mut->ws_complex);

                // FFT(v) → temp_fft2
                ssa_opt_zero(ssa_mut->ws_real2, fft_len);
                for (int i = 0; i < K; i++)
                    ssa_mut->ws_real2[i] = v_vec[i];
                DftiComputeForward(ssa_mut->fft_r2c, ssa_mut->ws_real2, temp_fft2);

                // Complex multiply: ws_complex = ws_complex ⊙ temp_fft2
                ssa_opt_complex_mul_r2c(ssa_mut->ws_complex, temp_fft2, ssa_mut->ws_complex, r2c_len);

                // Accumulate in frequency domain: freq_accum += ws_complex
                cblas_daxpy(2 * r2c_len, 1.0, ssa_mut->ws_complex, 1, freq_accum, 1);
            }
        }

        // Single IFFT at the end
        DftiComputeBackward(ssa_mut->fft_c2r, freq_accum, ssa_mut->ws_real);

        // Copy result (first N elements)
        memcpy(output, ssa_mut->ws_real, N * sizeof(double));

        // Diagonal averaging
        vdMul(N, output, ssa->inv_diag_count, output);

        return 0;
    }

    // ============================================================================
    // INCREMENTAL DECOMPOSITION (Extend) - R2C Version
    // ============================================================================

    // ============================================================================
    // EXTEND DECOMPOSITION
    //
    // Purpose: Add more singular triplets to an existing decomposition without
    // recomputing the ones we already have.
    //
    // Algorithm: Deflated power iteration
    //   - Start with random vector orthogonal to all existing V columns
    //   - Iterate: u = H·v, v = Hᵀ·u (power iteration on HᵀH)
    //   - After each step, project out existing components (deflation)
    //   - Converged when v stops changing (up to sign)
    //
    // Why this works:
    //   Power iteration on HᵀH converges to dominant eigenvector (largest σ).
    //   By projecting out existing components, we find the next largest.
    //   This is mathematically equivalent to SVD on the residual matrix.
    //
    // Complexity: O(additional_k × max_iter × N log N)
    //   - Each power iteration step requires two Hankel matvecs: O(N log N)
    //   - Plus orthogonalization against existing components: O(k × L) or O(k × K)
    //
    // ============================================================================

    int ssa_opt_extend(SSA_Opt *ssa, int additional_k, int max_iter)
    {
        if (!ssa || !ssa->decomposed || additional_k < 1)
            return -1;

        // ========================================================================
        // STEP 1: Invalidate cached FFTs
        //
        // Cached FFTs are keyed to current U/V matrices. After adding new
        // components, the cache would be stale. We must free it before modifying.
        // ========================================================================
        ssa_opt_free_cached_ffts(ssa);

        int L = ssa->L;
        int K = ssa->K;
        int old_k = ssa->n_components;
        int new_k = old_k + additional_k;

        // Maximum possible components is min(L, K) - the rank of the Hankel matrix
        new_k = ssa_opt_min(new_k, ssa_opt_min(L, K));
        if (new_k <= old_k)
            return 0; // Nothing to add

        // ========================================================================
        // STEP 2: Reallocate result arrays
        //
        // We need larger arrays to hold new_k components. Strategy:
        //   1. Allocate new arrays of size new_k
        //   2. Copy old data
        //   3. Free old arrays
        //   4. Point to new arrays
        //
        // This is safer than realloc() because it guarantees alignment.
        // ========================================================================
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

        // Copy existing components to new arrays
        memcpy(U_new, ssa->U, L * old_k * sizeof(double));
        memcpy(V_new, ssa->V, K * old_k * sizeof(double));
        memcpy(sigma_new, ssa->sigma, old_k * sizeof(double));
        memcpy(eigen_new, ssa->eigenvalues, old_k * sizeof(double));

        // Swap old arrays for new ones
        ssa_opt_free_ptr(ssa->U);
        ssa_opt_free_ptr(ssa->V);
        ssa_opt_free_ptr(ssa->sigma);
        ssa_opt_free_ptr(ssa->eigenvalues);

        ssa->U = U_new;
        ssa->V = V_new;
        ssa->sigma = sigma_new;
        ssa->eigenvalues = eigen_new;

        // ========================================================================
        // STEP 3: Reallocate projection workspace
        //
        // ws_proj stores projection coefficients during orthogonalization.
        // Size must be at least new_k to hold all inner products.
        // ========================================================================
        if (ssa->ws_proj)
        {
            ssa_opt_free_ptr(ssa->ws_proj);
        }
        ssa->ws_proj = (double *)ssa_opt_alloc(new_k * sizeof(double));
        if (!ssa->ws_proj)
            return -1;

        // Temporary vectors for power iteration
        double *u = (double *)ssa_opt_alloc(L * sizeof(double));     // Left singular vector
        double *v = (double *)ssa_opt_alloc(K * sizeof(double));     // Right singular vector
        double *v_new = (double *)ssa_opt_alloc(K * sizeof(double)); // Updated v for convergence check

        if (!u || !v || !v_new)
        {
            ssa_opt_free_ptr(u);
            ssa_opt_free_ptr(v);
            ssa_opt_free_ptr(v_new);
            return -1;
        }

        // ========================================================================
        // STEP 4: Compute additional components via deflated power iteration
        //
        // For each new component comp = old_k, old_k+1, ..., new_k-1:
        //
        //   (a) Initialize v randomly, orthogonalize against V[:, 0:comp]
        //   (b) Power iteration: u = H·v, v = Hᵀ·u, with orthogonalization
        //   (c) Extract singular triplet: σ = ||u||, normalize u and v
        //
        // The orthogonalization ensures we find components orthogonal to
        // all previously computed ones (deflation).
        // ========================================================================
        for (int comp = old_k; comp < new_k; comp++)
        {
            // --------------------------------------------------------------------
            // STEP 4a: Random initialization with orthogonalization
            //
            // Start with random vector, then project out all existing V columns:
            //   v := v - V @ (Vᵀ @ v)
            //
            // This ensures v is orthogonal to the subspace spanned by V[:, 0:comp].
            // Using BLAS dgemv:
            //   proj = Vᵀ @ v    (K×comp)ᵀ @ (K) → (comp)
            //   v := v - V @ proj   (K×comp) @ (comp) → (K), subtracted from v
            // --------------------------------------------------------------------
            vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, ssa->rng, K, v, -0.5, 0.5);

            // Compute projection coefficients: ws_proj = Vᵀ @ v
            // CblasTrans means we're computing Aᵀ @ x, where A = V (K × comp)
            cblas_dgemv(CblasColMajor, CblasTrans, K, comp,
                        1.0, ssa->V, K, v, 1, 0.0, ssa->ws_proj, 1);

            // Subtract projection: v := v - V @ ws_proj
            // CblasNoTrans: A @ x, where A = V, x = ws_proj
            // -1.0 alpha, 1.0 beta means: v := -1.0 * (V @ ws_proj) + 1.0 * v
            cblas_dgemv(CblasColMajor, CblasNoTrans, K, comp,
                        -1.0, ssa->V, K, ssa->ws_proj, 1, 1.0, v, 1);
            ssa_opt_normalize(v, K);

            // --------------------------------------------------------------------
            // STEP 4b: Power iteration
            //
            // Each iteration:
            //   1. u = H @ v            (forward Hankel matvec)
            //   2. Orthogonalize u against existing U columns
            //   3. Normalize u
            //   4. v_new = Hᵀ @ u       (transpose Hankel matvec)
            //   5. Orthogonalize v_new against existing V columns
            //   6. Normalize v_new
            //   7. Check convergence: ||v - v_new|| or ||v + v_new|| small
            //
            // The pair (u, v) converges to the (comp+1)-th singular vectors.
            // Orthogonalization after each matvec is the "deflation" step.
            // --------------------------------------------------------------------
            for (int iter = 0; iter < max_iter; iter++)
            {
                // u = H @ v (FFT-based Hankel matvec, O(N log N))
                ssa_opt_hankel_matvec(ssa, v, u);

                // Orthogonalize u against all existing U columns
                // Same pattern as above: project out U[:, 0:comp]
                cblas_dgemv(CblasColMajor, CblasTrans, L, comp,
                            1.0, ssa->U, L, u, 1, 0.0, ssa->ws_proj, 1);
                cblas_dgemv(CblasColMajor, CblasNoTrans, L, comp,
                            -1.0, ssa->U, L, ssa->ws_proj, 1, 1.0, u, 1);
                ssa_opt_normalize(u, L);

                // v_new = Hᵀ @ u (transpose Hankel matvec)
                ssa_opt_hankel_matvec_T(ssa, u, v_new);

                // Orthogonalize v_new against existing V columns
                cblas_dgemv(CblasColMajor, CblasTrans, K, comp,
                            1.0, ssa->V, K, v_new, 1, 0.0, ssa->ws_proj, 1);
                cblas_dgemv(CblasColMajor, CblasNoTrans, K, comp,
                            -1.0, ssa->V, K, ssa->ws_proj, 1, 1.0, v_new, 1);
                ssa_opt_normalize(v_new, K);

                // ----------------------------------------------------------------
                // CONVERGENCE CHECK: Handle sign flips correctly
                //
                // Eigenvectors are defined only up to sign: if v is an eigenvector,
                // so is -v. Power iteration can flip between v and -v on consecutive
                // iterations even when converged.
                //
                // Naive check ||v - v_new|| fails: if v_new ≈ -v, this gives ~2.
                //
                // Solution: Check BOTH ||v - v_new||² and ||v + v_new||², take min.
                // If either is small, we've converged.
                // ----------------------------------------------------------------
                double diff_same = 0.0, diff_flip = 0.0;
                for (int i = 0; i < K; i++)
                {
                    double d_same = v[i] - v_new[i];
                    double d_flip = v[i] + v_new[i];
                    diff_same += d_same * d_same;
                    diff_flip += d_flip * d_flip;
                }
                double diff = (diff_same < diff_flip) ? diff_same : diff_flip;
                ssa_opt_copy(v_new, v, K);

                // Require at least 10 iterations to avoid false convergence
                if (sqrt(diff) < SSA_CONVERGENCE_TOL && iter > 10)
                    break;
            }

            // --------------------------------------------------------------------
            // STEP 4c: Extract singular triplet
            //
            // After convergence, v is the right singular vector. Now compute:
            //   u = H @ v           (un-normalized)
            //   σ = ||u||           (singular value)
            //   u := u / σ          (normalized left singular vector)
            //   v = Hᵀ @ u / σ      (ensures SVD consistency: H @ v = σ @ u)
            //
            // Final orthogonalization ensures numerical orthogonality is maintained.
            // --------------------------------------------------------------------

            // Final orthogonalization of v
            cblas_dgemv(CblasColMajor, CblasTrans, K, comp,
                        1.0, ssa->V, K, v, 1, 0.0, ssa->ws_proj, 1);
            cblas_dgemv(CblasColMajor, CblasNoTrans, K, comp,
                        -1.0, ssa->V, K, ssa->ws_proj, 1, 1.0, v, 1);
            ssa_opt_normalize(v, K);

            // Compute u = H @ v and extract sigma
            ssa_opt_hankel_matvec(ssa, v, u);

            // Orthogonalize u against existing U columns
            cblas_dgemv(CblasColMajor, CblasTrans, L, comp,
                        1.0, ssa->U, L, u, 1, 0.0, ssa->ws_proj, 1);
            cblas_dgemv(CblasColMajor, CblasNoTrans, L, comp,
                        -1.0, ssa->U, L, ssa->ws_proj, 1, 1.0, u, 1);

            // σ = ||u||, then normalize u
            double sigma = ssa_opt_normalize(u, L);

            // Recompute v for SVD consistency: v = Hᵀ @ u / σ
            // This ensures H @ v = σ @ u exactly (up to numerical precision)
            ssa_opt_hankel_matvec_T(ssa, u, v);

            // Final orthogonalization of v
            cblas_dgemv(CblasColMajor, CblasTrans, K, comp,
                        1.0, ssa->V, K, v, 1, 0.0, ssa->ws_proj, 1);
            cblas_dgemv(CblasColMajor, CblasNoTrans, K, comp,
                        -1.0, ssa->V, K, ssa->ws_proj, 1, 1.0, v, 1);

            // Scale v by 1/σ to complete the SVD relationship
            if (sigma > 1e-12)
            {
                ssa_opt_scal(v, K, 1.0 / sigma);
            }

            // Store results in the arrays
            ssa_opt_copy(u, &ssa->U[comp * L], L);
            ssa_opt_copy(v, &ssa->V[comp * K], K);
            ssa->sigma[comp] = sigma;
            ssa->eigenvalues[comp] = sigma * sigma;
            ssa->total_variance += sigma * sigma;
        }

        // ========================================================================
        // STEP 5: Sort all components by descending singular value
        //
        // New components might have larger singular values than some existing ones
        // (if power iteration for existing components converged to wrong eigenpairs).
        // Re-sorting ensures σ₀ ≥ σ₁ ≥ ... ≥ σₖ₋₁.
        //
        // Using simple bubble sort since k is typically small (10-100).
        // For large k, could use qsort with index array.
        // ========================================================================
        for (int i = 0; i < new_k - 1; i++)
        {
            for (int j = i + 1; j < new_k; j++)
            {
                if (ssa->sigma[j] > ssa->sigma[i])
                {
                    // Swap singular values
                    double tmp = ssa->sigma[i];
                    ssa->sigma[i] = ssa->sigma[j];
                    ssa->sigma[j] = tmp;

                    // Swap eigenvalues
                    tmp = ssa->eigenvalues[i];
                    ssa->eigenvalues[i] = ssa->eigenvalues[j];
                    ssa->eigenvalues[j] = tmp;

                    // Swap U columns using BLAS dswap (vectorized)
                    cblas_dswap(L, &ssa->U[i * L], 1, &ssa->U[j * L], 1);

                    // Swap V columns
                    cblas_dswap(K, &ssa->V[i * K], 1, &ssa->V[j * K], 1);
                }
            }
        }

        // ========================================================================
        // STEP 6: Fix sign convention for consistent results
        //
        // SVD is unique only up to sign: (u, v) and (-u, -v) are both valid.
        // For reproducibility, we adopt the convention: sum(u) > 0.
        // If sum(u) < 0, flip both u and v.
        //
        // Note: Only need to fix new components (after sorting, these might
        // be anywhere in the array, so we check all indices >= old_k in the
        // original numbering. After sorting, we just fix all to be safe.)
        // ========================================================================
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

    // ============================================================================
    // W-CORRELATION ANALYSIS
    //
    // W-correlation measures similarity between reconstructed SSA components
    // using the diagonal averaging weights. High |ρ_W(i,j)| indicates components
    // i and j belong together (e.g., sine/cosine pairs for periodic signals).
    //
    // Formula: ρ_W(i,j) = <Xᵢ, Xⱼ>_W / (‖Xᵢ‖_W × ‖Xⱼ‖_W)
    // where the weighted inner product uses w[t] = min(t+1, L, K, N-t)
    //
    // Usage: Identify which components to group for reconstruction
    // ============================================================================

    int ssa_opt_wcorr_matrix(const SSA_Opt *ssa, double *W)
    {
        if (!ssa || !ssa->decomposed || !W)
            return -1;

        int N = ssa->N;
        int L = ssa->L;
        int K = ssa->K;
        int n = ssa->n_components;
        int fft_len = ssa->fft_len;
        int r2c_len = ssa->r2c_len;

        // =========================================================================
        // OPTIMIZED W-CORRELATION: O(n × N log N + n² × N) instead of O(n² × N log N)
        //
        // Key insight: X_i[t] = σ_i × h_i[t] / c[t]  where h_i = conv(u_i, v_i)
        //
        // <X_i, X_j>_W = σ_i × σ_j × Σ_t h_i[t] × h_j[t] / c[t]
        //
        // Define: g_i[t] = σ_i × h_i[t] / sqrt(c[t]) / ||X_i||_W
        // Then:   W[i,j] = Σ_t g_i[t] × g_j[t] = G @ Gᵀ  (BLAS dsyrk!)
        //
        // This avoids n reconstructions inside the n² loop!
        // =========================================================================

        // Cast away const for workspace access
        SSA_Opt *ssa_mut = (SSA_Opt *)ssa;

        // Step 1: Compute inverse diagonal counts: inv_c[t] = 1 / c[t]
        double *inv_c = (double *)ssa_opt_alloc(N * sizeof(double));
        double *sqrt_inv_c = (double *)ssa_opt_alloc(N * sizeof(double));
        if (!inv_c || !sqrt_inv_c)
        {
            ssa_opt_free_ptr(inv_c);
            ssa_opt_free_ptr(sqrt_inv_c);
            return -1;
        }

        for (int t = 0; t < N; t++)
        {
            int c_t = ssa_opt_min(ssa_opt_min(t + 1, L), ssa_opt_min(K, N - t));
            inv_c[t] = 1.0 / c_t;
            sqrt_inv_c[t] = sqrt(inv_c[t]);
        }

        // Step 2: Compute h_i = conv(u_i, v_i) for all components via FFT
        // and build G matrix: G[i,t] = σ_i × h_i[t] / sqrt(c[t]) / ||X_i||_W
        double *G = (double *)ssa_opt_alloc(n * N * sizeof(double));
        double *norms = (double *)ssa_opt_alloc(n * sizeof(double));
        double *h_temp = (double *)ssa_opt_alloc(fft_len * sizeof(double));
        double *u_fft = (double *)ssa_opt_alloc(2 * r2c_len * sizeof(double));
        double *v_fft = (double *)ssa_opt_alloc(2 * r2c_len * sizeof(double));

        if (!G || !norms || !h_temp || !u_fft || !v_fft)
        {
            ssa_opt_free_ptr(inv_c);
            ssa_opt_free_ptr(sqrt_inv_c);
            ssa_opt_free_ptr(G);
            ssa_opt_free_ptr(norms);
            ssa_opt_free_ptr(h_temp);
            ssa_opt_free_ptr(u_fft);
            ssa_opt_free_ptr(v_fft);
            return -1;
        }

        // Compute h_i and norms for each component
        for (int i = 0; i < n; i++)
        {
            double sigma = ssa->sigma[i];
            const double *u_vec = &ssa->U[i * L];
            const double *v_vec = &ssa->V[i * K];

            // FFT(u_i)
            ssa_opt_zero(ssa_mut->ws_real, fft_len);
            memcpy(ssa_mut->ws_real, u_vec, L * sizeof(double));
            DftiComputeForward(ssa_mut->fft_r2c, ssa_mut->ws_real, u_fft);

            // FFT(v_i)
            ssa_opt_zero(ssa_mut->ws_real, fft_len);
            memcpy(ssa_mut->ws_real, v_vec, K * sizeof(double));
            DftiComputeForward(ssa_mut->fft_r2c, ssa_mut->ws_real, v_fft);

            // h_i = IFFT(FFT(u) × FFT(v)) = conv(u, v)
            ssa_opt_complex_mul_r2c(u_fft, v_fft, ssa_mut->ws_complex, r2c_len);
            DftiComputeBackward(ssa_mut->fft_c2r, ssa_mut->ws_complex, h_temp);

            // Compute ||X_i||_W² = σ² × Σ_t h_i[t]² / c[t]
            double norm_sq = 0.0;
            for (int t = 0; t < N; t++)
            {
                norm_sq += h_temp[t] * h_temp[t] * inv_c[t];
            }
            norm_sq *= sigma * sigma;
            norms[i] = sqrt(norm_sq);

            // Store scaled h in G: G[i,t] = σ × h[t] / sqrt(c[t]) / ||X_i||_W
            double scale = (norms[i] > 1e-12) ? sigma / norms[i] : 0.0;
            double *g_row = &G[i * N];
            for (int t = 0; t < N; t++)
            {
                g_row[t] = scale * h_temp[t] * sqrt_inv_c[t];
            }
        }

        // Step 3: W = G @ Gᵀ via BLAS dsyrk (symmetric rank-k update)
        // G is stored row-major: G[i,t] at G[i*N + t], dimensions n×N
        // W[i,j] = Σ_t G[i,t] × G[j,t]
        // dsyrk with RowMajor, NoTrans computes C := α A Aᵀ + β C
        cblas_dsyrk(CblasRowMajor, CblasUpper, CblasNoTrans,
                    n, N,       // n×n result from n×N matrix
                    1.0, G, N,  // α=1, A=G, lda=N (row-major leading dim = num cols)
                    0.0, W, n); // β=0, C=W, ldc=n

        // Fill lower triangle (dsyrk only fills upper)
        for (int i = 0; i < n; i++)
        {
            for (int j = i + 1; j < n; j++)
            {
                W[j * n + i] = W[i * n + j];
            }
        }

        // Cleanup
        ssa_opt_free_ptr(inv_c);
        ssa_opt_free_ptr(sqrt_inv_c);
        ssa_opt_free_ptr(G);
        ssa_opt_free_ptr(norms);
        ssa_opt_free_ptr(h_temp);
        ssa_opt_free_ptr(u_fft);
        ssa_opt_free_ptr(v_fft);

        return 0;
    }

    /**
     * Compute W-correlation between two specific components using direct formula.
     * Uses h_i = conv(u_i, v_i) instead of full reconstruction.
     */
    double ssa_opt_wcorr_pair(const SSA_Opt *ssa, int i, int j)
    {
        if (!ssa || !ssa->decomposed ||
            i < 0 || i >= ssa->n_components ||
            j < 0 || j >= ssa->n_components)
            return 0.0;

        int N = ssa->N;
        int L = ssa->L;
        int K = ssa->K;
        int fft_len = ssa->fft_len;
        int r2c_len = ssa->r2c_len;

        SSA_Opt *ssa_mut = (SSA_Opt *)ssa;

        // Allocate temp buffers for h_i and h_j
        double *h_i = (double *)ssa_opt_alloc(N * sizeof(double));
        double *h_j = (double *)ssa_opt_alloc(N * sizeof(double));
        double *u_fft = (double *)ssa_opt_alloc(2 * r2c_len * sizeof(double));
        double *v_fft = (double *)ssa_opt_alloc(2 * r2c_len * sizeof(double));

        if (!h_i || !h_j || !u_fft || !v_fft)
        {
            ssa_opt_free_ptr(h_i);
            ssa_opt_free_ptr(h_j);
            ssa_opt_free_ptr(u_fft);
            ssa_opt_free_ptr(v_fft);
            return 0.0;
        }

        // Compute h_i = conv(u_i, v_i)
        {
            const double *u_vec = &ssa->U[i * L];
            const double *v_vec = &ssa->V[i * K];

            ssa_opt_zero(ssa_mut->ws_real, fft_len);
            memcpy(ssa_mut->ws_real, u_vec, L * sizeof(double));
            DftiComputeForward(ssa_mut->fft_r2c, ssa_mut->ws_real, u_fft);

            ssa_opt_zero(ssa_mut->ws_real, fft_len);
            memcpy(ssa_mut->ws_real, v_vec, K * sizeof(double));
            DftiComputeForward(ssa_mut->fft_r2c, ssa_mut->ws_real, v_fft);

            ssa_opt_complex_mul_r2c(u_fft, v_fft, ssa_mut->ws_complex, r2c_len);
            DftiComputeBackward(ssa_mut->fft_c2r, ssa_mut->ws_complex, ssa_mut->ws_real);
            memcpy(h_i, ssa_mut->ws_real, N * sizeof(double));
        }

        // Compute h_j = conv(u_j, v_j)
        {
            const double *u_vec = &ssa->U[j * L];
            const double *v_vec = &ssa->V[j * K];

            ssa_opt_zero(ssa_mut->ws_real, fft_len);
            memcpy(ssa_mut->ws_real, u_vec, L * sizeof(double));
            DftiComputeForward(ssa_mut->fft_r2c, ssa_mut->ws_real, u_fft);

            ssa_opt_zero(ssa_mut->ws_real, fft_len);
            memcpy(ssa_mut->ws_real, v_vec, K * sizeof(double));
            DftiComputeForward(ssa_mut->fft_r2c, ssa_mut->ws_real, v_fft);

            ssa_opt_complex_mul_r2c(u_fft, v_fft, ssa_mut->ws_complex, r2c_len);
            DftiComputeBackward(ssa_mut->fft_c2r, ssa_mut->ws_complex, ssa_mut->ws_real);
            memcpy(h_j, ssa_mut->ws_real, N * sizeof(double));
        }

        // Compute W-correlation using direct formula:
        // <X_i, X_j>_W = σ_i × σ_j × Σ_t h_i[t] × h_j[t] / c[t]
        // ||X_i||_W² = σ_i² × Σ_t h_i[t]² / c[t]
        double sigma_i = ssa->sigma[i];
        double sigma_j = ssa->sigma[j];

        double inner = 0.0, norm_i_sq = 0.0, norm_j_sq = 0.0;
        for (int t = 0; t < N; t++)
        {
            int c_t = ssa_opt_min(ssa_opt_min(t + 1, L), ssa_opt_min(K, N - t));
            double inv_c = 1.0 / c_t;
            inner += h_i[t] * h_j[t] * inv_c;
            norm_i_sq += h_i[t] * h_i[t] * inv_c;
            norm_j_sq += h_j[t] * h_j[t] * inv_c;
        }

        inner *= sigma_i * sigma_j;
        norm_i_sq *= sigma_i * sigma_i;
        norm_j_sq *= sigma_j * sigma_j;

        ssa_opt_free_ptr(h_i);
        ssa_opt_free_ptr(h_j);
        ssa_opt_free_ptr(u_fft);
        ssa_opt_free_ptr(v_fft);

        double denom = sqrt(norm_i_sq) * sqrt(norm_j_sq);
        return (denom > 1e-12) ? inner / denom : 0.0;
    }

    // ============================================================================
    // COMPONENT STATISTICS FOR AUTOMATIC SELECTION
    //
    // Analyzes singular value spectrum to help identify signal/noise boundary.
    // Provides multiple diagnostic metrics that can be used individually or combined.
    // ============================================================================

    int ssa_opt_component_stats(const SSA_Opt *ssa, SSA_ComponentStats *stats)
    {
        if (!ssa || !ssa->decomposed || !stats)
            return -1;

        int n = ssa->n_components;
        if (n < 2)
            return -1; // Need at least 2 components for gap analysis

        memset(stats, 0, sizeof(SSA_ComponentStats));
        stats->n = n;

        // Allocate diagnostic arrays
        stats->singular_values = (double *)ssa_opt_alloc(n * sizeof(double));
        stats->log_sv = (double *)ssa_opt_alloc(n * sizeof(double));
        stats->gaps = (double *)ssa_opt_alloc((n - 1) * sizeof(double));
        stats->cumulative_var = (double *)ssa_opt_alloc(n * sizeof(double));
        stats->second_diff = (double *)ssa_opt_alloc(n * sizeof(double));

        if (!stats->singular_values || !stats->log_sv || !stats->gaps ||
            !stats->cumulative_var || !stats->second_diff)
        {
            ssa_opt_component_stats_free(stats);
            return -1;
        }

        // =========================================================================
        // Compute log singular values for scree plot analysis
        //
        // In a scree plot, signal components typically show a steep decline,
        // while noise components form a nearly flat "elbow". The log transform
        // makes this pattern easier to detect algorithmically.
        // =========================================================================
        for (int i = 0; i < n; i++)
        {
            stats->singular_values[i] = ssa->sigma[i];
            stats->log_sv[i] = log(ssa->sigma[i] + 1e-300); // Avoid log(0)
        }

        // =========================================================================
        // Compute gap ratios: gaps[i] = σᵢ / σᵢ₊₁
        //
        // Large gap ratio indicates a boundary between component groups.
        // Signal components decay slowly (gap ≈ 1), while the signal/noise
        // boundary often shows a large gap (gap >> 1).
        //
        // Track the maximum gap as a simple automatic cutoff suggestion.
        // =========================================================================
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

        // =========================================================================
        // Compute cumulative explained variance ratio
        //
        // cumulative_var[i] = (σ₀² + σ₁² + ... + σᵢ²) / (total variance)
        //
        // Useful for determining how many components capture "most" of the signal.
        // Common heuristics: keep components until 90% or 95% variance explained.
        // =========================================================================
        double cumsum = 0.0;
        for (int i = 0; i < n; i++)
        {
            cumsum += ssa->eigenvalues[i];
            stats->cumulative_var[i] = cumsum / ssa->total_variance;
        }

        // =========================================================================
        // Compute second difference of log singular values
        //
        // d²[i] = log(σᵢ₋₁) - 2·log(σᵢ) + log(σᵢ₊₁)
        //
        // This is a discrete approximation to the second derivative, which
        // helps detect the "elbow" in the scree plot. Large positive d² values
        // indicate convex regions (potential signal/noise boundaries).
        // =========================================================================
        stats->second_diff[0] = 0.0;
        stats->second_diff[n - 1] = 0.0;

        for (int i = 1; i < n - 1; i++)
        {
            double d2 = stats->log_sv[i - 1] - 2.0 * stats->log_sv[i] + stats->log_sv[i + 1];
            stats->second_diff[i] = d2;
        }

        // Simple heuristic: suggest including components 0..max_gap_idx as signal
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

    // ============================================================================
    // PERIODIC PAIR DETECTION
    //
    // Periodic signals in SSA typically appear as pairs of components with:
    //   1. Nearly equal singular values (sine and cosine have same amplitude)
    //   2. High W-correlation (they reconstruct similar-looking signals)
    //
    // This function identifies such pairs automatically for grouping.
    // ============================================================================

    int ssa_opt_find_periodic_pairs(const SSA_Opt *ssa, int *pairs, int max_pairs,
                                    double sv_tol, double wcorr_thresh)
    {
        if (!ssa || !ssa->decomposed || !pairs || max_pairs < 1)
            return 0;

        // Default tolerances if not specified
        if (sv_tol <= 0)
            sv_tol = 0.1; // Within 10% singular value ratio
        if (wcorr_thresh <= 0)
            wcorr_thresh = 0.5; // W-correlation > 0.5 indicates related components

        int n = ssa->n_components;
        int n_pairs = 0;

        // Track which components have already been paired
        bool *used = (bool *)calloc(n, sizeof(bool));
        if (!used)
            return 0;

        // Scan through components looking for pairs
        // We check consecutive or nearby components since periodic pairs
        // typically have adjacent or very close singular values
        for (int i = 0; i < n - 1 && n_pairs < max_pairs; i++)
        {
            if (used[i])
                continue;

            for (int j = i + 1; j < n && n_pairs < max_pairs; j++)
            {
                if (used[j])
                    continue;

                // Check 1: Singular values must be close
                // For a pure periodic signal, sin and cos components have equal σ
                double sv_ratio = ssa->sigma[j] / (ssa->sigma[i] + 1e-300);
                if (fabs(1.0 - sv_ratio) > sv_tol)
                    continue;

                // Check 2: W-correlation must be high
                // Periodic pairs reconstruct similar-looking signals
                double wcorr = fabs(ssa_opt_wcorr_pair(ssa, i, j));
                if (wcorr < wcorr_thresh)
                    continue;

                // Found a pair! Record it
                pairs[2 * n_pairs] = i;
                pairs[2 * n_pairs + 1] = j;
                used[i] = true;
                used[j] = true;
                n_pairs++;
                break; // Move to next i
            }
        }

        free(used);
        return n_pairs;
    }

    // ============================================================================
    // SSA FORECASTING (Linear Recurrence Formula)
    //
    // The key insight is that SSA signal components satisfy a linear recurrence:
    //   x̃[t] = R[0]·x̃[t-L+1] + R[1]·x̃[t-L+2] + ... + R[L-2]·x̃[t-1]
    //
    // The recurrence coefficients R are derived from the left singular vectors:
    //   Let π = [U₁[L-1], U₂[L-1], ...] be the last row of U for selected components
    //   Let U∇ = U without last row (first L-1 rows)
    //   ν² = ‖π‖² (verticality coefficient, must be < 1)
    //   R = U∇ × πᵀ / (1 - ν²)
    //
    // This enables extrapolation beyond the observed data for forecasting.
    //
    // Reference: Golyandina, N., & Zhigljavsky, A. (2013). Singular Spectrum
    //            Analysis for Time Series. Springer. Chapter 3.
    // ============================================================================

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

        // =========================================================================
        // Step 1: Extract π = last row of U for selected components
        //
        // π[i] = U[group[i]][L-1] for each component in the group
        // This encodes the "vertical" component that drives the recurrence
        // =========================================================================
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
            // U is stored column-major: U[row + col*L]
            pi[g] = ssa->U[idx * L + (L - 1)];
            nu_sq += pi[g] * pi[g];
        }

        lrf->verticality = nu_sq;

        // =========================================================================
        // Step 2: Check verticality condition
        //
        // For the LRF to be valid, we need ν² < 1. If ν² ≥ 1, the recurrence
        // is unstable and forecasting is not possible. This happens when the
        // signal has a strong component in the "vertical" direction.
        // =========================================================================
        if (nu_sq >= 1.0 - 1e-10)
        {
            ssa_opt_free_ptr(pi);
            lrf->valid = false;
            return -1; // Cannot forecast - verticality too high
        }

        // =========================================================================
        // Step 3: Compute recurrence coefficients
        //
        // R[j] = Σᵢ (π[i] × U[group[i]][j]) / (1 - ν²)  for j = 0..L-2
        //
        // This is equivalent to R = U∇ × πᵀ / (1 - ν²) where U∇ is U without
        // the last row. The coefficients define the linear recurrence.
        // =========================================================================
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
                // U[row j, column idx] = U[idx * L + j]
                sum += pi[g] * ssa->U[idx * L + j];
            }
            lrf->R[j] = sum * scale;
        }

        ssa_opt_free_ptr(pi);
        lrf->valid = true;

        return 0;
    }

    int ssa_opt_forecast_with_lrf(const SSA_LRF *lrf, const double *base_signal, int base_len,
                                  int n_forecast, double *output)
    {
        if (!lrf || !lrf->valid || !lrf->R || !base_signal || !output)
            return -1;

        int L = lrf->L;

        // Need at least L-1 points to start the recurrence
        if (base_len < L - 1 || n_forecast < 1)
            return -1;

        // =========================================================================
        // Apply the linear recurrence to forecast
        //
        // x̃[t] = Σⱼ R[j] × x̃[t-L+1+j]  for j = 0..L-2
        //
        // We need to maintain a sliding window of the last L-1 values.
        // Initially, this comes from the end of base_signal.
        // As we forecast, new predictions enter the window.
        // =========================================================================

        // Create a working buffer with base signal tail + space for forecasts
        int window_size = L - 1;
        double *buffer = (double *)ssa_opt_alloc((window_size + n_forecast) * sizeof(double));
        if (!buffer)
            return -1;

        // Copy the last L-1 points of base_signal into buffer
        for (int i = 0; i < window_size; i++)
        {
            buffer[i] = base_signal[base_len - window_size + i];
        }

        // Apply recurrence to generate forecasts
        for (int h = 0; h < n_forecast; h++)
        {
            double forecast = 0.0;

            // x̃[t] = R[0]·x̃[t-L+1] + R[1]·x̃[t-L+2] + ... + R[L-2]·x̃[t-1]
            // In buffer coordinates: buffer[h + j] for j = 0..L-2
            for (int j = 0; j < window_size; j++)
            {
                forecast += lrf->R[j] * buffer[h + j];
            }

            buffer[window_size + h] = forecast;
            output[h] = forecast;
        }

        ssa_opt_free_ptr(buffer);
        return 0;
    }

    int ssa_opt_forecast(const SSA_Opt *ssa, const int *group, int n_group,
                         int n_forecast, double *output)
    {
        if (!ssa || !ssa->decomposed || !group || !output || n_group < 1 || n_forecast < 1)
            return -1;

        int N = ssa->N;

        // =========================================================================
        // Step 1: Compute LRF coefficients from selected components
        // =========================================================================
        SSA_LRF lrf = {0};
        if (ssa_opt_compute_lrf(ssa, group, n_group, &lrf) != 0)
        {
            return -1;
        }

        // =========================================================================
        // Step 2: Reconstruct signal from selected components
        // =========================================================================
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

        // =========================================================================
        // Step 3: Apply LRF to forecast
        // =========================================================================
        int result = ssa_opt_forecast_with_lrf(&lrf, reconstructed, N, n_forecast, output);

        ssa_opt_free_ptr(reconstructed);
        ssa_opt_lrf_free(&lrf);

        return result;
    }

    int ssa_opt_forecast_full(const SSA_Opt *ssa, const int *group, int n_group,
                              int n_forecast, double *output)
    {
        if (!ssa || !ssa->decomposed || !group || !output || n_group < 1 || n_forecast < 1)
            return -1;

        int N = ssa->N;

        // =========================================================================
        // Step 1: Compute LRF coefficients
        // =========================================================================
        SSA_LRF lrf = {0};
        if (ssa_opt_compute_lrf(ssa, group, n_group, &lrf) != 0)
        {
            return -1;
        }

        // =========================================================================
        // Step 2: Reconstruct signal into output[0..N-1]
        // =========================================================================
        if (ssa_opt_reconstruct(ssa, group, n_group, output) != 0)
        {
            ssa_opt_lrf_free(&lrf);
            return -1;
        }

        // =========================================================================
        // Step 3: Forecast into output[N..N+n_forecast-1]
        // =========================================================================
        int result = ssa_opt_forecast_with_lrf(&lrf, output, N, n_forecast, output + N);

        ssa_opt_lrf_free(&lrf);
        return result;
    }

    // ============================================================================
    // CONVENIENCE FUNCTIONS
    // ============================================================================

    void ssa_opt_free(SSA_Opt *ssa)
    {
        if (!ssa)
            return;

        if (ssa->fft_r2c)
            DftiFreeDescriptor(&ssa->fft_r2c);
        if (ssa->fft_c2r)
            DftiFreeDescriptor(&ssa->fft_c2r);
        if (ssa->fft_r2c_batch)
            DftiFreeDescriptor(&ssa->fft_r2c_batch);
        if (ssa->fft_c2r_batch)
            DftiFreeDescriptor(&ssa->fft_c2r_batch);
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

        memset(ssa, 0, sizeof(SSA_Opt));
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

    // ============================================================================
    // MSSA (MULTIVARIATE SSA) IMPLEMENTATION - R2C VERSION
    //
    // MSSA extends univariate SSA to M correlated time series by forming a
    // block trajectory matrix:
    //
    //   H = [ H₁ ]    where Hₘ is the L×K Hankel matrix for series m
    //       [ H₂ ]
    //       [ ⋮  ]
    //       [ Hₘ ]
    //
    // The block matrix has dimensions (M*L) × K. SVD of H captures:
    //   - Common factors shared across series (market effect, sector trend)
    //   - Cross-correlations between series (pairs relationships)
    //   - Series-specific components (idiosyncratic movements)
    //
    // KEY INSIGHT: Block Hankel matvec is M independent convolutions:
    //   y = H @ v  →  y[m*L : (m+1)*L] = conv(xₘ, reverse(v))[K-1 : K-1+L]
    //   y = Hᵀ @ u →  y = Σₘ conv(xₘ, reverse(u[m*L : (m+1)*L]))[L-1 : L-1+K]
    //
    // Reference: Golyandina, N. (2010). "On the choice of parameters in
    //            Singular Spectrum Analysis and related subspace-based methods"
    // ============================================================================

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
        if (!mssa || !X || M < 1 || N < 4 || L < 2 || L > N - 1)
        {
            return -1;
        }

        memset(mssa, 0, sizeof(MSSA_Opt));

        mssa->M = M;
        mssa->N = N;
        mssa->L = L;
        mssa->K = N - L + 1;

        // FFT length for convolution
        int conv_len = N + mssa->K - 1;
        int fft_n = ssa_opt_next_pow2(conv_len);
        mssa->fft_len = fft_n;
        mssa->r2c_len = fft_n / 2 + 1;

        // =========================================================================
        // Allocate workspace (R2C format - 50% less than C2C!)
        // =========================================================================
        mssa->ws_real = (double *)ssa_opt_alloc(fft_n * sizeof(double));
        mssa->ws_complex = (double *)ssa_opt_alloc(2 * mssa->r2c_len * sizeof(double));

        // Batch buffers sized for M series (for parallel matvec)
        size_t batch_real_size = M * fft_n * sizeof(double);
        size_t batch_complex_size = M * 2 * mssa->r2c_len * sizeof(double);
        mssa->ws_batch_real = (double *)ssa_opt_alloc(batch_real_size);
        mssa->ws_batch_complex = (double *)ssa_opt_alloc(batch_complex_size);

        // Precomputed FFT(x) for each series (R2C format - 50% less!)
        mssa->fft_x = (double *)ssa_opt_alloc(M * 2 * mssa->r2c_len * sizeof(double));

        if (!mssa->ws_real || !mssa->ws_complex || !mssa->ws_batch_real ||
            !mssa->ws_batch_complex || !mssa->fft_x)
        {
            mssa_opt_free(mssa);
            return -1;
        }

        // =========================================================================
        // Create MKL R2C/C2R FFT descriptors
        // =========================================================================
        MKL_LONG status;

        // Single R2C FFT
        status = DftiCreateDescriptor(&mssa->fft_r2c, DFTI_DOUBLE, DFTI_REAL, 1, fft_n);
        if (status != 0)
        {
            mssa_opt_free(mssa);
            return -1;
        }
        DftiSetValue(mssa->fft_r2c, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
        DftiSetValue(mssa->fft_r2c, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
        status = DftiCommitDescriptor(mssa->fft_r2c);
        if (status != 0)
        {
            mssa_opt_free(mssa);
            return -1;
        }

        // Single C2R FFT
        status = DftiCreateDescriptor(&mssa->fft_c2r, DFTI_DOUBLE, DFTI_REAL, 1, fft_n);
        if (status != 0)
        {
            mssa_opt_free(mssa);
            return -1;
        }
        DftiSetValue(mssa->fft_c2r, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
        DftiSetValue(mssa->fft_c2r, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
        DftiSetValue(mssa->fft_c2r, DFTI_BACKWARD_SCALE, 1.0 / fft_n);
        status = DftiCommitDescriptor(mssa->fft_c2r);
        if (status != 0)
        {
            mssa_opt_free(mssa);
            return -1;
        }

        // Batched R2C for M series
        status = DftiCreateDescriptor(&mssa->fft_r2c_batch, DFTI_DOUBLE, DFTI_REAL, 1, fft_n);
        if (status != 0)
        {
            mssa_opt_free(mssa);
            return -1;
        }
        DftiSetValue(mssa->fft_r2c_batch, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
        DftiSetValue(mssa->fft_r2c_batch, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
        DftiSetValue(mssa->fft_r2c_batch, DFTI_NUMBER_OF_TRANSFORMS, (MKL_LONG)M);
        DftiSetValue(mssa->fft_r2c_batch, DFTI_INPUT_DISTANCE, (MKL_LONG)fft_n);
        DftiSetValue(mssa->fft_r2c_batch, DFTI_OUTPUT_DISTANCE, (MKL_LONG)mssa->r2c_len);
        status = DftiCommitDescriptor(mssa->fft_r2c_batch);
        if (status != 0)
        {
            mssa_opt_free(mssa);
            return -1;
        }

        // Batched C2R
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
        status = DftiCommitDescriptor(mssa->fft_c2r_batch);
        if (status != 0)
        {
            mssa_opt_free(mssa);
            return -1;
        }

        // =========================================================================
        // Precompute FFT(x) for each series using R2C
        // =========================================================================
        for (int m = 0; m < M; m++)
        {
            double *fft_xm = mssa->fft_x + m * 2 * mssa->r2c_len;

            ssa_opt_zero(mssa->ws_real, fft_n);
            memcpy(mssa->ws_real, X + m * N, N * sizeof(double));

            DftiComputeForward(mssa->fft_r2c, mssa->ws_real, fft_xm);
        }

        // =========================================================================
        // Initialize RNG for randomized SVD
        // =========================================================================
        status = vslNewStream(&mssa->rng, VSL_BRNG_MT19937, 42);
        if (status != 0)
        {
            mssa_opt_free(mssa);
            return -1;
        }

        // =========================================================================
        // Precompute diagonal averaging weights
        // =========================================================================
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

    // ============================================================================
    // MSSA Hankel Matrix-Vector Products (R2C Version)
    // ============================================================================

    /**
     * @brief Compute y = H @ v for block Hankel matrix (R2C version).
     *
     * y has length M*L (M blocks of L elements each)
     * v has length K
     *
     * Each block is an independent convolution with the corresponding series.
     */
    static void mssa_opt_hankel_matvec(MSSA_Opt *mssa, const double *v, double *y)
    {
        int M = mssa->M;
        int K = mssa->K;
        int L = mssa->L;
        int fft_len = mssa->fft_len;
        int r2c_len = mssa->r2c_len;

        // Prepare FFT(reverse(v)) once - R2C version with SIMD reverse
        ssa_opt_zero(mssa->ws_real, fft_len);
        ssa_opt_reverse_copy(v, mssa->ws_real, K);
        DftiComputeForward(mssa->fft_r2c, mssa->ws_real, mssa->ws_complex);

        // For each series, compute convolution
        for (int m = 0; m < M; m++)
        {
            double *fft_xm = mssa->fft_x + m * 2 * r2c_len;
            double *ym = y + m * L;
            double *ws_out = mssa->ws_batch_complex; // Reuse as temp

            // Pointwise multiply: FFT(xm) ⊙ FFT(v_rev) - HALF the work!
            vzMul(r2c_len, (const MKL_Complex16 *)fft_xm, (const MKL_Complex16 *)mssa->ws_complex,
                  (MKL_Complex16 *)ws_out);

            // C2R IFFT - direct real output!
            DftiComputeBackward(mssa->fft_c2r, ws_out, mssa->ws_batch_real);

            // Extract result: conv[K-1 : K-1+L] via contiguous memcpy
            memcpy(ym, mssa->ws_batch_real + (K - 1), L * sizeof(double));
        }
    }

    /**
     * @brief Compute y = Hᵀ @ u for block Hankel matrix (R2C version).
     *
     * u has length M*L (M blocks of L elements each)
     * y has length K
     *
     * Result is sum of M convolutions.
     */
    static void mssa_opt_hankel_matvec_T(MSSA_Opt *mssa, const double *u, double *y)
    {
        int M = mssa->M;
        int K = mssa->K;
        int L = mssa->L;
        int fft_len = mssa->fft_len;
        int r2c_len = mssa->r2c_len;

        ssa_opt_zero(y, K);

        for (int m = 0; m < M; m++)
        {
            const double *um = u + m * L;
            double *fft_xm = mssa->fft_x + m * 2 * r2c_len;

            // Pack reverse(um) using SIMD-optimized BLAS copy
            ssa_opt_zero(mssa->ws_real, fft_len);
            ssa_opt_reverse_copy(um, mssa->ws_real, L);

            // R2C FFT
            DftiComputeForward(mssa->fft_r2c, mssa->ws_real, mssa->ws_complex);

            // Complex multiply - HALF the work!
            double *ws_out = mssa->ws_batch_complex;
            vzMul(r2c_len, (const MKL_Complex16 *)fft_xm, (const MKL_Complex16 *)mssa->ws_complex,
                  (MKL_Complex16 *)ws_out);

            // C2R IFFT
            DftiComputeBackward(mssa->fft_c2r, ws_out, mssa->ws_batch_real);

            // Accumulate: y += conv[L-1 : L-1+K]
            for (int j = 0; j < K; j++)
            {
                y[j] += mssa->ws_batch_real[(L - 1) + j];
            }
        }
    }

    /**
     * @brief Batched Hankel matvec: Y = H @ V_block (R2C version).
     *
     * V_block is K × b (b vectors in columns)
     * Y_block is (M*L) × b
     */
    static void mssa_opt_hankel_matvec_batch(MSSA_Opt *mssa, const double *V_block,
                                             double *Y_block, int b)
    {
        int ML = mssa->M * mssa->L;

        // Sequential approach - can be optimized with 2D batched FFT if needed
        for (int j = 0; j < b; j++)
        {
            mssa_opt_hankel_matvec(mssa, V_block + j * mssa->K, Y_block + j * ML);
        }
    }

    /**
     * @brief Batched Hankel transpose matvec: Y = Hᵀ @ U_block (R2C version).
     *
     * U_block is (M*L) × b
     * Y_block is K × b
     */
    static void mssa_opt_hankel_matvec_T_batch(MSSA_Opt *mssa, const double *U_block,
                                               double *Y_block, int b)
    {
        int ML = mssa->M * mssa->L;
        int K = mssa->K;

        for (int j = 0; j < b; j++)
        {
            mssa_opt_hankel_matvec_T(mssa, U_block + j * ML, Y_block + j * K);
        }
    }

    // ============================================================================
    // MSSA Decomposition (Randomized SVD) - R2C Version
    // ============================================================================

    int mssa_opt_decompose(MSSA_Opt *mssa, int k, int oversampling)
    {
        if (!mssa || !mssa->initialized || k < 1)
            return -1;

        int M = mssa->M;
        int L = mssa->L;
        int K = mssa->K;
        int ML = M * L; // Block Hankel row dimension

        int p = (oversampling <= 0) ? 8 : oversampling;
        int kp = k + p;

        kp = ssa_opt_min(kp, ssa_opt_min(ML, K));
        k = ssa_opt_min(k, kp);

        // =========================================================================
        // Allocate result storage
        // =========================================================================
        mssa->U = (double *)ssa_opt_alloc(ML * k * sizeof(double));
        mssa->V = (double *)ssa_opt_alloc(K * k * sizeof(double));
        mssa->sigma = (double *)ssa_opt_alloc(k * sizeof(double));
        mssa->eigenvalues = (double *)ssa_opt_alloc(k * sizeof(double));

        if (!mssa->U || !mssa->V || !mssa->sigma || !mssa->eigenvalues)
        {
            return -1;
        }

        mssa->n_components = k;
        mssa->total_variance = 0.0;

        // =========================================================================
        // Allocate workspace for randomized SVD
        // =========================================================================
        double *Omega = (double *)ssa_opt_alloc(K * kp * sizeof(double));
        double *Y = (double *)ssa_opt_alloc(ML * kp * sizeof(double));
        double *Q = (double *)ssa_opt_alloc(ML * kp * sizeof(double));
        double *B = (double *)ssa_opt_alloc(K * kp * sizeof(double));
        double *tau = (double *)ssa_opt_alloc(kp * sizeof(double));

        double *U_svd = (double *)ssa_opt_alloc(K * kp * sizeof(double));
        double *Vt_svd = (double *)ssa_opt_alloc(kp * kp * sizeof(double));
        double *S_svd = (double *)ssa_opt_alloc(kp * sizeof(double));

        double work_query;
        int *iwork = (int *)ssa_opt_alloc(8 * kp * sizeof(int));
        int lwork = -1;
        int info;

        // Query optimal workspace
        LAPACKE_dgesdd_work(LAPACK_COL_MAJOR, 'S', K, kp, B, K, S_svd,
                            U_svd, K, Vt_svd, kp, &work_query, lwork, iwork);
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

        // =========================================================================
        // Step 1: Generate random Gaussian matrix Ω (K × kp)
        // =========================================================================
        vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER, mssa->rng, K * kp, Omega, 0.0, 1.0);

        // =========================================================================
        // Step 2: Range sampling Y = H @ Ω (uses R2C FFT internally)
        // =========================================================================
        mssa_opt_hankel_matvec_batch(mssa, Omega, Y, kp);

        // =========================================================================
        // Step 3: Orthonormalize Q = orth(Y) via QR
        // =========================================================================
        cblas_dcopy(ML * kp, Y, 1, Q, 1);
        LAPACKE_dgeqrf(LAPACK_COL_MAJOR, ML, kp, Q, ML, tau);
        LAPACKE_dorgqr(LAPACK_COL_MAJOR, ML, kp, kp, Q, ML, tau);

        // =========================================================================
        // Step 4: Project B = Hᵀ @ Q (uses R2C FFT internally)
        // =========================================================================
        mssa_opt_hankel_matvec_T_batch(mssa, Q, B, kp);

        // =========================================================================
        // Step 5: SVD of B
        // =========================================================================
        info = LAPACKE_dgesdd_work(LAPACK_COL_MAJOR, 'S', K, kp, B, K, S_svd,
                                   U_svd, K, Vt_svd, kp, work, lwork, iwork);

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

        // =========================================================================
        // Step 6: Recover U and V
        //
        // U_final = Q @ Vt_svdᵀ (first k columns)
        // V_final = U_svd (first k columns)
        // =========================================================================
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                    ML, k, kp,
                    1.0, Q, ML, Vt_svd, kp,
                    0.0, mssa->U, ML);

        for (int i = 0; i < k; i++)
        {
            cblas_dcopy(K, &U_svd[i * K], 1, &mssa->V[i * K], 1);
        }

        for (int i = 0; i < k; i++)
        {
            mssa->sigma[i] = S_svd[i];
            mssa->eigenvalues[i] = S_svd[i] * S_svd[i];
            mssa->total_variance += mssa->eigenvalues[i];
        }

        // Fix sign convention
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

        // Cleanup
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

    // ============================================================================
    // MSSA Reconstruction (R2C Version)
    // ============================================================================

    int mssa_opt_reconstruct(const MSSA_Opt *mssa, int series_idx,
                             const int *group, int n_group, double *output)
    {
        if (!mssa || !mssa->decomposed || !group || !output ||
            n_group < 1 || series_idx < 0 || series_idx >= mssa->M)
        {
            return -1;
        }

        int M = mssa->M;
        int N = mssa->N;
        int L = mssa->L;
        int K = mssa->K;
        int ML = M * L;
        int fft_len = mssa->fft_len;
        int r2c_len = mssa->r2c_len;

        ssa_opt_zero(output, N);

        // Cast away const for workspace access
        MSSA_Opt *mssa_mut = (MSSA_Opt *)mssa;

        // =========================================================================
        // For each component in group:
        //   Extract U block for this series: U_m = U[series_idx*L : (series_idx+1)*L, :]
        //   Compute reconstruction: conv(σᵢ × u_m_i, v_i) and accumulate
        // =========================================================================
        for (int g = 0; g < n_group; g++)
        {
            int idx = group[g];
            if (idx < 0 || idx >= mssa->n_components)
                continue;

            double sigma = mssa->sigma[idx];
            const double *u_full = &mssa->U[idx * ML];   // Full U column (M*L)
            const double *u_m = u_full + series_idx * L; // Block for this series (L)
            const double *v = &mssa->V[idx * K];         // V column (K)

            // Pack σ × u_m into real array (R2C - no interleaving!)
            ssa_opt_zero(mssa_mut->ws_real, fft_len);
            for (int i = 0; i < L; i++)
            {
                mssa_mut->ws_real[i] = sigma * u_m[i];
            }

            // R2C FFT of scaled u
            DftiComputeForward(mssa_mut->fft_r2c, mssa_mut->ws_real, mssa_mut->ws_complex);

            // Pack v into real array
            ssa_opt_zero(mssa_mut->ws_batch_real, fft_len);
            for (int i = 0; i < K; i++)
            {
                mssa_mut->ws_batch_real[i] = v[i];
            }

            // R2C FFT of v
            double *ws_v = mssa_mut->ws_batch_complex;
            DftiComputeForward(mssa_mut->fft_r2c, mssa_mut->ws_batch_real, ws_v);

            // Complex multiply - HALF the work!
            vzMul(r2c_len, (const MKL_Complex16 *)mssa_mut->ws_complex,
                  (const MKL_Complex16 *)ws_v, (MKL_Complex16 *)mssa_mut->ws_complex);

            // C2R IFFT - direct real output!
            DftiComputeBackward(mssa_mut->fft_c2r, mssa_mut->ws_complex, mssa_mut->ws_real);

            // Accumulate
            for (int t = 0; t < N; t++)
            {
                output[t] += mssa_mut->ws_real[t];
            }
        }

        // Diagonal averaging
        vdMul(N, output, mssa->inv_diag_count, output);

        return 0;
    }

    int mssa_opt_reconstruct_all(const MSSA_Opt *mssa,
                                 const int *group, int n_group, double *output)
    {
        if (!mssa || !mssa->decomposed || !group || !output || n_group < 1)
            return -1;

        int M = mssa->M;
        int N = mssa->N;

        // Reconstruct each series
        for (int m = 0; m < M; m++)
        {
            if (mssa_opt_reconstruct(mssa, m, group, n_group, output + m * N) != 0)
            {
                return -1;
            }
        }

        return 0;
    }

    int mssa_opt_series_contributions(const MSSA_Opt *mssa, double *contributions)
    {
        if (!mssa || !mssa->decomposed || !contributions)
            return -1;

        int M = mssa->M;
        int L = mssa->L;
        int ML = M * L;
        int k = mssa->n_components;

        // =========================================================================
        // For each component i:
        //   contributions[m, i] = ‖U[m*L : (m+1)*L, i]‖² / ‖U[:, i]‖²
        //
        // Since U columns are normalized, ‖U[:, i]‖² = 1
        // So we just compute ‖block‖² for each series
        // =========================================================================
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

#endif // SSA_OPT_IMPLEMENTATION

#ifdef __cplusplus
}
#endif

#endif // SSA_OPT_R2C_H