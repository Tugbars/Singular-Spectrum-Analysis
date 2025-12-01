/*
 * ============================================================================
 * SSA-OPT: High-Performance Singular Spectrum Analysis (MKL Version)
 * ============================================================================
 *
 * PURPOSE:
 *   Production-grade SSA implementation using Intel MKL for maximum performance.
 *   Use ssa_opt_ref.h for accuracy testing and educational purposes.
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
 * PERFORMANCE SUMMARY (N=5000, L=2000, k=32):
 *   Sequential:  ~550 ms  (1.0x)
 *   Block:       ~195 ms  (2.8x)
 *   Randomized:  ~20-40ms (15-25x)
 *
 * USAGE:
 *   #define SSA_OPT_IMPLEMENTATION
 *   #include "ssa_opt.h"
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

#ifndef SSA_OPT_H
#define SSA_OPT_H

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

#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__GNUC__) || defined(__clang__)
#include <xmmintrin.h>
#endif

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
     * MEMORY LAYOUT:
     *   - All matrices are column-major (BLAS/LAPACK convention)
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

        // MKL FFT Descriptors
        DFTI_DESCRIPTOR_HANDLE fft_handle;        // Single C2C FFT (in-place)
        DFTI_DESCRIPTOR_HANDLE fft_batch_c2c;     // Batched C2C (not-in-place)
        DFTI_DESCRIPTOR_HANDLE fft_batch_inplace; // Batched C2C (in-place)

        // Random Number Generator
        VSLStreamStatePtr rng;

        // Precomputed Data
        double *fft_x; // FFT of input signal, interleaved complex

        // Workspace Buffers
        double *ws_fft1;      // FFT scratch buffer 1
        double *ws_fft2;      // FFT scratch buffer 2
        double *ws_real;      // Real-valued scratch
        double *ws_v;         // Vector scratch for sequential iteration
        double *ws_u;         // Vector scratch for sequential iteration
        double *ws_proj;      // Projection coefficients for orthogonalization
        double *ws_batch_u;   // Batch buffer for FFT
        double *ws_batch_v;   // Batch buffer for FFT
        double *ws_batch_out; // Output buffer for NOT_INPLACE FFT

        // Results
        double *U;           // Left singular vectors, L × k, column-major
        double *V;           // Right singular vectors, K × k, column-major
        double *sigma;       // Singular values
        double *eigenvalues; // Squared singular values
        int n_components;    // Number of computed components

        // Reconstruction Optimization
        double *inv_diag_count; // Precomputed 1/count for diagonal averaging
        double *U_fft;          // Cached FFT of U vectors
        double *V_fft;          // Cached FFT of V vectors
        bool fft_cached;        // True if U_fft/V_fft are populated

        // State
        bool initialized;
        bool decomposed;
        double total_variance;
    } SSA_Opt;

    /**
     * @brief Statistics for automatic component selection and analysis.
     *
     * Populated by ssa_opt_component_stats(). Provides gap ratios, cumulative
     * variance, and automatic signal/noise cutoff suggestions based on singular
     * value decay analysis.
     */
    typedef struct
    {
        int n;                   ///< Number of components analyzed
        double *singular_values; ///< Copy of singular values σ₁ ≥ σ₂ ≥ ... ≥ σₙ
        double *log_sv;          ///< log(σᵢ) for scree plot visualization
        double *gaps;            ///< Gap ratios: σᵢ/σᵢ₊₁ for i=0..n-2 (large gap = signal/noise boundary)
        double *cumulative_var;  ///< Cumulative explained variance ratio at each component
        double *second_diff;     ///< Second difference of log(σ) for elbow detection
        int suggested_signal;    ///< Suggested signal component count (0..suggested_signal-1 are signal)
        double gap_threshold;    ///< Gap ratio at the suggested cutoff point
    } SSA_ComponentStats;

    // ============================================================================
    // Public API
    // ============================================================================

    /**
     * @brief Initialize SSA context with input signal.
     *
     * Allocates all workspace buffers, creates MKL FFT descriptors, and precomputes
     * FFT(x) for reuse in all subsequent matvec operations. After this call, no
     * further allocations occur in hot paths (decompose, reconstruct).
     *
     * @param[out] ssa  Pointer to zero-initialized SSA_Opt struct
     * @param[in]  x    Input time series, length N (copied internally)
     * @param[in]  N    Length of input signal (must be ≥ 4)
     * @param[in]  L    Window length (embedding dimension), must satisfy 2 ≤ L ≤ N-1.
     *                  Typical choice: N/3 to N/2. Larger L gives better frequency
     *                  resolution; smaller L gives better trend extraction.
     * @return          0 on success, -1 on invalid parameters or allocation failure
     *
     * @note The Hankel matrix dimensions are L×K where K = N - L + 1.
     * @note Call ssa_opt_free() to release all allocated resources.
     */
    int ssa_opt_init(SSA_Opt *ssa, const double *x, int N, int L);

    /**
     * @brief Compute SVD via sequential power iteration (baseline method).
     *
     * Computes k singular triplets (σᵢ, uᵢ, vᵢ) one at a time using power iteration
     * with deflation. Each component requires O(max_iter × N log N) operations.
     * This is the simplest but slowest method; use decompose_block or
     * decompose_randomized for better performance.
     *
     * @param[in,out] ssa       Initialized SSA context (from ssa_opt_init)
     * @param[in]     k         Number of singular triplets to compute (1 to min(L,K))
     * @param[in]     max_iter  Maximum iterations per component (typical: 100-200)
     * @return                  0 on success, -1 on error
     *
     * @note Results stored in ssa->U, ssa->V, ssa->sigma, ssa->eigenvalues
     * @note Components are sorted by descending singular value after computation
     *
     * @see ssa_opt_decompose_block() for ~3x faster block method
     * @see ssa_opt_decompose_randomized() for ~15-25x faster randomized method
     */
    int ssa_opt_decompose(SSA_Opt *ssa, int k, int max_iter);

    /**
     * @brief Compute SVD via block power iteration with Rayleigh-Ritz refinement.
     *
     * Processes multiple vectors simultaneously using batched FFT operations and
     * GEMM-based orthogonalization. Uses periodic QR factorization for stability
     * and Rayleigh-Ritz extraction for accurate singular values.
     *
     * Algorithm complexity: O(max_iter × N log N) for all k components together
     * (vs O(k × max_iter × N log N) for sequential method).
     *
     * @param[in,out] ssa        Initialized SSA context
     * @param[in]     k          Number of singular triplets to compute
     * @param[in]     block_size Block size b (use 0 for default=SSA_BATCH_SIZE).
     *                           Should match SSA_BATCH_SIZE for optimal FFT batching.
     * @param[in]     max_iter   Maximum iterations per block (typical: 100)
     * @return                   0 on success, -1 on error
     *
     * @note ~3x faster than sequential for k ≥ block_size
     * @note Requires MKL LAPACK for QR factorization (dgeqrf/dorgqr)
     */
    int ssa_opt_decompose_block(SSA_Opt *ssa, int k, int block_size, int max_iter);

    /**
     * @brief Compute SVD via Halko-Martinsson-Tropp randomized algorithm.
     *
     * Uses random projection to find an approximate basis for the column space,
     * then computes a small dense SVD. Only requires 2 passes over the data,
     * making it the fastest method for most use cases.
     *
     * Algorithm:
     *   1. Ω = randn(K, k+p)     — Gaussian random test matrix
     *   2. Y = H × Ω             — Range sampling (batched matvec)
     *   3. Q = orth(Y)           — QR factorization
     *   4. B = Hᵀ × Q            — Projection (batched matvec transpose)
     *   5. SVD(B) → U, Σ, V      — Small dense SVD via MKL dgesdd
     *
     * @param[in,out] ssa          Initialized SSA context
     * @param[in]     k            Number of singular triplets to compute
     * @param[in]     oversampling Oversampling parameter p (use 0 for default=8).
     *                             Larger p improves accuracy at cost of speed.
     *                             Theory guarantees error bounded by O(σₖ₊₁).
     * @return                     0 on success, -1 on error
     *
     * @note ~15-25x faster than sequential method
     * @note Best when k << min(L, K) and singular values decay rapidly
     * @note Reference: Halko, Martinsson, Tropp. "Finding structure with randomness",
     *       SIAM Review 2011, Algorithm 5.1
     */
    int ssa_opt_decompose_randomized(SSA_Opt *ssa, int k, int oversampling);

    /**
     * @brief Extend existing decomposition with additional components.
     *
     * Computes more singular triplets without recomputing existing ones. Uses
     * power iteration with orthogonalization against all previously computed
     * components. Useful when initial k was too small.
     *
     * @param[in,out] ssa          Already decomposed SSA context
     * @param[in]     additional_k Number of additional components to compute
     * @param[in]     max_iter     Maximum iterations per new component
     * @return                     0 on success, -1 on error
     *
     * @note Invalidates any cached FFTs (call ssa_opt_cache_ffts() again if needed)
     * @note New components are merged and sorted with existing ones
     */
    int ssa_opt_extend(SSA_Opt *ssa, int additional_k, int max_iter);

    /**
     * @brief Reconstruct signal from selected SSA components.
     *
     * Computes the sum of rank-1 matrices for selected components and applies
     * diagonal averaging (Hankelization) to produce the reconstructed signal:
     *
     *   output[t] = (1/count[t]) × Σᵢ∈group σᵢ × (uᵢ ⊗ vᵢ)[anti-diagonal t]
     *
     * The outer products are computed implicitly via FFT convolution for O(N log N)
     * complexity per component.
     *
     * @param[in]  ssa      Decomposed SSA context
     * @param[in]  group    Array of component indices to include (0-based)
     * @param[in]  n_group  Number of components in group array
     * @param[out] output   Output buffer of length N (will be overwritten)
     * @return              0 on success, -1 on error
     *
     * @note For multiple reconstructions, call ssa_opt_cache_ffts() first for 2-3x speedup
     *
     * @example
     *   // Extract trend (first component)
     *   int trend[] = {0};
     *   ssa_opt_reconstruct(&ssa, trend, 1, trend_output);
     *
     *   // Extract periodic signal (components 1-4)
     *   int periodic[] = {1, 2, 3, 4};
     *   ssa_opt_reconstruct(&ssa, periodic, 4, periodic_output);
     */
    int ssa_opt_reconstruct(const SSA_Opt *ssa, const int *group, int n_group, double *output);

    /**
     * @brief Cache FFTs of U and V vectors for faster repeated reconstruction.
     *
     * Precomputes FFT(σᵢ × uᵢ) and FFT(vᵢ) for all components. Subsequent calls
     * to ssa_opt_reconstruct() skip the forward FFT step, saving 2k FFT operations
     * per reconstruction.
     *
     * Typical speedup: 2-3x for workflows that call reconstruct multiple times
     * with different groupings (e.g., extract trend, then periodic, then noise).
     *
     * @param[in,out] ssa  Decomposed SSA context
     * @return             0 on success, -1 on error
     *
     * @note Memory cost: ~16 × fft_len × k bytes (e.g., ~4MB for k=32, N=5000)
     * @note Cache is automatically invalidated by ssa_opt_extend()
     * @note Call ssa_opt_free_cached_ffts() to manually release cache memory
     */
    int ssa_opt_cache_ffts(SSA_Opt *ssa);

    /**
     * @brief Free cached FFTs to release memory.
     *
     * Optional cleanup function. Cached FFTs are also freed by ssa_opt_free()
     * and automatically invalidated by ssa_opt_extend().
     *
     * @param[in,out] ssa  SSA context (safe to call if cache not populated)
     */
    void ssa_opt_free_cached_ffts(SSA_Opt *ssa);

    /**
     * @brief Free all memory associated with SSA context.
     *
     * Releases all allocated buffers, destroys MKL FFT descriptors, and frees
     * the RNG stream. Safe to call on partially initialized or already-freed contexts.
     *
     * @param[in,out] ssa  SSA context to free (zeroed after call)
     */
    void ssa_opt_free(SSA_Opt *ssa);

    // ============================================================================
    // Analysis API
    // ============================================================================

    /**
     * @brief Compute W-correlation matrix between all SSA components.
     *
     * W-correlation measures similarity between reconstructed components using
     * diagonal averaging weights. High |ρ_W(i,j)| indicates components i and j
     * should be grouped together (e.g., sine/cosine pairs for periodic signals).
     *
     * Formula: ρ_W(i,j) = <Xᵢ, Xⱼ>_W / (‖Xᵢ‖_W × ‖Xⱼ‖_W)
     * where <·,·>_W uses weights w[t] = min(t+1, L, K, N-t).
     *
     * @param[in]  ssa  Decomposed SSA context
     * @param[out] W    Output matrix, n_components × n_components, row-major.
     *                  W[i*n + j] = ρ_W(component_i, component_j).
     *                  Caller must allocate n_components² doubles.
     * @return          0 on success, -1 on error
     *
     * @note Diagonal elements W[i,i] = 1.0; matrix is symmetric
     * @note Computationally expensive: reconstructs all components internally
     */
    int ssa_opt_wcorr_matrix(const SSA_Opt *ssa, double *W);

    /**
     * @brief Compute W-correlation between two specific components.
     *
     * Efficient alternative to ssa_opt_wcorr_matrix() when only one pair is needed.
     *
     * @param[in] ssa  Decomposed SSA context
     * @param[in] i    First component index (0-based)
     * @param[in] j    Second component index (0-based)
     * @return         W-correlation value in [-1, 1], or 0.0 on error
     */
    double ssa_opt_wcorr_pair(const SSA_Opt *ssa, int i, int j);

    /**
     * @brief Analyze singular value spectrum for automatic component selection.
     *
     * Computes various diagnostics to help identify signal/noise boundary:
     * - Gap ratios (σᵢ/σᵢ₊₁): large gaps suggest component boundaries
     * - Cumulative variance: how much variance each component explains
     * - Second differences of log(σ): for elbow detection in scree plots
     *
     * @param[in]  ssa    Decomposed SSA context (must have ≥2 components)
     * @param[out] stats  Output statistics structure (caller allocates struct,
     *                    function allocates internal arrays)
     * @return            0 on success, -1 on error
     *
     * @note Call ssa_opt_component_stats_free() to release internal arrays
     */
    int ssa_opt_component_stats(const SSA_Opt *ssa, SSA_ComponentStats *stats);

    /**
     * @brief Free memory allocated inside SSA_ComponentStats structure.
     *
     * @param[in,out] stats  Statistics structure to free (zeroed after call)
     */
    void ssa_opt_component_stats_free(SSA_ComponentStats *stats);

    /**
     * @brief Find component pairs that likely represent periodic signals.
     *
     * Periodic signals in SSA typically appear as pairs of components with nearly
     * equal singular values and high W-correlation (sine/cosine pairs at same
     * frequency). This function identifies such pairs automatically.
     *
     * @param[in]  ssa          Decomposed SSA context
     * @param[out] pairs        Output array: pairs[2*j] and pairs[2*j+1] are the
     *                          indices of the j-th detected pair. Caller must
     *                          allocate 2*max_pairs integers.
     * @param[in]  max_pairs    Maximum number of pairs to find
     * @param[in]  sv_tol       Singular value tolerance: |1 - σⱼ/σᵢ| < sv_tol
     *                          (use 0 for default=0.1, i.e., within 10%)
     * @param[in]  wcorr_thresh W-correlation threshold: |ρ_W| > wcorr_thresh
     *                          (use 0 for default=0.5)
     * @return                  Number of pairs found (0 to max_pairs)
     *
     * @example
     *   int pairs[20];  // Space for up to 10 pairs
     *   int n = ssa_opt_find_periodic_pairs(&ssa, pairs, 10, 0, 0);
     *   for (int i = 0; i < n; i++) {
     *       printf("Periodic pair: components %d and %d\n", pairs[2*i], pairs[2*i+1]);
     *   }
     */
    int ssa_opt_find_periodic_pairs(const SSA_Opt *ssa, int *pairs, int max_pairs,
                                    double sv_tol, double wcorr_thresh);

    // ============================================================================
    // Convenience Functions
    // ============================================================================

    /**
     * @brief Reconstruct trend component (component 0 only).
     *
     * Shorthand for reconstructing just the first singular component, which
     * typically captures the overall trend in the time series.
     *
     * @param[in]  ssa     Decomposed SSA context
     * @param[out] output  Output buffer of length N
     * @return             0 on success, -1 on error
     */
    int ssa_opt_get_trend(const SSA_Opt *ssa, double *output);

    /**
     * @brief Reconstruct noise components (from noise_start to end).
     *
     * Reconstructs all components from index noise_start to n_components-1,
     * typically representing the noise portion of the signal.
     *
     * @param[in]  ssa         Decomposed SSA context
     * @param[in]  noise_start First noise component index (0-based)
     * @param[out] output      Output buffer of length N
     * @return                 0 on success, -1 on error
     */
    int ssa_opt_get_noise(const SSA_Opt *ssa, int noise_start, double *output);

    /**
     * @brief Compute explained variance ratio for a component range.
     *
     * Returns (Σᵢ₌ₛₜₐᵣₜᵉⁿᵈ σᵢ²) / (Σⱼ σⱼ²), i.e., the fraction of total variance
     * explained by components in range [start, end].
     *
     * @param[in] ssa    Decomposed SSA context
     * @param[in] start  First component index (0-based, inclusive)
     * @param[in] end    Last component index (inclusive, use -1 for last component)
     * @return           Variance ratio in [0, 1], or 0.0 on error
     *
     * @example
     *   // Variance explained by first 5 components
     *   double var = ssa_opt_variance_explained(&ssa, 0, 4);
     *   printf("Components 0-4 explain %.1f%% of variance\n", var * 100);
     */
    double ssa_opt_variance_explained(const SSA_Opt *ssa, int start, int end);

    // ============================================================================
    // Implementation
    // ============================================================================

#ifdef SSA_OPT_IMPLEMENTATION

    // ----------------------------------------------------------------------------
    // Internal Helpers
    // ----------------------------------------------------------------------------

    /** @brief Round up to next power of 2. Used for FFT length calculation. */
    static inline int ssa_opt_next_pow2(int n)
    {
        int p = 1;
        while (p < n)
            p <<= 1;
        return p;
    }

    static inline int ssa_opt_min(int a, int b) { return a < b ? a : b; }
    static inline int ssa_opt_max(int a, int b) { return a > b ? a : b; }

    /** @brief Allocate aligned memory using MKL allocator (64-byte alignment for AVX-512). */
    static inline void *ssa_opt_alloc(size_t size)
    {
        return mkl_malloc(size, SSA_ALIGN);
    }

    /** @brief Free memory allocated with ssa_opt_alloc(). */
    static inline void ssa_opt_free_ptr(void *ptr)
    {
        mkl_free(ptr);
    }

    // ----------------------------------------------------------------------------
    // Vectorized BLAS Operations (MKL wrappers)
    // ----------------------------------------------------------------------------

    /**
     * @brief Reverse array with prefetching: out[i] = in[n-1-i].
     * Uses software prefetch to hide memory latency for large arrays.
     */
    static inline void ssa_opt_reverse(const double *in, double *out, int n)
    {
        const double *src = in + n - 1;
        for (int i = 0; i < n; i += 4)
        {
            _mm_prefetch((char *)(src - i - 16), _MM_HINT_T0);
            out[i] = src[-i];
            if (i + 1 < n)
                out[i + 1] = src[-i - 1];
            if (i + 2 < n)
                out[i + 2] = src[-i - 2];
            if (i + 3 < n)
                out[i + 3] = src[-i - 3];
        }
    }

    /**
     * @brief Element-wise complex multiply: c = a ⊙ b (interleaved format).
     * Uses MKL vzMul for vectorized complex multiplication.
     * @param n Number of complex elements (array lengths are 2n doubles)
     */
    static void ssa_opt_complex_mul(const double *a, const double *b, double *c, int n)
    {
        vzMul(n, (const MKL_Complex16 *)a, (const MKL_Complex16 *)b, (MKL_Complex16 *)c);
    }

    /** @brief Dot product: return a·b = Σᵢ aᵢbᵢ */
    static inline double ssa_opt_dot(const double *a, const double *b, int n)
    {
        return cblas_ddot(n, a, 1, b, 1);
    }

    /** @brief Euclidean norm: return ‖v‖₂ = √(Σᵢ vᵢ²) */
    static inline double ssa_opt_nrm2(const double *v, int n)
    {
        return cblas_dnrm2(n, v, 1);
    }

    /** @brief Scale vector: v ← s·v */
    static inline void ssa_opt_scal(double *v, int n, double s)
    {
        cblas_dscal(n, s, v, 1);
    }

    /** @brief AXPY operation: y ← y + a·x */
    static inline void ssa_opt_axpy(double *y, const double *x, double a, int n)
    {
        cblas_daxpy(n, a, x, 1, y, 1);
    }

    /** @brief Copy vector: dst ← src */
    static inline void ssa_opt_copy(const double *src, double *dst, int n)
    {
        cblas_dcopy(n, src, 1, dst, 1);
    }

    /**
     * @brief Normalize vector to unit length: v ← v/‖v‖₂
     * @return Original norm before normalization
     */
    static inline double ssa_opt_normalize(double *v, int n)
    {
        double norm = cblas_dnrm2(n, v, 1);
        if (norm > 1e-15)
        {
            double inv = 1.0 / norm;
            cblas_dscal(n, inv, v, 1);
        }
        return norm;
    }

    /** @brief Zero-fill array: v[i] = 0 for all i */
    static inline void ssa_opt_zero(double *v, int n)
    {
        memset(v, 0, n * sizeof(double));
    }

    // ----------------------------------------------------------------------------
    // FFT Operations (MKL DFTI wrappers)
    // ----------------------------------------------------------------------------

    /** @brief In-place forward FFT using precomputed MKL descriptor. */
    static void ssa_opt_fft_forward_inplace(SSA_Opt *ssa, double *data)
    {
        DftiComputeForward(ssa->fft_handle, data);
    }

    /** @brief In-place inverse FFT (includes 1/n scaling). */
    static void ssa_opt_fft_inverse_inplace(SSA_Opt *ssa, double *data)
    {
        DftiComputeBackward(ssa->fft_handle, data);
    }

    /**
     * @brief Real-to-complex FFT: zero-pads input, computes FFT, stores interleaved.
     * @param input      Real input array of length input_len
     * @param input_len  Length of real input
     * @param output     Output interleaved complex array of length 2*fft_len
     */
    static void ssa_opt_fft_r2c(SSA_Opt *ssa, const double *input, int input_len, double *output)
    {
        int n = ssa->fft_len;
        ssa_opt_zero(output, 2 * n);
        for (int i = 0; i < input_len; i++)
        {
            output[2 * i] = input[i]; // Pack real into complex: [re, 0, re, 0, ...]
        }
        ssa_opt_fft_forward_inplace(ssa, output);
    }

    // ----------------------------------------------------------------------------
    // Hankel Matrix-Vector Products via FFT Convolution
    //
    // KEY INSIGHT: The Hankel matrix H has structure H[i,j] = x[i+j], so
    // y = H @ v is equivalent to convolution: y[i] = Σⱼ x[i+j]·v[j].
    // Using convolution theorem: y = IFFT(FFT(x) ⊙ FFT(v_reversed))[K-1:K-1+L]
    // This converts O(L×K) multiply to O(N log N) FFT operations.
    // ----------------------------------------------------------------------------

    /**
     * @brief Compute y = H @ v via FFT convolution.
     *
     * Hankel matvec: y[i] = Σⱼ x[i+j] × v[j] for i=0..L-1, j=0..K-1
     * Implemented as: y = conv(x, reverse(v))[K-1 : K-1+L]
     *
     * @param v  Input vector of length K (right singular vector direction)
     * @param y  Output vector of length L (left singular vector direction)
     */
    static void ssa_opt_hankel_matvec(SSA_Opt *ssa, const double *v, double *y)
    {
        int K = ssa->K;
        int L = ssa->L;
        int n = ssa->fft_len;
        double *ws = ssa->ws_fft1;

        // Pack reversed v into interleaved complex format
        ssa_opt_zero(ws, 2 * n);
        for (int i = 0; i < K; i++)
        {
            ws[2 * i] = v[K - 1 - i];
        }

        // Compute convolution via FFT: y = IFFT(FFT(x) ⊙ FFT(v_rev))
        DftiComputeForward(ssa->fft_handle, ws);
        ssa_opt_complex_mul(ssa->fft_x, ws, ws, n); // fft_x precomputed in init()
        DftiComputeBackward(ssa->fft_handle, ws);

        // Extract result from convolution: conv[K-1 : K-1+L]
        // cblas_dcopy with incx=2 extracts real parts from interleaved complex
        cblas_dcopy(L, ws + 2 * (K - 1), 2, y, 1);
    }

    /**
     * @brief Compute y = Hᵀ @ u via FFT convolution.
     *
     * Hankel transpose matvec: y[j] = Σᵢ x[i+j] × u[i] for j=0..K-1, i=0..L-1
     * Same convolution structure with dimensions swapped.
     *
     * @param u  Input vector of length L (left singular vector direction)
     * @param y  Output vector of length K (right singular vector direction)
     */
    static void ssa_opt_hankel_matvec_T(SSA_Opt *ssa, const double *u, double *y)
    {
        int K = ssa->K;
        int L = ssa->L;
        int n = ssa->fft_len;
        double *ws = ssa->ws_fft1;

        // Pack reversed u into interleaved complex format
        ssa_opt_zero(ws, 2 * n);
        for (int i = 0; i < L; i++)
        {
            ws[2 * i] = u[L - 1 - i];
        }

        // Convolution via FFT
        DftiComputeForward(ssa->fft_handle, ws);
        ssa_opt_complex_mul(ssa->fft_x, ws, ws, n);
        DftiComputeBackward(ssa->fft_handle, ws);

        // Extract result: conv[L-1 : L-1+K]
        cblas_dcopy(K, ws + 2 * (L - 1), 2, y, 1);
    }

    // ----------------------------------------------------------------------------
    // Batched Hankel Matrix-Vector Products
    //
    // Process multiple vectors simultaneously using MKL's batched FFT.
    // For block_size vectors, this is ~3x faster than sequential calls due to:
    //   1. Amortized FFT descriptor overhead
    //   2. Better cache utilization
    //   3. Cross-transform SIMD parallelism in MKL
    // ----------------------------------------------------------------------------

    /**
     * @brief Batched Hankel matvec: Y = H @ V where V and Y are column-major matrices.
     *
     * Computes b matrix-vector products simultaneously using MKL batched FFT.
     * Falls back to sequential for small batches (< BATCH_THRESHOLD).
     *
     * @param V_block  Input matrix, K × b, column-major (b right singular vectors)
     * @param Y_block  Output matrix, L × b, column-major (b left singular vectors)
     * @param b        Number of vectors to process
     */
    static void ssa_opt_hankel_matvec_block(SSA_Opt *ssa, const double *V_block, double *Y_block, int b)
    {
        int K = ssa->K;
        int L = ssa->L;
        int n = ssa->fft_len;
        size_t stride = 2 * n; // Stride between transforms in complex array

        double *ws_in = ssa->ws_batch_u;
        double *ws_out = ssa->ws_batch_out;

        // Threshold for using batched vs sequential FFT
        // Below this, sequential avoids computing unused transforms
        const int BATCH_THRESHOLD = SSA_BATCH_SIZE / 4;

        int col = 0;
        while (col < b)
        {
            int batch_count = (b - col < SSA_BATCH_SIZE) ? (b - col) : SSA_BATCH_SIZE;

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

            // Zero entire workspace (single memset vs per-vector)
            memset(ws_in, 0, SSA_BATCH_SIZE * stride * sizeof(double));

            // Pack reversed vectors into batch workspace
            for (int i = 0; i < batch_count; i++)
            {
                const double *v = &V_block[(col + i) * K];
                double *dst = ws_in + i * stride;
                for (int j = 0; j < K; j++)
                {
                    dst[2 * j] = v[K - 1 - j];
                }
            }

            // Batched forward FFT: ws_in → ws_out (NOT_INPLACE is faster)
            DftiComputeForward(ssa->fft_batch_c2c, ws_in, ws_out);

            // Element-wise complex multiply with precomputed FFT(x)
            for (int i = 0; i < batch_count; i++)
            {
                double *fft_v = ws_out + i * stride;
                vzMul(n, (const MKL_Complex16 *)ssa->fft_x, (const MKL_Complex16 *)fft_v,
                      (MKL_Complex16 *)fft_v);
            }

            // Batched inverse FFT
            DftiComputeBackward(ssa->fft_batch_c2c, ws_out, ws_in);

            // Extract results from convolutions
            for (int i = 0; i < batch_count; i++)
            {
                double *conv = ws_in + i * stride;
                double *y = &Y_block[(col + i) * L];
                cblas_dcopy(L, conv + 2 * (K - 1), 2, y, 1); // Extract real parts
            }

            col += batch_count;
        }
    }

    /**
     * @brief Batched Hankel transpose matvec: Y = Hᵀ @ U where U and Y are column-major.
     *
     * Transpose version of ssa_opt_hankel_matvec_block(). Uses same batching strategy.
     *
     * @param U_block  Input matrix, L × b, column-major (b left singular vectors)
     * @param Y_block  Output matrix, K × b, column-major (b right singular vectors)
     * @param b        Number of vectors to process
     */
    static void ssa_opt_hankel_matvec_T_block(SSA_Opt *ssa, const double *U_block, double *Y_block, int b)
    {
        int K = ssa->K;
        int L = ssa->L;
        int n = ssa->fft_len;
        size_t stride = 2 * n;

        double *ws_in = ssa->ws_batch_u;
        double *ws_out = ssa->ws_batch_out;

        const int BATCH_THRESHOLD = SSA_BATCH_SIZE / 4;

        int col = 0;
        while (col < b)
        {
            int batch_count = (b - col < SSA_BATCH_SIZE) ? (b - col) : SSA_BATCH_SIZE;

            // Fall back to sequential for small batches
            if (batch_count < BATCH_THRESHOLD)
            {
                for (int i = 0; i < batch_count; i++)
                {
                    ssa_opt_hankel_matvec_T(ssa, &U_block[(col + i) * L], &Y_block[(col + i) * K]);
                }
                col += batch_count;
                continue;
            }

            // Zero workspace and pack reversed u vectors
            memset(ws_in, 0, SSA_BATCH_SIZE * stride * sizeof(double));

            for (int i = 0; i < batch_count; i++)
            {
                const double *u = &U_block[(col + i) * L];
                double *dst = ws_in + i * stride;
                for (int j = 0; j < L; j++)
                {
                    dst[2 * j] = u[L - 1 - j];
                }
            }

            // Batched FFT → multiply → IFFT
            DftiComputeForward(ssa->fft_batch_c2c, ws_in, ws_out);

            for (int i = 0; i < batch_count; i++)
            {
                double *fft_u = ws_out + i * stride;
                vzMul(n, (const MKL_Complex16 *)ssa->fft_x, (const MKL_Complex16 *)fft_u,
                      (MKL_Complex16 *)fft_u);
            }

            DftiComputeBackward(ssa->fft_batch_c2c, ws_out, ws_in);

            // Extract results: conv[L-1 : L-1+K]
            for (int i = 0; i < batch_count; i++)
            {
                double *conv = ws_in + i * stride;
                double *y = &Y_block[(col + i) * K];
                cblas_dcopy(K, conv + 2 * (L - 1), 2, y, 1);
            }

            col += batch_count;
        }
    }

    // ============================================================================
    // INITIALIZATION
    // ============================================================================

    int ssa_opt_init(SSA_Opt *ssa, const double *x, int N, int L)
    {
        if (!ssa || !x || N < 4 || L < 2 || L > N - 1)
        {
            return -1;
        }

        memset(ssa, 0, sizeof(SSA_Opt));

        // Store dimensions
        ssa->N = N;
        ssa->L = L;
        ssa->K = N - L + 1; // Number of lagged vectors in trajectory matrix

        // FFT length: next power of 2 ≥ convolution length
        int conv_len = N + ssa->K - 1;
        int fft_n = ssa_opt_next_pow2(conv_len);
        ssa->fft_len = fft_n;

        // -------------------------------------------------------------------------
        // Allocate workspace buffers (all allocations done here, none in hot paths)
        // -------------------------------------------------------------------------
        size_t fft_size = 2 * fft_n * sizeof(double); // Interleaved complex
        size_t vec_size = ssa_opt_max(L, ssa->K) * sizeof(double);

        ssa->fft_x = (double *)ssa_opt_alloc(fft_size);                 // Precomputed FFT(x)
        ssa->ws_fft1 = (double *)ssa_opt_alloc(fft_size);               // FFT workspace 1
        ssa->ws_fft2 = (double *)ssa_opt_alloc(fft_size);               // FFT workspace 2
        ssa->ws_real = (double *)ssa_opt_alloc(fft_n * sizeof(double)); // Real scratch
        ssa->ws_v = (double *)ssa_opt_alloc(vec_size);                  // Vector scratch (v)
        ssa->ws_u = (double *)ssa_opt_alloc(vec_size);                  // Vector scratch (u)

        // Batch workspace for block methods (SSA_BATCH_SIZE simultaneous transforms)
        size_t batch_size = 2 * fft_n * SSA_BATCH_SIZE * sizeof(double);
        ssa->ws_batch_u = (double *)ssa_opt_alloc(batch_size);
        ssa->ws_batch_v = (double *)ssa_opt_alloc(batch_size);
        ssa->ws_batch_out = (double *)ssa_opt_alloc(batch_size); // For NOT_INPLACE FFT

        if (!ssa->fft_x || !ssa->ws_fft1 || !ssa->ws_fft2 || !ssa->ws_real ||
            !ssa->ws_v || !ssa->ws_u || !ssa->ws_batch_u || !ssa->ws_batch_v || !ssa->ws_batch_out)
        {
            ssa_opt_free(ssa);
            return -1;
        }

        // -------------------------------------------------------------------------
        // Create MKL FFT descriptors
        // -------------------------------------------------------------------------
        MKL_LONG status;

        // Single FFT descriptor (in-place, for sequential matvec)
        status = DftiCreateDescriptor(&ssa->fft_handle, DFTI_DOUBLE, DFTI_COMPLEX, 1, fft_n);
        if (status != 0)
        {
            ssa_opt_free(ssa);
            return -1;
        }
        DftiSetValue(ssa->fft_handle, DFTI_PLACEMENT, DFTI_INPLACE);
        DftiSetValue(ssa->fft_handle, DFTI_BACKWARD_SCALE, 1.0 / fft_n); // Auto-scale IFFT
        status = DftiCommitDescriptor(ssa->fft_handle);
        if (status != 0)
        {
            ssa_opt_free(ssa);
            return -1;
        }

        // Batched FFT descriptor (NOT_INPLACE - faster for block operations)
        status = DftiCreateDescriptor(&ssa->fft_batch_c2c, DFTI_DOUBLE, DFTI_COMPLEX, 1, fft_n);
        if (status != 0)
        {
            ssa_opt_free(ssa);
            return -1;
        }
        DftiSetValue(ssa->fft_batch_c2c, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
        DftiSetValue(ssa->fft_batch_c2c, DFTI_BACKWARD_SCALE, 1.0 / fft_n);
        DftiSetValue(ssa->fft_batch_c2c, DFTI_NUMBER_OF_TRANSFORMS, (MKL_LONG)SSA_BATCH_SIZE);
        DftiSetValue(ssa->fft_batch_c2c, DFTI_INPUT_DISTANCE, (MKL_LONG)fft_n);  // Stride between inputs
        DftiSetValue(ssa->fft_batch_c2c, DFTI_OUTPUT_DISTANCE, (MKL_LONG)fft_n); // Stride between outputs
        status = DftiCommitDescriptor(ssa->fft_batch_c2c);
        if (status != 0)
        {
            ssa_opt_free(ssa);
            return -1;
        }

        // Batched FFT descriptor (INPLACE - for reconstruction)
        status = DftiCreateDescriptor(&ssa->fft_batch_inplace, DFTI_DOUBLE, DFTI_COMPLEX, 1, fft_n);
        if (status != 0)
        {
            ssa_opt_free(ssa);
            return -1;
        }
        DftiSetValue(ssa->fft_batch_inplace, DFTI_PLACEMENT, DFTI_INPLACE);
        DftiSetValue(ssa->fft_batch_inplace, DFTI_BACKWARD_SCALE, 1.0 / fft_n);
        DftiSetValue(ssa->fft_batch_inplace, DFTI_NUMBER_OF_TRANSFORMS, (MKL_LONG)SSA_BATCH_SIZE);
        DftiSetValue(ssa->fft_batch_inplace, DFTI_INPUT_DISTANCE, (MKL_LONG)fft_n);
        DftiSetValue(ssa->fft_batch_inplace, DFTI_OUTPUT_DISTANCE, (MKL_LONG)fft_n);
        status = DftiCommitDescriptor(ssa->fft_batch_inplace);
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

        // Precompute FFT(x)
        ssa_opt_fft_r2c(ssa, x, N, ssa->fft_x);

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
    //
    // Algorithm: Compute k singular triplets one at a time using power iteration
    // with deflation. For each component i:
    //   1. Initialize v randomly, orthogonalize against v₀..vᵢ₋₁
    //   2. Iterate: u = H@v, orthog against u₀..uᵢ₋₁, v = Hᵀ@u, orthog, normalize
    //   3. Converged when |vₙₑᵥ - vₒₗₐ| < tol
    //   4. Extract σᵢ = ‖H@v‖, store uᵢ, vᵢ
    //
    // Complexity: O(k × max_iter × N log N)
    // ============================================================================

    int ssa_opt_decompose(SSA_Opt *ssa, int k, int max_iter)
    {
        if (!ssa || !ssa->initialized || k < 1)
        {
            return -1;
        }

        // Invalidate any cached FFTs from previous decomposition
        ssa_opt_free_cached_ffts(ssa);

        int L = ssa->L;
        int K = ssa->K;

        k = ssa_opt_min(k, ssa_opt_min(L, K)); // Can't compute more than rank

        // Allocate result storage
        ssa->U = (double *)ssa_opt_alloc(L * k * sizeof(double));       // Left singular vectors
        ssa->V = (double *)ssa_opt_alloc(K * k * sizeof(double));       // Right singular vectors
        ssa->sigma = (double *)ssa_opt_alloc(k * sizeof(double));       // Singular values
        ssa->eigenvalues = (double *)ssa_opt_alloc(k * sizeof(double)); // σ² values

        if (!ssa->U || !ssa->V || !ssa->sigma || !ssa->eigenvalues)
        {
            return -1;
        }

        ssa->n_components = k;

        // Use preallocated workspace for iteration
        double *u = ssa->ws_u;
        double *v = ssa->ws_v;
        double *v_new = (double *)ssa_opt_alloc(K * sizeof(double));

        // Projection coefficients for GEMM-based orthogonalization
        ssa->ws_proj = (double *)ssa_opt_alloc(k * sizeof(double));
        if (!ssa->ws_proj || !v_new)
        {
            ssa_opt_free_ptr(v_new);
            return -1;
        }

        ssa->total_variance = 0.0;

        // -------------------------------------------------------------------------
        // Main loop: compute one singular triplet per iteration
        // -------------------------------------------------------------------------
        for (int comp = 0; comp < k; comp++)
        {
            // Random initialization using MKL RNG
            vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, ssa->rng, K, v, -0.5, 0.5);

            // Orthogonalize against previous V's using GEMM
            // v = v - V_prev @ (V_prevᵀ @ v)
            if (comp > 0)
            {
                cblas_dgemv(CblasColMajor, CblasTrans, K, comp,
                            1.0, ssa->V, K, v, 1, 0.0, ssa->ws_proj, 1);
                cblas_dgemv(CblasColMajor, CblasNoTrans, K, comp,
                            -1.0, ssa->V, K, ssa->ws_proj, 1, 1.0, v, 1);
            }
            ssa_opt_normalize(v, K);

            // Power iteration
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

                double diff = 0.0;
                for (int i = 0; i < K; i++)
                {
                    double d = fabs(v[i]) - fabs(v_new[i]);
                    diff += d * d;
                }

                ssa_opt_copy(v_new, v, K);

                if (sqrt(diff) < SSA_CONVERGENCE_TOL && iter > 10)
                    break;
            }

            // Final orthogonalization
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

            if (sigma > 1e-15)
            {
                ssa_opt_scal(v, K, 1.0 / sigma);
            }

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

        ssa_opt_free_ptr(v_new);

        ssa->decomposed = true;
        return 0;
    }

    // ----------------------------------------------------------------------------
    // Incremental Decomposition (Extend)
    // ----------------------------------------------------------------------------

    int ssa_opt_extend(SSA_Opt *ssa, int additional_k, int max_iter)
    {
        if (!ssa || !ssa->decomposed || additional_k < 1)
            return -1;

        ssa_opt_free_cached_ffts(ssa);

        int L = ssa->L;
        int K = ssa->K;
        int old_k = ssa->n_components;
        int new_k = old_k + additional_k;

        new_k = ssa_opt_min(new_k, ssa_opt_min(L, K));
        if (new_k <= old_k)
            return 0;

        // Reallocate result arrays
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

        // Reallocate projection workspace
        if (ssa->ws_proj)
        {
            ssa_opt_free_ptr(ssa->ws_proj);
        }
        ssa->ws_proj = (double *)ssa_opt_alloc(new_k * sizeof(double));
        if (!ssa->ws_proj)
            return -1;

        double *u = ssa->ws_u;
        double *v = ssa->ws_v;
        double *v_new = (double *)ssa_opt_alloc(K * sizeof(double));
        if (!v_new)
            return -1;

        // Compute additional components
        for (int comp = old_k; comp < new_k; comp++)
        {
            vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, ssa->rng, K, v, -0.5, 0.5);

            cblas_dgemv(CblasColMajor, CblasTrans, K, comp,
                        1.0, ssa->V, K, v, 1, 0.0, ssa->ws_proj, 1);
            cblas_dgemv(CblasColMajor, CblasNoTrans, K, comp,
                        -1.0, ssa->V, K, ssa->ws_proj, 1, 1.0, v, 1);
            ssa_opt_normalize(v, K);

            for (int iter = 0; iter < max_iter; iter++)
            {
                ssa_opt_hankel_matvec(ssa, v, u);

                cblas_dgemv(CblasColMajor, CblasTrans, L, comp,
                            1.0, ssa->U, L, u, 1, 0.0, ssa->ws_proj, 1);
                cblas_dgemv(CblasColMajor, CblasNoTrans, L, comp,
                            -1.0, ssa->U, L, ssa->ws_proj, 1, 1.0, u, 1);
                ssa_opt_normalize(u, L);

                ssa_opt_hankel_matvec_T(ssa, u, v_new);

                cblas_dgemv(CblasColMajor, CblasTrans, K, comp,
                            1.0, ssa->V, K, v_new, 1, 0.0, ssa->ws_proj, 1);
                cblas_dgemv(CblasColMajor, CblasNoTrans, K, comp,
                            -1.0, ssa->V, K, ssa->ws_proj, 1, 1.0, v_new, 1);
                ssa_opt_normalize(v_new, K);

                double diff = 0.0;
                for (int i = 0; i < K; i++)
                {
                    double d = fabs(v[i]) - fabs(v_new[i]);
                    diff += d * d;
                }
                ssa_opt_copy(v_new, v, K);

                if (sqrt(diff) < SSA_CONVERGENCE_TOL && iter > 10)
                    break;
            }

            // Final orthogonalization
            cblas_dgemv(CblasColMajor, CblasTrans, K, comp,
                        1.0, ssa->V, K, v, 1, 0.0, ssa->ws_proj, 1);
            cblas_dgemv(CblasColMajor, CblasNoTrans, K, comp,
                        -1.0, ssa->V, K, ssa->ws_proj, 1, 1.0, v, 1);
            ssa_opt_normalize(v, K);

            ssa_opt_hankel_matvec(ssa, v, u);

            cblas_dgemv(CblasColMajor, CblasTrans, L, comp,
                        1.0, ssa->U, L, u, 1, 0.0, ssa->ws_proj, 1);
            cblas_dgemv(CblasColMajor, CblasNoTrans, L, comp,
                        -1.0, ssa->U, L, ssa->ws_proj, 1, 1.0, u, 1);

            double sigma = ssa_opt_normalize(u, L);

            ssa_opt_hankel_matvec_T(ssa, u, v);

            cblas_dgemv(CblasColMajor, CblasTrans, K, comp,
                        1.0, ssa->V, K, v, 1, 0.0, ssa->ws_proj, 1);
            cblas_dgemv(CblasColMajor, CblasNoTrans, K, comp,
                        -1.0, ssa->V, K, ssa->ws_proj, 1, 1.0, v, 1);

            if (sigma > 1e-15)
            {
                ssa_opt_scal(v, K, 1.0 / sigma);
            }

            ssa_opt_copy(u, &ssa->U[comp * L], L);
            ssa_opt_copy(v, &ssa->V[comp * K], K);
            ssa->sigma[comp] = sigma;
            ssa->eigenvalues[comp] = sigma * sigma;
            ssa->total_variance += sigma * sigma;
        }

        // Sort all components
        for (int i = 0; i < new_k - 1; i++)
        {
            for (int j = i + 1; j < new_k; j++)
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

        // Fix sign convention for new components
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
        ssa_opt_free_ptr(v_new);

        return 0;
    }

    // ============================================================================
    // BLOCK POWER METHOD DECOMPOSITION
    //
    // Algorithm: Process b vectors simultaneously instead of one at a time.
    // Key optimizations over sequential method:
    //   1. Batched FFT operations (amortize descriptor overhead)
    //   2. GEMM-based orthogonalization (vs loop of dot/axpy)
    //   3. Periodic QR (every 5 iters vs every iter)
    //   4. Rayleigh-Ritz refinement for accurate singular values
    //
    // Complexity: O(max_iter × N log N) for all k components together
    // (vs O(k × max_iter × N log N) for sequential)
    //
    // Speedup: ~3x faster than sequential for k ≥ block_size
    // ============================================================================

    int ssa_opt_decompose_block(SSA_Opt *ssa, int k, int block_size, int max_iter)
    {
        if (!ssa || !ssa->initialized || k < 1 || block_size < 1)
        {
            return -1;
        }

        ssa_opt_free_cached_ffts(ssa);

        int L = ssa->L;
        int K = ssa->K;

        // Block size should match SSA_BATCH_SIZE for optimal FFT batching
        if (block_size <= 0)
        {
            block_size = SSA_BATCH_SIZE;
        }
        int b = ssa_opt_min(block_size, ssa_opt_min(k, ssa_opt_min(L, K)));
        k = ssa_opt_min(k, ssa_opt_min(L, K));

        // Allocate result storage
        ssa->U = (double *)ssa_opt_alloc(L * k * sizeof(double));
        ssa->V = (double *)ssa_opt_alloc(K * k * sizeof(double));
        ssa->sigma = (double *)ssa_opt_alloc(k * sizeof(double));
        ssa->eigenvalues = (double *)ssa_opt_alloc(k * sizeof(double));

        if (!ssa->U || !ssa->V || !ssa->sigma || !ssa->eigenvalues)
        {
            return -1;
        }

        ssa->n_components = k;
        ssa->total_variance = 0.0;

        // -------------------------------------------------------------------------
        // Allocate block workspace
        // -------------------------------------------------------------------------
        double *V_block = (double *)ssa_opt_alloc(K * b * sizeof(double));  // Right subspace
        double *U_block = (double *)ssa_opt_alloc(L * b * sizeof(double));  // Left subspace
        double *U_block2 = (double *)ssa_opt_alloc(L * b * sizeof(double)); // For Rayleigh-Ritz
        double *tau_u = (double *)ssa_opt_alloc(b * sizeof(double));        // QR Householder coeffs
        double *tau_v = (double *)ssa_opt_alloc(b * sizeof(double));

        // Rayleigh-Ritz workspace: M = Uᵀ @ H @ V is b×b dense matrix
        double *M = (double *)ssa_opt_alloc(b * b * sizeof(double));
        double *U_small = (double *)ssa_opt_alloc(b * b * sizeof(double));  // SVD left vectors
        double *Vt_small = (double *)ssa_opt_alloc(b * b * sizeof(double)); // SVD right vectors
        double *S_small = (double *)ssa_opt_alloc(b * sizeof(double));      // SVD singular values
        double *superb = (double *)ssa_opt_alloc(b * sizeof(double));       // LAPACK workspace

        // GEMM workspace for deflation orthogonalization
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

            // Initialize V_block randomly
            vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, ssa->rng, K * cur_b, V_block, -0.5, 0.5);

            // GEMM-based orthogonalization
            if (comp > 0)
            {
                cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                            comp, cur_b, K,
                            1.0, ssa->V, K, V_block, K,
                            0.0, work, comp);
                cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                            K, cur_b, comp,
                            -1.0, ssa->V, K, work, comp,
                            1.0, V_block, K);
            }

            // Initial QR
            LAPACKE_dgeqrf(LAPACK_COL_MAJOR, K, cur_b, V_block, K, tau_v);
            LAPACKE_dorgqr(LAPACK_COL_MAJOR, K, cur_b, cur_b, V_block, K, tau_v);

            const int QR_INTERVAL = 5;

            for (int iter = 0; iter < max_iter; iter++)
            {
                ssa_opt_hankel_matvec_block(ssa, V_block, U_block, cur_b);

                if (comp > 0)
                {
                    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                                comp, cur_b, L,
                                1.0, ssa->U, L, U_block, L,
                                0.0, work, comp);
                    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                L, cur_b, comp,
                                -1.0, ssa->U, L, work, comp,
                                1.0, U_block, L);
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
                                comp, cur_b, K,
                                1.0, ssa->V, K, V_block, K,
                                0.0, work, comp);
                    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                K, cur_b, comp,
                                -1.0, ssa->V, K, work, comp,
                                1.0, V_block, K);
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
                            comp, cur_b, L,
                            1.0, ssa->U, L, U_block2, L,
                            0.0, work, comp);
                cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                            L, cur_b, comp,
                            -1.0, ssa->U, L, work, comp,
                            1.0, U_block2, L);
            }

            cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                        cur_b, cur_b, L,
                        1.0, U_block, L, U_block2, L,
                        0.0, M, cur_b);

            int svd_info = LAPACKE_dgesvd(LAPACK_COL_MAJOR, 'A', 'A',
                                          cur_b, cur_b, M, cur_b,
                                          S_small, U_small, cur_b, Vt_small, cur_b,
                                          superb);
            if (svd_info != 0)
            {
                for (int i = 0; i < cur_b; i++)
                {
                    S_small[i] = cblas_dnrm2(L, &U_block2[i * L], 1);
                }
                memset(U_small, 0, cur_b * cur_b * sizeof(double));
                memset(Vt_small, 0, cur_b * cur_b * sizeof(double));
                for (int i = 0; i < cur_b; i++)
                {
                    U_small[i + i * cur_b] = 1.0;
                    Vt_small[i + i * cur_b] = 1.0;
                }
            }

            // Rotate U_block by U_small
            cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                        L, cur_b, cur_b,
                        1.0, U_block, L, U_small, cur_b,
                        0.0, U_block2, L);

            // Rotate V_block by V_small = Vt_small^T
            cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                        K, cur_b, cur_b,
                        1.0, V_block, K, Vt_small, cur_b,
                        0.0, work, K);

            // Copy to results
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
            {
                cblas_dcopy(L, &ssa->U[(comp + i) * L], 1, &U_block[i * L], 1);
            }

            ssa_opt_hankel_matvec_T_block(ssa, U_block, V_block, cur_b);

            if (comp > 0)
            {
                cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                            comp, cur_b, K,
                            1.0, ssa->V, K, V_block, K,
                            0.0, work, comp);
                cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                            K, cur_b, comp,
                            -1.0, ssa->V, K, work, comp,
                            1.0, V_block, K);
            }

            for (int i = 0; i < cur_b; i++)
            {
                double sigma = ssa->sigma[comp + i];
                double *v_col = &V_block[i * K];

                if (sigma > 1e-15)
                {
                    cblas_dscal(K, 1.0 / sigma, v_col, 1);
                }

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
    // RANDOMIZED SVD DECOMPOSITION (Halko-Martinsson-Tropp)
    //
    // Algorithm: Only 2 passes over the data using random projection.
    //   Pass 1 (Range Finding):
    //     Ω = randn(K, k+p)      — Gaussian random test matrix
    //     Y = H × Ω              — Sample column space (batched matvec)
    //     Q = orth(Y)            — Orthonormal basis via QR
    //
    //   Pass 2 (Projection):
    //     B = Hᵀ × Q             — Project onto basis (batched matvec transpose)
    //     SVD(B) → Ũ, Σ, Ṽ      — Small dense SVD (MKL dgesdd)
    //     U = Q × Ṽ, V = Ũ       — Recover full vectors
    //
    // Complexity: O(2 × (k+p) × N log N) for all k components
    // Speedup: ~15-25x faster than sequential method
    //
    // Reference: Halko, Martinsson, Tropp. "Finding structure with randomness:
    //            Probabilistic algorithms for constructing approximate matrix
    //            decompositions." SIAM Review, 2011. (Algorithm 5.1)
    // ============================================================================

    int ssa_opt_decompose_randomized(SSA_Opt *ssa, int k, int oversampling)
    {
        if (!ssa || !ssa->initialized || k < 1)
        {
            return -1;
        }

        ssa_opt_free_cached_ffts(ssa);

        int L = ssa->L;
        int K = ssa->K;

        // Oversampling p: extra columns for numerical stability
        // Larger p = more accurate but slower; p=8 is typical
        int p = (oversampling <= 0) ? 8 : oversampling;
        int kp = k + p;

        kp = ssa_opt_min(kp, ssa_opt_min(L, K));
        k = ssa_opt_min(k, kp);

        // Allocate result storage
        ssa->U = (double *)ssa_opt_alloc(L * k * sizeof(double));
        ssa->V = (double *)ssa_opt_alloc(K * k * sizeof(double));
        ssa->sigma = (double *)ssa_opt_alloc(k * sizeof(double));
        ssa->eigenvalues = (double *)ssa_opt_alloc(k * sizeof(double));

        if (!ssa->U || !ssa->V || !ssa->sigma || !ssa->eigenvalues)
        {
            return -1;
        }

        ssa->n_components = k;
        ssa->total_variance = 0.0;

        // -------------------------------------------------------------------------
        // Allocate workspace for randomized SVD
        // -------------------------------------------------------------------------
        double *Omega = (double *)ssa_opt_alloc(K * kp * sizeof(double)); // Random test matrix
        double *Y = (double *)ssa_opt_alloc(L * kp * sizeof(double));     // Y = H × Ω
        double *Q = (double *)ssa_opt_alloc(L * kp * sizeof(double));     // Orthonormal basis
        double *B = (double *)ssa_opt_alloc(K * kp * sizeof(double));     // B = Hᵀ × Q
        double *tau = (double *)ssa_opt_alloc(kp * sizeof(double));       // QR Householder coeffs

        // Dense SVD workspace (divide-and-conquer via dgesdd)
        double *U_svd = (double *)ssa_opt_alloc(K * kp * sizeof(double));   // SVD left vectors
        double *Vt_svd = (double *)ssa_opt_alloc(kp * kp * sizeof(double)); // SVD right vectors
        double *S_svd = (double *)ssa_opt_alloc(kp * sizeof(double));       // Singular values

        // Query optimal LAPACK workspace size
        double work_query;
        int *iwork = (int *)ssa_opt_alloc(8 * kp * sizeof(int));
        int lwork = -1;
        int info;

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

        // -------------------------------------------------------------------------
        // Step 1: Generate random Gaussian test matrix Ω
        // -------------------------------------------------------------------------
        vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER, ssa->rng, K * kp, Omega, 0.0, 1.0);

        // -------------------------------------------------------------------------
        // Step 2: Range sampling Y = H × Ω (batched FFT matvec)
        // -------------------------------------------------------------------------
        ssa_opt_hankel_matvec_block(ssa, Omega, Y, kp);

        // -------------------------------------------------------------------------
        // Step 3: Orthonormalize Q = orth(Y) via QR factorization
        // -------------------------------------------------------------------------
        cblas_dcopy(L * kp, Y, 1, Q, 1);
        LAPACKE_dgeqrf(LAPACK_COL_MAJOR, L, kp, Q, L, tau);
        LAPACKE_dorgqr(LAPACK_COL_MAJOR, L, kp, kp, Q, L, tau);

        // Step 4: B = H^T @ Q
        ssa_opt_hankel_matvec_T_block(ssa, Q, B, kp);

        // Step 5: SVD of B
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

        // -------------------------------------------------------------------------
        // Step 6: Recover U and V from SVD of B
        //
        // From B = U_svd × Σ × Vt_svd and B = Hᵀ × Q:
        //   H ≈ Q × Vt_svdᵀ × Σ × U_svdᵀ
        //
        // So: U_final = Q × Vt_svdᵀ (first k columns)
        //     V_final = U_svd (first k columns)
        //     σ_final = S_svd (first k values)
        // -------------------------------------------------------------------------
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                    L, k, kp,
                    1.0, Q, L, Vt_svd, kp,
                    0.0, ssa->U, L);

        for (int i = 0; i < k; i++)
        {
            cblas_dcopy(K, &U_svd[i * K], 1, &ssa->V[i * K], 1);
        }

        // Copy singular values and compute eigenvalues (σ²)
        for (int i = 0; i < k; i++)
        {
            ssa->sigma[i] = S_svd[i];
            ssa->eigenvalues[i] = S_svd[i] * S_svd[i];
            ssa->total_variance += ssa->eigenvalues[i];
        }

        // Fix sign convention: make sum(U) positive for reproducibility
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

        // Cleanup workspace
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

        ssa->decomposed = true;
        return 0;
    }

    // ============================================================================
    // FFT CACHING FOR RECONSTRUCTION ACCELERATION
    //
    // Precomputes FFT(σᵢ × uᵢ) and FFT(vᵢ) for all components. Subsequent
    // reconstruct() calls skip forward FFT step, saving 2k FFT operations.
    // Typical speedup: 2-3x for workflows with multiple reconstructions.
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
        {
            return -1;
        }

        ssa_opt_free_cached_ffts(ssa);

        int L = ssa->L;
        int K = ssa->K;
        int fft_n = ssa->fft_len;
        int k = ssa->n_components;
        size_t fft_size = 2 * fft_n;

        ssa->U_fft = (double *)ssa_opt_alloc(fft_size * k * sizeof(double));
        ssa->V_fft = (double *)ssa_opt_alloc(fft_size * k * sizeof(double));

        if (!ssa->U_fft || !ssa->V_fft)
        {
            ssa_opt_free_cached_ffts(ssa);
            return -1;
        }

        for (int i = 0; i < k; i++)
        {
            double sigma = ssa->sigma[i];
            const double *u_vec = &ssa->U[i * L];
            double *dst = &ssa->U_fft[i * fft_size];

            memset(dst, 0, fft_size * sizeof(double));
            for (int j = 0; j < L; j++)
            {
                dst[2 * j] = sigma * u_vec[j];
            }
            DftiComputeForward(ssa->fft_handle, dst);
        }

        for (int i = 0; i < k; i++)
        {
            const double *v_vec = &ssa->V[i * K];
            double *dst = &ssa->V_fft[i * fft_size];

            memset(dst, 0, fft_size * sizeof(double));
            for (int j = 0; j < K; j++)
            {
                dst[2 * j] = v_vec[j];
            }
            DftiComputeForward(ssa->fft_handle, dst);
        }

        ssa->fft_cached = true;
        return 0;
    }

    // ============================================================================
    // SIGNAL RECONSTRUCTION
    //
    // Reconstructs time series from selected SSA components. Each component i
    // contributes σᵢ × (uᵢ ⊗ vᵢ) where ⊗ is outer product forming an L×K matrix.
    // The sum is converted back to a 1D signal via diagonal averaging (Hankelization).
    //
    // Key insight: The outer product uᵢ ⊗ vᵢ is equivalent to convolution:
    //   (u ⊗ v)[row i, col j] = u[i] × v[j]
    //   Diagonal sum: Σ u[i] × v[t-i] = conv(u, v)[t]
    //
    // This allows O(N log N) reconstruction per component via FFT, avoiding
    // the O(L×K) explicit matrix formation.
    //
    // Two paths:
    //   1. Fast path: If FFTs cached, skip forward FFT (saves 2k FFTs)
    //   2. Standard path: Compute FFT(σu), FFT(v), multiply, IFFT
    // ============================================================================

    int ssa_opt_reconstruct(const SSA_Opt *ssa, const int *group, int n_group, double *output)
    {
        if (!ssa || !ssa->decomposed || !group || !output || n_group < 1)
        {
            return -1;
        }

        int N = ssa->N;
        int L = ssa->L;
        int K = ssa->K;
        int fft_n = ssa->fft_len;
        size_t fft_size = 2 * fft_n; // Interleaved complex

        ssa_opt_zero(output, N);

        // Cast away const for workspace access (workspace not logically part of result)
        SSA_Opt *ssa_mut = (SSA_Opt *)ssa;
        double *ws_u = ssa_mut->ws_batch_u;
        double *ws_v = ssa_mut->ws_batch_v;
        size_t stride = fft_size;

        if (ssa->fft_cached && ssa->U_fft && ssa->V_fft)
        {
            // =====================================================================
            // FAST PATH: Use precomputed FFTs (skip forward FFT entirely)
            // =====================================================================
            int g = 0;
            while (g < n_group)
            {
                // Gather batch of valid component indices
                int batch_count = 0;
                int batch_indices[SSA_BATCH_SIZE];

                while (batch_count < SSA_BATCH_SIZE && g < n_group)
                {
                    int idx = group[g];
                    if (idx >= 0 && idx < ssa->n_components)
                    {
                        batch_indices[batch_count++] = idx;
                    }
                    g++;
                }

                if (batch_count == 0)
                    continue;

                // Element-wise complex multiply cached FFTs
                for (int b = 0; b < batch_count; b++)
                {
                    int idx = batch_indices[b];
                    const double *u_fft_cached = &ssa->U_fft[idx * fft_size];
                    const double *v_fft_cached = &ssa->V_fft[idx * fft_size];
                    double *dst = ws_u + b * stride;

                    vzMul(fft_n, (const MKL_Complex16 *)u_fft_cached,
                          (const MKL_Complex16 *)v_fft_cached,
                          (MKL_Complex16 *)dst);
                }

                if (batch_count < SSA_BATCH_SIZE)
                {
                    memset(ws_u + batch_count * stride, 0,
                           (SSA_BATCH_SIZE - batch_count) * stride * sizeof(double));
                }

                DftiComputeBackward(ssa_mut->fft_batch_inplace, ws_u);

                for (int b = 0; b < batch_count; b++)
                {
                    double *conv = ws_u + b * stride;
                    cblas_daxpy(N, 1.0, conv, 2, output, 1);
                }
            }
        }
        else
        {
            // Standard path
            int g = 0;
            while (g < n_group)
            {
                int batch_count = 0;
                int batch_indices[SSA_BATCH_SIZE];

                while (batch_count < SSA_BATCH_SIZE && g < n_group)
                {
                    int idx = group[g];
                    if (idx >= 0 && idx < ssa->n_components)
                    {
                        batch_indices[batch_count++] = idx;
                    }
                    g++;
                }

                if (batch_count == 0)
                    continue;

                memset(ws_u, 0, SSA_BATCH_SIZE * stride * sizeof(double));
                memset(ws_v, 0, SSA_BATCH_SIZE * stride * sizeof(double));

                for (int b = 0; b < batch_count; b++)
                {
                    int idx = batch_indices[b];
                    double sigma = ssa->sigma[idx];
                    const double *u_vec = &ssa->U[idx * L];
                    double *dst = ws_u + b * stride;

                    for (int i = 0; i < L; i++)
                    {
                        dst[2 * i] = sigma * u_vec[i];
                    }
                }

                for (int b = 0; b < batch_count; b++)
                {
                    int idx = batch_indices[b];
                    const double *v_vec = &ssa->V[idx * K];
                    double *dst = ws_v + b * stride;

                    for (int i = 0; i < K; i++)
                    {
                        dst[2 * i] = v_vec[i];
                    }
                }

                DftiComputeForward(ssa_mut->fft_batch_inplace, ws_u);
                DftiComputeForward(ssa_mut->fft_batch_inplace, ws_v);

                for (int b = 0; b < batch_count; b++)
                {
                    double *u_fft = ws_u + b * stride;
                    double *v_fft = ws_v + b * stride;
                    vzMul(fft_n, (const MKL_Complex16 *)u_fft, (const MKL_Complex16 *)v_fft,
                          (MKL_Complex16 *)u_fft);
                }

                DftiComputeBackward(ssa_mut->fft_batch_inplace, ws_u);

                for (int b = 0; b < batch_count; b++)
                {
                    double *conv = ws_u + b * stride;
                    cblas_daxpy(N, 1.0, conv, 2, output, 1);
                }
            }
        }

        // Diagonal averaging
        // Apply diagonal averaging weights using precomputed 1/count
        vdMul(N, output, ssa->inv_diag_count, output);

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

        // Compute diagonal averaging weights
        double *weights = (double *)ssa_opt_alloc(N * sizeof(double));
        if (!weights)
            return -1;

        for (int t = 0; t < N; t++)
        {
            weights[t] = (double)ssa_opt_min(ssa_opt_min(t + 1, L),
                                             ssa_opt_min(K, N - t));
        }

        double *reconstructed = (double *)ssa_opt_alloc(N * n * sizeof(double));
        if (!reconstructed)
        {
            ssa_opt_free_ptr(weights);
            return -1;
        }

        for (int i = 0; i < n; i++)
        {
            int group[] = {i};
            ssa_opt_reconstruct(ssa, group, 1, &reconstructed[i * N]);
        }

        double *norms = (double *)ssa_opt_alloc(n * sizeof(double));
        if (!norms)
        {
            ssa_opt_free_ptr(weights);
            ssa_opt_free_ptr(reconstructed);
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

        ssa_opt_free_ptr(weights);
        ssa_opt_free_ptr(reconstructed);
        ssa_opt_free_ptr(norms);

        return 0;
    }

    double ssa_opt_wcorr_pair(const SSA_Opt *ssa, int i, int j)
    {
        if (!ssa || !ssa->decomposed ||
            i < 0 || i >= ssa->n_components ||
            j < 0 || j >= ssa->n_components)
            return 0.0;

        int N = ssa->N;
        int L = ssa->L;
        int K = ssa->K;

        double *x_i = (double *)ssa_opt_alloc(N * sizeof(double));
        double *x_j = (double *)ssa_opt_alloc(N * sizeof(double));

        if (!x_i || !x_j)
        {
            ssa_opt_free_ptr(x_i);
            ssa_opt_free_ptr(x_j);
            return 0.0;
        }

        int group_i[] = {i};
        int group_j[] = {j};
        ssa_opt_reconstruct(ssa, group_i, 1, x_i);
        ssa_opt_reconstruct(ssa, group_j, 1, x_j);

        double inner = 0.0, norm_i = 0.0, norm_j = 0.0;

        for (int t = 0; t < N; t++)
        {
            double w = (double)ssa_opt_min(ssa_opt_min(t + 1, L),
                                           ssa_opt_min(K, N - t));
            inner += w * x_i[t] * x_j[t];
            norm_i += w * x_i[t] * x_i[t];
            norm_j += w * x_j[t] * x_j[t];
        }

        ssa_opt_free_ptr(x_i);
        ssa_opt_free_ptr(x_j);

        double denom = sqrt(norm_i) * sqrt(norm_j);
        return (denom > 1e-15) ? inner / denom : 0.0;
    }

    // ----------------------------------------------------------------------------
    // Component Statistics
    // ----------------------------------------------------------------------------

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

        if (!stats->singular_values || !stats->log_sv || !stats->gaps ||
            !stats->cumulative_var || !stats->second_diff)
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
        {
            double d2 = stats->log_sv[i - 1] - 2.0 * stats->log_sv[i] + stats->log_sv[i + 1];
            stats->second_diff[i] = d2;
        }

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

        if (sv_tol <= 0)
            sv_tol = 0.1;
        if (wcorr_thresh <= 0)
            wcorr_thresh = 0.5;

        int n = ssa->n_components;
        int n_pairs = 0;

        bool *used = (bool *)calloc(n, sizeof(bool));
        if (!used)
            return 0;

        for (int i = 0; i < n - 1 && n_pairs < max_pairs; i++)
        {
            if (used[i])
                continue;

            for (int j = i + 1; j < n && n_pairs < max_pairs; j++)
            {
                if (used[j])
                    continue;

                double sv_ratio = ssa->sigma[j] / (ssa->sigma[i] + 1e-300);
                if (fabs(1.0 - sv_ratio) > sv_tol)
                    continue;

                double wcorr = fabs(ssa_opt_wcorr_pair(ssa, i, j));
                if (wcorr < wcorr_thresh)
                    continue;

                pairs[2 * n_pairs] = i;
                pairs[2 * n_pairs + 1] = j;
                used[i] = true;
                used[j] = true;
                n_pairs++;
                break;
            }
        }

        free(used);
        return n_pairs;
    }

    // ----------------------------------------------------------------------------
    // Convenience Functions
    // ----------------------------------------------------------------------------

    void ssa_opt_free(SSA_Opt *ssa)
    {
        if (!ssa)
            return;

        if (ssa->fft_handle)
            DftiFreeDescriptor(&ssa->fft_handle);
        if (ssa->fft_batch_c2c)
            DftiFreeDescriptor(&ssa->fft_batch_c2c);
        if (ssa->fft_batch_inplace)
            DftiFreeDescriptor(&ssa->fft_batch_inplace);
        if (ssa->rng)
            vslDeleteStream(&ssa->rng);

        ssa_opt_free_ptr(ssa->fft_x);
        ssa_opt_free_ptr(ssa->ws_fft1);
        ssa_opt_free_ptr(ssa->ws_fft2);
        ssa_opt_free_ptr(ssa->ws_real);
        ssa_opt_free_ptr(ssa->ws_v);
        ssa_opt_free_ptr(ssa->ws_u);
        ssa_opt_free_ptr(ssa->ws_proj);
        ssa_opt_free_ptr(ssa->ws_batch_u);
        ssa_opt_free_ptr(ssa->ws_batch_v);
        ssa_opt_free_ptr(ssa->ws_batch_out);
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

#endif // SSA_OPT_IMPLEMENTATION

#ifdef __cplusplus
}
#endif

#endif // SSA_OPT_H