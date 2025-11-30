/*
 * ============================================================================
 * SSA-OPT: High-Performance Singular Spectrum Analysis
 * ============================================================================
 *
 * WHAT IS SSA?
 *   Singular Spectrum Analysis decomposes a time series into trend, periodic
 *   components, and noise by embedding it into a Hankel matrix and computing
 *   its SVD. This implementation avoids forming the explicit matrix.
 *
 * WHAT THIS IMPLEMENTATION PROVIDES:
 *   - O(N log N) Hankel matvec via FFT convolution (vs O(N²) naive)
 *   - Three decomposition methods:
 *       1. Sequential power iteration (baseline)
 *       2. Block power iteration with Rayleigh-Ritz (Phase 2)
 *       3. Randomized SVD (Phase 3)
 *   - Batched FFT operations for throughput
 *   - MKL-optimized BLAS/LAPACK throughout
 *
 * PERFORMANCE SUMMARY (N=5000, L=2000, k=32):
 *   Sequential:  ~550 ms  (1.0x)
 *   Block:       ~195 ms  (2.8x)
 *   Randomized:  ~20-40ms (15-25x)
 *
 * USAGE:
 *   SSA_Opt ssa = {0};
 *   ssa_opt_init(&ssa, signal, N, L);
 *   ssa_opt_decompose_randomized(&ssa, k, 8);  // or _block, or _decompose
 *   ssa_opt_reconstruct(&ssa, components, n_components, output);
 *   ssa_opt_free(&ssa);
 *
 * BUILD (Windows + MKL):
 *   cl /O2 /DSSA_USE_MKL /DSSA_OPT_IMPLEMENTATION your_code.c
 *      /I"%MKLROOT%\include" /link /LIBPATH:"%MKLROOT%\lib"
 *      mkl_intel_lp64.lib mkl_intel_thread.lib mkl_core.lib
 *
 * KEY INSIGHT - WHY FFT?
 *   The Hankel matrix H has structure: H[i,j] = x[i+j]. This means
 *   y = H @ v is equivalent to convolution: y[i] = Σⱼ x[i+j]·v[j].
 *   Convolution theorem converts O(N²) multiply to O(N log N) FFT.
 *
 * MKL DOCUMENTATION REFERENCES:
 *   - DFTI: Intel MKL Developer Reference > FFT Functions
 *   - BLAS: Intel MKL Developer Reference > BLAS Routines
 *   - LAPACK: Intel MKL Developer Reference > LAPACK Routines
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

// Platform-specific intrinsics
#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__GNUC__) || defined(__clang__)
#include <xmmintrin.h>
#endif

#ifdef SSA_USE_MKL
#include <mkl.h>
#include <mkl_dfti.h>
#include <mkl_vml.h>
#include <mkl_vsl.h>     // Vector Statistical Library for RNG
#include <mkl_lapacke.h> // LAPACK for QR factorization (Phase 2)
#endif

#ifdef __cplusplus
extern "C"
{
#endif

// ============================================================================
// Configuration
// ============================================================================

// Convergence tolerance for power iteration
// Iteration stops when |σ_new - σ_old| / σ_old < TOL
#ifndef SSA_CONVERGENCE_TOL
#define SSA_CONVERGENCE_TOL 1e-12
#endif

// Memory alignment for SIMD (AVX-512 = 64 bytes)
#define SSA_ALIGN 64

// Number of vectors processed simultaneously in batched FFT operations.
// Larger = better throughput but more memory. 32 is optimal for most cases.
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
     *
     * LIFECYCLE:
     *   1. Zero-initialize: SSA_Opt ssa = {0};
     *   2. ssa_opt_init() — allocates workspace, precomputes FFT(x)
     *   3. ssa_opt_decompose*() — computes SVD, fills U, V, sigma
     *   4. ssa_opt_reconstruct() — reconstructs signal from components
     *   5. ssa_opt_free() — releases all memory
     *
     * WORKSPACE STRATEGY:
     *   All workspace is allocated once in init(). Hot paths (matvec, iteration)
     *   perform zero allocations. This is critical for performance.
     */
    typedef struct
    {
        // -------------------------------------------------------------------------
        // Dimensions (set in init, immutable thereafter)
        // -------------------------------------------------------------------------
        int N;       // Original series length
        int L;       // Window length (embedding dimension), typically N/3 to N/2
        int K;       // K = N - L + 1, number of lagged vectors
        int fft_len; // FFT length, next power of 2 ≥ N + K - 1

#ifdef SSA_USE_MKL
        // -------------------------------------------------------------------------
        // MKL FFT Descriptors
        //
        // MKL requires pre-configured "descriptor" objects for FFT. We create
        // multiple descriptors for different use cases:
        //   - fft_handle: single transform (sequential matvec)
        //   - fft_batch_c2c: batched NOT_INPLACE (block matvec) — faster
        //   - fft_batch_inplace: batched INPLACE (reconstruction)
        //
        // See Intel MKL DFTI documentation for descriptor semantics.
        // -------------------------------------------------------------------------
        DFTI_DESCRIPTOR_HANDLE fft_handle;        // Single C2C FFT (in-place)
        DFTI_DESCRIPTOR_HANDLE fft_batch_c2c;     // Batched C2C (not-in-place, faster)
        DFTI_DESCRIPTOR_HANDLE fft_batch_inplace; // Batched C2C (in-place, for reconstruction)

        // -------------------------------------------------------------------------
        // Random Number Generator (for randomized SVD and initialization)
        // -------------------------------------------------------------------------
        VSLStreamStatePtr rng;

        // -------------------------------------------------------------------------
        // Precomputed Data
        //
        // FFT(x) is computed once in init() and reused for all matvecs.
        // This saves one FFT call per iteration.
        // -------------------------------------------------------------------------
        double *fft_x; // FFT of input signal, interleaved complex, length 2×fft_len

        // -------------------------------------------------------------------------
        // Workspace Buffers (pre-allocated, zero allocations in hot paths)
        //
        // Naming: ws_* = workspace
        // All complex buffers use interleaved format: [re, im, re, im, ...]
        // -------------------------------------------------------------------------
        double *ws_fft1; // FFT scratch buffer 1, length 2×fft_len
        double *ws_fft2; // FFT scratch buffer 2, length 2×fft_len
        double *ws_real; // Real-valued scratch, length fft_len
        double *ws_v;    // Vector scratch for sequential iteration, length max(L,K)
        double *ws_u;    // Vector scratch for sequential iteration, length max(L,K)
        double *ws_proj; // Projection coefficients for orthogonalization, length k

        // Batch workspace for block methods (sized for SSA_BATCH_SIZE transforms)
        double *ws_batch_u;   // Batch buffer, length 2×fft_len×SSA_BATCH_SIZE
        double *ws_batch_v;   // Batch buffer, length 2×fft_len×SSA_BATCH_SIZE
        double *ws_batch_out; // Output buffer for NOT_INPLACE FFT
#else
    // Non-MKL fallback (minimal workspace)
    double *fft_x;
    double *ws_fft1;
    double *ws_fft2;
#endif

        // -------------------------------------------------------------------------
        // Results (populated by decompose, used by reconstruct)
        // -------------------------------------------------------------------------
        double *U;           // Left singular vectors, L × k, column-major
        double *V;           // Right singular vectors, K × k, column-major
        double *sigma;       // Singular values σ₁ ≥ σ₂ ≥ ... ≥ σₖ
        double *eigenvalues; // Squared singular values (σᵢ²), for variance computation
        int n_components;    // Number of computed components (≤ k requested)

        // -------------------------------------------------------------------------
        // Reconstruction Optimization
        // -------------------------------------------------------------------------
        double *inv_diag_count; // Precomputed 1/count[t] for diagonal averaging, length N
        double *U_fft;          // Cached FFT of U vectors, 2×fft_len×k (NULL if not cached)
        double *V_fft;          // Cached FFT of V vectors, 2×fft_len×k (NULL if not cached)
        bool fft_cached;        // True if U_fft/V_fft are populated

        // -------------------------------------------------------------------------
        // State Flags
        // -------------------------------------------------------------------------
        bool initialized;      // True after successful init()
        bool decomposed;       // True after successful decompose*()
        double total_variance; // Σᵢ σᵢ², for computing explained variance ratio
    } SSA_Opt;

    // ============================================================================
    // Component Statistics (for automatic selection)
    // ============================================================================

    typedef struct
    {
        int n;                   // Number of components analyzed
        double *singular_values; // Copy of σ_i
        double *log_sv;          // log(σ_i) for scree plot
        double *gaps;            // σ_i / σ_{i+1} for i = 0..n-2
        double *cumulative_var;  // Cumulative explained variance ratio
        double *second_diff;     // Second difference of log(σ) for elbow detection

        // Automatic selection results
        int suggested_signal; // Components 0..suggested_signal-1 are "signal"
        double gap_threshold; // Gap ratio at suggested cutoff
    } SSA_ComponentStats;

    // ============================================================================
    // Public API
    // ============================================================================

    /**
     * @brief Initialize SSA context with input signal.
     *
     * Allocates all workspace, precomputes FFT(x), sets up MKL descriptors.
     * After this call, no further allocations occur in hot paths.
     *
     * @param ssa   Pointer to zero-initialized SSA_Opt struct
     * @param x     Input time series, length N
     * @param N     Length of input signal
     * @param L     Window length (embedding dimension). Typical choice: N/3 to N/2.
     *              Larger L = better frequency resolution, smaller L = better trend.
     * @return      0 on success, -1 on error
     */
    int ssa_opt_init(SSA_Opt *ssa, const double *x, int N, int L);

    /**
     * @brief Compute SVD via sequential power iteration (baseline method).
     *
     * Computes k singular triplets one at a time using power iteration with
     * deflation. Each component requires O(max_iter × N log N) operations.
     *
     * @param ssa       Initialized SSA context
     * @param k         Number of singular triplets to compute
     * @param max_iter  Maximum iterations per component (100-200 typical)
     * @return          0 on success, -1 on error
     */
    int ssa_opt_decompose(SSA_Opt *ssa, int k, int max_iter);

    /**
     * @brief Compute SVD via block power iteration (Phase 2).
     *
     * Computes k singular triplets in blocks of size b. Uses Rayleigh-Ritz
     * refinement for accurate singular values and periodic QR for stability.
     * ~3x faster than sequential for k ≥ block_size.
     *
     * @param ssa        Initialized SSA context
     * @param k          Number of singular triplets to compute
     * @param block_size Block size (0 = use SSA_BATCH_SIZE). Should match
     *                   SSA_BATCH_SIZE for optimal FFT batching.
     * @param max_iter   Maximum iterations per block (100 typical)
     * @return           0 on success, -1 on error
     */
    int ssa_opt_decompose_block(SSA_Opt *ssa, int k, int block_size, int max_iter);

    /**
     * @brief Compute SVD via randomized algorithm (Phase 3).
     *
     * Uses Halko-Martinsson-Tropp randomized SVD: only 2 passes over the data.
     * Fastest method for most cases, especially when k << min(L, K).
     *
     * Algorithm:
     *   1. Y = X @ Ω          (random sampling, Ω is K×(k+p) Gaussian)
     *   2. Q = orth(Y)        (QR factorization)
     *   3. B = Xᵀ @ Q         (projection)
     *   4. SVD(B) → U, Σ, V   (small dense SVD)
     *
     * @param ssa          Initialized SSA context
     * @param k            Number of singular triplets to compute
     * @param oversampling Oversampling parameter p (typically 5-10, use 0 for default=8)
     *                     Larger p = more accurate but slower
     * @return             0 on success, -1 on error
     */
    int ssa_opt_decompose_randomized(SSA_Opt *ssa, int k, int oversampling);

    /**
     * @brief Extend existing decomposition with additional components.
     *
     * Adds more components to an already-decomposed SSA context without
     * recomputing existing ones. Uses power iteration with orthogonalization
     * against all existing components.
     *
     * @param ssa          Already decomposed SSA context
     * @param additional_k Number of additional components to compute
     * @param max_iter     Maximum iterations per component
     * @return             0 on success, -1 on error
     */
    int ssa_opt_extend(SSA_Opt *ssa, int additional_k, int max_iter);

    /**
     * @brief Reconstruct signal from selected components.
     *
     * Computes: output = Σᵢ∈group σᵢ · (uᵢ ⊗ vᵢ) averaged along anti-diagonals.
     * The outer product uᵢ ⊗ vᵢ is computed implicitly via FFT convolution.
     *
     * @param ssa      Decomposed SSA context
     * @param group    Array of component indices to include (0-based)
     * @param n_group  Number of components in group
     * @param output   Output buffer, length N (will be overwritten)
     * @return         0 on success, -1 on error
     *
     * @example
     *   // Reconstruct trend (first component)
     *   int trend[] = {0};
     *   ssa_opt_reconstruct(&ssa, trend, 1, trend_output);
     *
     *   // Reconstruct periodic (components 1-4)
     *   int periodic[] = {1, 2, 3, 4};
     *   ssa_opt_reconstruct(&ssa, periodic, 4, periodic_output);
     */
    int ssa_opt_reconstruct(const SSA_Opt *ssa, const int *group, int n_group, double *output);

    /**
     * @brief Cache FFTs of U and V vectors for faster reconstruction.
     *
     * Precomputes FFT(u_i) and FFT(v_i) for all components. Subsequent calls
     * to ssa_opt_reconstruct() will skip the forward FFT step, saving k FFTs
     * per reconstruction call.
     *
     * Typical speedup: 2-3x for workflows that call reconstruct multiple times
     * with different groupings (e.g., extract trend, then periodic, then noise).
     *
     * Memory cost: ~4MB for k=32, N=5000 (2 × fft_len × k × 8 bytes)
     *
     * @param ssa  Decomposed SSA context
     * @return     0 on success, -1 on error
     */
    int ssa_opt_cache_ffts(SSA_Opt *ssa);

    /**
     * @brief Free cached FFTs (optional, also freed by ssa_opt_free).
     * @param ssa  SSA context with cached FFTs
     */
    void ssa_opt_free_cached_ffts(SSA_Opt *ssa);

    /**
     * @brief Free all memory associated with SSA context.
     * @param ssa  SSA context to free (safe to call on partially initialized or NULL)
     */
    void ssa_opt_free(SSA_Opt *ssa);

    // ============================================================================
    // Analysis API
    // ============================================================================

    /**
     * @brief Compute W-correlation matrix between SSA components.
     *
     * W-correlation uses diagonal averaging weights to measure similarity
     * between reconstructed components. High |ρ_W| indicates components
     * that should be grouped together.
     *
     * @param ssa  Decomposed SSA context
     * @param W    Output matrix, n_components × n_components, row-major
     *             W[i*n + j] = ρ_W(component_i, component_j)
     * @return     0 on success, -1 on error
     */
    int ssa_opt_wcorr_matrix(const SSA_Opt *ssa, double *W);

    /**
     * @brief Compute W-correlation between two specific components.
     * @return W-correlation value, or 0.0 on error
     */
    double ssa_opt_wcorr_pair(const SSA_Opt *ssa, int i, int j);

    /**
     * @brief Analyze singular value spectrum for component selection.
     *
     * Computes gap ratios, cumulative variance, and suggests signal/noise cutoff.
     *
     * @param ssa    Decomposed SSA context
     * @param stats  Output statistics structure
     * @return       0 on success, -1 on error
     */
    int ssa_opt_component_stats(const SSA_Opt *ssa, SSA_ComponentStats *stats);

    /**
     * @brief Free component statistics structure.
     */
    void ssa_opt_component_stats_free(SSA_ComponentStats *stats);

    /**
     * @brief Find components that likely form periodic pairs.
     *
     * @param ssa          Decomposed SSA context
     * @param pairs        Output: pairs[2*j], pairs[2*j+1] are paired indices
     * @param max_pairs    Maximum pairs to find
     * @param sv_tol       Singular value tolerance (0 = default 0.1)
     * @param wcorr_thresh W-correlation threshold (0 = default 0.5)
     * @return             Number of pairs found
     */
    int ssa_opt_find_periodic_pairs(const SSA_Opt *ssa, int *pairs, int max_pairs,
                                    double sv_tol, double wcorr_thresh);

    // Convenience functions
    int ssa_opt_get_trend(const SSA_Opt *ssa, double *output);                  ///< Reconstruct component 0
    int ssa_opt_get_noise(const SSA_Opt *ssa, int noise_start, double *output); ///< Reconstruct components [noise_start, end]
    double ssa_opt_variance_explained(const SSA_Opt *ssa, int start, int end);  ///< Compute Σᵢ σᵢ² / total

    // ============================================================================
    // Implementation
    // ============================================================================

#ifdef SSA_OPT_IMPLEMENTATION

    // ----------------------------------------------------------------------------
    // Helpers
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
#ifdef SSA_USE_MKL
        return mkl_malloc(size, SSA_ALIGN);
#else
        return aligned_alloc(SSA_ALIGN, (size + SSA_ALIGN - 1) & ~(SSA_ALIGN - 1));
#endif
    }

    static inline void ssa_opt_free_ptr(void *ptr)
    {
#ifdef SSA_USE_MKL
        mkl_free(ptr);
#else
        free(ptr);
#endif
    }

    // ----------------------------------------------------------------------------
    // Vectorized Operations (using MKL VML when available)
    // ----------------------------------------------------------------------------

#ifdef SSA_USE_MKL

    // Vectorized reverse: out[i] = in[n-1-i]
    static inline void ssa_opt_reverse(const double *in, double *out, int n)
    {
        // MKL doesn't have a direct reverse, but we can use gather
        // For now, use optimized scalar loop with prefetch
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

    // Vectorized complex multiply: c = a * b (interleaved format)
    // c[2i] = a[2i]*b[2i] - a[2i+1]*b[2i+1]
    // c[2i+1] = a[2i]*b[2i+1] + a[2i+1]*b[2i]
    static void ssa_opt_complex_mul(
        const double *a, const double *b, double *c, int n)
    {
        // Use MKL's vzMul for complex multiplication
        // vzMul expects MKL_Complex16 format which is same as interleaved double
        vzMul(n, (const MKL_Complex16 *)a, (const MKL_Complex16 *)b, (MKL_Complex16 *)c);
    }

    // Dot product
    static inline double ssa_opt_dot(const double *a, const double *b, int n)
    {
        return cblas_ddot(n, a, 1, b, 1);
    }

    // Norm
    static inline double ssa_opt_nrm2(const double *v, int n)
    {
        return cblas_dnrm2(n, v, 1);
    }

    // Scale: v *= s
    static inline void ssa_opt_scal(double *v, int n, double s)
    {
        cblas_dscal(n, s, v, 1);
    }

    // AXPY: y += a*x
    static inline void ssa_opt_axpy(double *y, const double *x, double a, int n)
    {
        cblas_daxpy(n, a, x, 1, y, 1);
    }

    // Copy
    static inline void ssa_opt_copy(const double *src, double *dst, int n)
    {
        cblas_dcopy(n, src, 1, dst, 1);
    }

    // Normalize and return norm
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

    // Zero array
    static inline void ssa_opt_zero(double *v, int n)
    {
        // MKL doesn't have memset, but we can use dscal with 0
        // Actually, just use memset - it's optimized
        memset(v, 0, n * sizeof(double));
    }

#else
    // Non-MKL fallbacks

    static inline void ssa_opt_reverse(const double *in, double *out, int n)
    {
        for (int i = 0; i < n; i++)
            out[i] = in[n - 1 - i];
    }

    static void ssa_opt_complex_mul(const double *a, const double *b, double *c, int n)
    {
        for (int i = 0; i < n; i++)
        {
            double ar = a[2 * i], ai = a[2 * i + 1];
            double br = b[2 * i], bi = b[2 * i + 1];
            c[2 * i] = ar * br - ai * bi;
            c[2 * i + 1] = ar * bi + ai * br;
        }
    }

    static inline double ssa_opt_dot(const double *a, const double *b, int n)
    {
        double sum = 0;
        for (int i = 0; i < n; i++)
            sum += a[i] * b[i];
        return sum;
    }

    static inline double ssa_opt_nrm2(const double *v, int n)
    {
        return sqrt(ssa_opt_dot(v, v, n));
    }

    static inline void ssa_opt_scal(double *v, int n, double s)
    {
        for (int i = 0; i < n; i++)
            v[i] *= s;
    }

    static inline void ssa_opt_axpy(double *y, const double *x, double a, int n)
    {
        for (int i = 0; i < n; i++)
            y[i] += a * x[i];
    }

    static inline void ssa_opt_copy(const double *src, double *dst, int n)
    {
        memcpy(dst, src, n * sizeof(double));
    }

    static inline double ssa_opt_normalize(double *v, int n)
    {
        double norm = ssa_opt_nrm2(v, n);
        if (norm > 1e-15)
            ssa_opt_scal(v, n, 1.0 / norm);
        return norm;
    }

    static inline void ssa_opt_zero(double *v, int n)
    {
        memset(v, 0, n * sizeof(double));
    }

#endif // SSA_USE_MKL

    // ----------------------------------------------------------------------------
    // FFT Operations
    // ----------------------------------------------------------------------------

#ifdef SSA_USE_MKL

    // In-place FFT (interleaved complex format)
    // Input/output: [re0, im0, re1, im1, ...]
    static void ssa_opt_fft_forward_inplace(SSA_Opt *ssa, double *data)
    {
        // MKL DFTI for complex-to-complex
        // We'll use the real FFT and convert, or just do complex FFT
        // For simplicity, use complex FFT (input is zero-padded real in complex format)
        DftiComputeForward(ssa->fft_handle, data);
    }

    static void ssa_opt_fft_inverse_inplace(SSA_Opt *ssa, double *data)
    {
        DftiComputeBackward(ssa->fft_handle, data);
    }

    // Forward FFT: real input -> complex output (interleaved)
    static void ssa_opt_fft_r2c(
        SSA_Opt *ssa,
        const double *input, int input_len,
        double *output // length 2*fft_len
    )
    {
        int n = ssa->fft_len;

        // Zero-pad and convert to complex format
        ssa_opt_zero(output, 2 * n);
        for (int i = 0; i < input_len; i++)
        {
            output[2 * i] = input[i];
            // output[2*i+1] = 0 already
        }

        ssa_opt_fft_forward_inplace(ssa, output);
    }

    // Inverse FFT: complex input -> real output (takes first n real values)
    static void ssa_opt_fft_c2r(
        SSA_Opt *ssa,
        double *data, // in-place, length 2*fft_len
        double *output, int output_len)
    {
        ssa_opt_fft_inverse_inplace(ssa, data);

        // Extract real parts
        for (int i = 0; i < output_len; i++)
        {
            output[i] = data[2 * i];
        }
    }

#else
    // Built-in FFT (Cooley-Tukey)

    static void ssa_opt_fft_builtin(double *data, int n, int sign)
    {
        // Bit-reversal
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

        // Cooley-Tukey
        for (int len = 2; len <= n; len <<= 1)
        {
            double ang = sign * 2.0 * M_PI / len;
            double wpr = cos(ang), wpi = sin(ang);
            for (int i = 0; i < n; i += len)
            {
                double wr = 1.0, wi = 0.0;
                for (int jj = 0; jj < len / 2; jj++)
                {
                    int a = i + jj, b = a + len / 2;
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

        if (sign == 1)
        {
            double scale = 1.0 / n;
            for (int i = 0; i < 2 * n; i++)
                data[i] *= scale;
        }
    }

    static void ssa_opt_fft_r2c(SSA_Opt *ssa, const double *input, int input_len, double *output)
    {
        int n = ssa->fft_len;
        ssa_opt_zero(output, 2 * n);
        for (int i = 0; i < input_len; i++)
            output[2 * i] = input[i];
        ssa_opt_fft_builtin(output, n, -1);
    }

    static void ssa_opt_fft_c2r(SSA_Opt *ssa, double *data, double *output, int output_len)
    {
        ssa_opt_fft_builtin(data, ssa->fft_len, 1);
        for (int i = 0; i < output_len; i++)
            output[i] = data[2 * i];
    }

#endif // SSA_USE_MKL

    // ============================================================================
    // HANKEL MATRIX-VECTOR PRODUCTS
    // ============================================================================

    static void ssa_opt_hankel_matvec(
        SSA_Opt *ssa,
        const double *v, // Input: length K
        double *y        // Output: length L
    )
    {
        int K = ssa->K;
        int L = ssa->L;
        int n = ssa->fft_len;
        double *ws = ssa->ws_fft1;

        ssa_opt_zero(ws, 2 * n);
        for (int i = 0; i < K; i++)
        {
            ws[2 * i] = v[K - 1 - i];
        }

#ifdef SSA_USE_MKL
        DftiComputeForward(ssa->fft_handle, ws);
#else
        ssa_opt_fft_builtin(ws, n, -1);
#endif

        ssa_opt_complex_mul(ssa->fft_x, ws, ws, n);

#ifdef SSA_USE_MKL
        DftiComputeBackward(ssa->fft_handle, ws);
#else
        ssa_opt_fft_builtin(ws, n, 1);
#endif

#ifdef SSA_USE_MKL
        cblas_dcopy(L, ws + 2 * (K - 1), 2, y, 1);
#else
        for (int i = 0; i < L; i++)
        {
            y[i] = ws[2 * (K - 1 + i)];
        }
#endif
    }

    static void ssa_opt_hankel_matvec_T(
        SSA_Opt *ssa,
        const double *u, // Input: length L
        double *y        // Output: length K
    )
    {
        int K = ssa->K;
        int L = ssa->L;
        int n = ssa->fft_len;
        double *ws = ssa->ws_fft1;

        ssa_opt_zero(ws, 2 * n);
        for (int i = 0; i < L; i++)
        {
            ws[2 * i] = u[L - 1 - i];
        }

#ifdef SSA_USE_MKL
        DftiComputeForward(ssa->fft_handle, ws);
#else
        ssa_opt_fft_builtin(ws, n, -1);
#endif

        ssa_opt_complex_mul(ssa->fft_x, ws, ws, n);

#ifdef SSA_USE_MKL
        DftiComputeBackward(ssa->fft_handle, ws);
#else
        ssa_opt_fft_builtin(ws, n, 1);
#endif

#ifdef SSA_USE_MKL
        cblas_dcopy(K, ws + 2 * (L - 1), 2, y, 1);
#else
        for (int j = 0; j < K; j++)
        {
            y[j] = ws[2 * (L - 1 + j)];
        }
#endif
    }

    // ============================================================================
    // BATCHED HANKEL MATRIX-VECTOR PRODUCTS
    // ============================================================================

#ifdef SSA_USE_MKL

    static void ssa_opt_hankel_matvec_block(
        SSA_Opt *ssa,
        const double *V_block,
        double *Y_block,
        int b)
    {
        int K = ssa->K;
        int L = ssa->L;
        int n = ssa->fft_len;
        size_t stride = 2 * n;

        double *ws_in = ssa->ws_batch_u;
        double *ws_out = ssa->ws_batch_out;

        // Threshold for using batched FFT vs sequential
        // Below this, sequential FFT avoids computing unused transforms
        const int BATCH_THRESHOLD = SSA_BATCH_SIZE / 4;

        int col = 0;
        while (col < b)
        {
            int batch_count = (b - col < SSA_BATCH_SIZE) ? (b - col) : SSA_BATCH_SIZE;

            // OPTIMIZATION: Fall back to sequential FFT for small batches
            // MKL batch descriptor always computes SSA_BATCH_SIZE transforms,
            // so for small batches, sequential is more efficient
            if (batch_count < BATCH_THRESHOLD)
            {
                for (int i = 0; i < batch_count; i++)
                {
                    ssa_opt_hankel_matvec(ssa, &V_block[(col + i) * K], &Y_block[(col + i) * L]);
                }
                col += batch_count;
                continue;
            }

            // OPTIMIZATION: Single memset for entire workspace
            // Replaces per-vector memset + trailing padding memset
            memset(ws_in, 0, SSA_BATCH_SIZE * stride * sizeof(double));

            // Pack reversed vectors into workspace in interleaved complex format
            // Each transform occupies 'stride' doubles: [re₀,im₀,re₁,im₁,...]
            for (int i = 0; i < batch_count; i++)
            {
                const double *v = &V_block[(col + i) * K]; // Column i of V_block
                double *dst = ws_in + i * stride;

                // Only write non-zero values (buffer already zeroed)
                for (int j = 0; j < K; j++)
                {
                    dst[2 * j] = v[K - 1 - j]; // Pack reversed v into real parts
                }
            }

            // Batched forward FFT: ws_in → ws_out (NOT_INPLACE is faster)
            DftiComputeForward(ssa->fft_batch_c2c, ws_in, ws_out);

            // Element-wise complex multiply: FFT(vᵢ) ⊙ FFT(x) for each vector
            // vzMul is MKL's vectorized complex multiply
            for (int i = 0; i < batch_count; i++)
            {
                double *fft_v = ws_out + i * stride;
                vzMul(n, (const MKL_Complex16 *)ssa->fft_x, (const MKL_Complex16 *)fft_v,
                      (MKL_Complex16 *)fft_v);
            }

            // Batched inverse FFT: ws_out → ws_in
            DftiComputeBackward(ssa->fft_batch_c2c, ws_out, ws_in);

            // Extract results from convolution output
            // cblas_dcopy with incx=2 extracts real parts from interleaved complex
            for (int i = 0; i < batch_count; i++)
            {
                double *conv = ws_in + i * stride;
                double *y = &Y_block[(col + i) * L];
                cblas_dcopy(L, conv + 2 * (K - 1), 2, y, 1); // Stride-2 extracts reals
            }

            col += batch_count;
        }
    }

    static void ssa_opt_hankel_matvec_T_block(
        SSA_Opt *ssa,
        const double *U_block,
        double *Y_block,
        int b)
    {
        int K = ssa->K;
        int L = ssa->L;
        int n = ssa->fft_len;
        size_t stride = 2 * n;

        double *ws_in = ssa->ws_batch_u;
        double *ws_out = ssa->ws_batch_out;

        // Threshold for using batched FFT vs sequential
        const int BATCH_THRESHOLD = SSA_BATCH_SIZE / 4;

        int col = 0;
        while (col < b)
        {
            int batch_count = (b - col < SSA_BATCH_SIZE) ? (b - col) : SSA_BATCH_SIZE;

            // OPTIMIZATION: Fall back to sequential FFT for small batches
            if (batch_count < BATCH_THRESHOLD)
            {
                for (int i = 0; i < batch_count; i++)
                {
                    ssa_opt_hankel_matvec_T(ssa, &U_block[(col + i) * L], &Y_block[(col + i) * K]);
                }
                col += batch_count;
                continue;
            }

            // OPTIMIZATION: Single memset for entire workspace
            memset(ws_in, 0, SSA_BATCH_SIZE * stride * sizeof(double));

            // Pack reversed u vectors (buffer already zeroed)
            for (int i = 0; i < batch_count; i++)
            {
                const double *u = &U_block[(col + i) * L];
                double *dst = ws_in + i * stride;

                for (int j = 0; j < L; j++)
                {
                    dst[2 * j] = u[L - 1 - j];
                }
            }

            // Batch forward FFT (NOT_INPLACE)
            DftiComputeForward(ssa->fft_batch_c2c, ws_in, ws_out);

            // Complex multiply with FFT(x)
            for (int i = 0; i < batch_count; i++)
            {
                double *fft_u = ws_out + i * stride;
                vzMul(n, (const MKL_Complex16 *)ssa->fft_x, (const MKL_Complex16 *)fft_u,
                      (MKL_Complex16 *)fft_u);
            }

            // Batch inverse FFT (NOT_INPLACE)
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

#endif // SSA_USE_MKL

    // ----------------------------------------------------------------------------
    // Core API
    // ----------------------------------------------------------------------------

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

        size_t fft_size = 2 * fft_n * sizeof(double);
        size_t vec_size = ssa_opt_max(L, ssa->K) * sizeof(double);

        ssa->fft_x = (double *)ssa_opt_alloc(fft_size);
        ssa->ws_fft1 = (double *)ssa_opt_alloc(fft_size);
        ssa->ws_fft2 = (double *)ssa_opt_alloc(fft_size);

        if (!ssa->fft_x || !ssa->ws_fft1 || !ssa->ws_fft2)
        {
            ssa_opt_free(ssa);
            return -1;
        }

#ifdef SSA_USE_MKL
        ssa->ws_real = (double *)ssa_opt_alloc(fft_n * sizeof(double));
        ssa->ws_v = (double *)ssa_opt_alloc(vec_size);
        ssa->ws_u = (double *)ssa_opt_alloc(vec_size);

        size_t batch_size = 2 * fft_n * SSA_BATCH_SIZE * sizeof(double);

        ssa->ws_batch_u = (double *)ssa_opt_alloc(batch_size);
        ssa->ws_batch_v = (double *)ssa_opt_alloc(batch_size);
        ssa->ws_batch_out = (double *)ssa_opt_alloc(batch_size);

        if (!ssa->ws_batch_u || !ssa->ws_batch_v || !ssa->ws_batch_out)
        {
            ssa_opt_free(ssa);
            return -1;
        }

        MKL_LONG status;

        status = DftiCreateDescriptor(&ssa->fft_handle, DFTI_DOUBLE, DFTI_COMPLEX, 1, fft_n);
        if (status != 0)
        {
            ssa_opt_free(ssa);
            return -1;
        }

        DftiSetValue(ssa->fft_handle, DFTI_PLACEMENT, DFTI_INPLACE);
        DftiSetValue(ssa->fft_handle, DFTI_BACKWARD_SCALE, 1.0 / fft_n);

        status = DftiCommitDescriptor(ssa->fft_handle);
        if (status != 0)
        {
            ssa_opt_free(ssa);
            return -1;
        }

        status = DftiCreateDescriptor(&ssa->fft_batch_c2c, DFTI_DOUBLE, DFTI_COMPLEX, 1, fft_n);
        if (status != 0)
        {
            ssa_opt_free(ssa);
            return -1;
        }

        DftiSetValue(ssa->fft_batch_c2c, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
        DftiSetValue(ssa->fft_batch_c2c, DFTI_BACKWARD_SCALE, 1.0 / fft_n);
        DftiSetValue(ssa->fft_batch_c2c, DFTI_NUMBER_OF_TRANSFORMS, (MKL_LONG)SSA_BATCH_SIZE);
        DftiSetValue(ssa->fft_batch_c2c, DFTI_INPUT_DISTANCE, (MKL_LONG)fft_n);
        DftiSetValue(ssa->fft_batch_c2c, DFTI_OUTPUT_DISTANCE, (MKL_LONG)fft_n);

        status = DftiCommitDescriptor(ssa->fft_batch_c2c);
        if (status != 0)
        {
            ssa_opt_free(ssa);
            return -1;
        }

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

        status = vslNewStream(&ssa->rng, VSL_BRNG_MT2203, 42);
        if (status != VSL_STATUS_OK)
        {
            ssa_opt_free(ssa);
            return -1;
        }
#endif

        // Precompute FFT(x) — reused by all matvec and reconstruction operations
        ssa_opt_fft_r2c(ssa, x, N, ssa->fft_x);

        // Precompute inverse diagonal counts for reconstruction
        // count[t] = number of matrix elements on anti-diagonal t
        // Precomputing 1/count avoids division in reconstruct hot path
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
        {
            return -1;
        }

        // Invalidate any cached FFTs from previous decomposition
        ssa_opt_free_cached_ffts(ssa);

        int L = ssa->L;
        int K = ssa->K;

        k = ssa_opt_min(k, ssa_opt_min(L, K));

        ssa->U = (double *)ssa_opt_alloc(L * k * sizeof(double));
        ssa->V = (double *)ssa_opt_alloc(K * k * sizeof(double));
        ssa->sigma = (double *)ssa_opt_alloc(k * sizeof(double));
        ssa->eigenvalues = (double *)ssa_opt_alloc(k * sizeof(double));

        if (!ssa->U || !ssa->V || !ssa->sigma || !ssa->eigenvalues)
        {
            return -1;
        }

        ssa->n_components = k;

#ifdef SSA_USE_MKL
        double *u = ssa->ws_u;
        double *v = ssa->ws_v;
        double *v_new = (double *)ssa_opt_alloc(K * sizeof(double));

        ssa->ws_proj = (double *)ssa_opt_alloc(k * sizeof(double));
        if (!ssa->ws_proj || !v_new)
        {
            ssa_opt_free_ptr(v_new);
            return -1;
        }
#else
        double *u = (double *)ssa_opt_alloc(L * sizeof(double));
        double *v = (double *)ssa_opt_alloc(K * sizeof(double));
        double *v_new = (double *)ssa_opt_alloc(K * sizeof(double));
#endif

        if (!v_new)
            return -1;

#ifndef SSA_USE_MKL
        unsigned int seed = 42;
#endif

        ssa->total_variance = 0.0;

        for (int comp = 0; comp < k; comp++)
        {
#ifdef SSA_USE_MKL
            vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, ssa->rng, K, v, -0.5, 0.5);
#else
            for (int i = 0; i < K; i++)
            {
                seed = seed * 1103515245 + 12345;
                v[i] = (double)((seed >> 16) & 0x7fff) / 32768.0 - 0.5;
            }
#endif

            if (comp > 0)
            {
#ifdef SSA_USE_MKL
                cblas_dgemv(CblasColMajor, CblasTrans, K, comp,
                            1.0, ssa->V, K, v, 1, 0.0, ssa->ws_proj, 1);
                cblas_dgemv(CblasColMajor, CblasNoTrans, K, comp,
                            -1.0, ssa->V, K, ssa->ws_proj, 1, 1.0, v, 1);
#else
                for (int j = 0; j < comp; j++)
                {
                    double *v_j = &ssa->V[j * K];
                    double dot = ssa_opt_dot(v, v_j, K);
                    ssa_opt_axpy(v, v_j, -dot, K);
                }
#endif
            }
            ssa_opt_normalize(v, K);

            for (int iter = 0; iter < max_iter; iter++)
            {
                ssa_opt_hankel_matvec(ssa, v, u);

                if (comp > 0)
                {
#ifdef SSA_USE_MKL
                    cblas_dgemv(CblasColMajor, CblasTrans, L, comp,
                                1.0, ssa->U, L, u, 1, 0.0, ssa->ws_proj, 1);
                    cblas_dgemv(CblasColMajor, CblasNoTrans, L, comp,
                                -1.0, ssa->U, L, ssa->ws_proj, 1, 1.0, u, 1);
#else
                    for (int j = 0; j < comp; j++)
                    {
                        double *u_j = &ssa->U[j * L];
                        double dot = ssa_opt_dot(u, u_j, L);
                        ssa_opt_axpy(u, u_j, -dot, L);
                    }
#endif
                }

                ssa_opt_hankel_matvec_T(ssa, u, v_new);

                if (comp > 0)
                {
#ifdef SSA_USE_MKL
                    cblas_dgemv(CblasColMajor, CblasTrans, K, comp,
                                1.0, ssa->V, K, v_new, 1, 0.0, ssa->ws_proj, 1);
                    cblas_dgemv(CblasColMajor, CblasNoTrans, K, comp,
                                -1.0, ssa->V, K, ssa->ws_proj, 1, 1.0, v_new, 1);
#else
                    for (int j = 0; j < comp; j++)
                    {
                        double *v_j = &ssa->V[j * K];
                        double dot = ssa_opt_dot(v_new, v_j, K);
                        ssa_opt_axpy(v_new, v_j, -dot, K);
                    }
#endif
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

            // Final orthogonalization and SVD consistency
            if (comp > 0)
            {
#ifdef SSA_USE_MKL
                cblas_dgemv(CblasColMajor, CblasTrans, K, comp,
                            1.0, ssa->V, K, v, 1, 0.0, ssa->ws_proj, 1);
                cblas_dgemv(CblasColMajor, CblasNoTrans, K, comp,
                            -1.0, ssa->V, K, ssa->ws_proj, 1, 1.0, v, 1);
#else
                for (int j = 0; j < comp; j++)
                {
                    double *v_j = &ssa->V[j * K];
                    double dot = ssa_opt_dot(v, v_j, K);
                    ssa_opt_axpy(v, v_j, -dot, K);
                }
#endif
            }
            ssa_opt_normalize(v, K);

            ssa_opt_hankel_matvec(ssa, v, u);

            if (comp > 0)
            {
#ifdef SSA_USE_MKL
                cblas_dgemv(CblasColMajor, CblasTrans, L, comp,
                            1.0, ssa->U, L, u, 1, 0.0, ssa->ws_proj, 1);
                cblas_dgemv(CblasColMajor, CblasNoTrans, L, comp,
                            -1.0, ssa->U, L, ssa->ws_proj, 1, 1.0, u, 1);
#else
                for (int j = 0; j < comp; j++)
                {
                    double *u_j = &ssa->U[j * L];
                    double dot = ssa_opt_dot(u, u_j, L);
                    ssa_opt_axpy(u, u_j, -dot, L);
                }
#endif
            }

            double sigma = ssa_opt_normalize(u, L);

            ssa_opt_hankel_matvec_T(ssa, u, v);

            if (comp > 0)
            {
#ifdef SSA_USE_MKL
                cblas_dgemv(CblasColMajor, CblasTrans, K, comp,
                            1.0, ssa->V, K, v, 1, 0.0, ssa->ws_proj, 1);
                cblas_dgemv(CblasColMajor, CblasNoTrans, K, comp,
                            -1.0, ssa->V, K, ssa->ws_proj, 1, 1.0, v, 1);
#else
                for (int j = 0; j < comp; j++)
                {
                    double *v_j = &ssa->V[j * K];
                    double dot = ssa_opt_dot(v, v_j, K);
                    ssa_opt_axpy(v, v_j, -dot, K);
                }
#endif
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

#ifdef SSA_USE_MKL
                    cblas_dswap(L, &ssa->U[i * L], 1, &ssa->U[j * L], 1);
                    cblas_dswap(K, &ssa->V[i * K], 1, &ssa->V[j * K], 1);
#else
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
#endif
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
#ifndef SSA_USE_MKL
        ssa_opt_free_ptr(u);
        ssa_opt_free_ptr(v);
#endif

        ssa->decomposed = true;
        return 0;
    }

    // ============================================================================
    // INCREMENTAL DECOMPOSITION (EXTEND)
    // ============================================================================

    int ssa_opt_extend(SSA_Opt *ssa, int additional_k, int max_iter)
    {
        if (!ssa || !ssa->decomposed || additional_k < 1)
            return -1;

        // Invalidate cached FFTs since we're adding new components
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

        // Copy existing components
        memcpy(U_new, ssa->U, L * old_k * sizeof(double));
        memcpy(V_new, ssa->V, K * old_k * sizeof(double));
        memcpy(sigma_new, ssa->sigma, old_k * sizeof(double));
        memcpy(eigen_new, ssa->eigenvalues, old_k * sizeof(double));

        // Swap pointers
        ssa_opt_free_ptr(ssa->U);
        ssa_opt_free_ptr(ssa->V);
        ssa_opt_free_ptr(ssa->sigma);
        ssa_opt_free_ptr(ssa->eigenvalues);

        ssa->U = U_new;
        ssa->V = V_new;
        ssa->sigma = sigma_new;
        ssa->eigenvalues = eigen_new;

        // Workspace for iteration
#ifdef SSA_USE_MKL
        // Reallocate projection workspace if needed
        if (ssa->ws_proj)
        {
            ssa_opt_free_ptr(ssa->ws_proj);
        }
        ssa->ws_proj = (double *)ssa_opt_alloc(new_k * sizeof(double));
        if (!ssa->ws_proj)
            return -1;

        double *u = ssa->ws_u;
        double *v = ssa->ws_v;
#else
        double *u = (double *)ssa_opt_alloc(L * sizeof(double));
        double *v = (double *)ssa_opt_alloc(K * sizeof(double));
        if (!u || !v)
        {
            ssa_opt_free_ptr(u);
            ssa_opt_free_ptr(v);
            return -1;
        }
#endif
        double *v_new = (double *)ssa_opt_alloc(K * sizeof(double));
        if (!v_new)
        {
#ifndef SSA_USE_MKL
            ssa_opt_free_ptr(u);
            ssa_opt_free_ptr(v);
#endif
            return -1;
        }

        // Compute additional components
        for (int comp = old_k; comp < new_k; comp++)
        {
            // Initialize v randomly
#ifdef SSA_USE_MKL
            vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, ssa->rng, K, v, -0.5, 0.5);
#else
            static unsigned int seed = 12345;
            for (int i = 0; i < K; i++)
            {
                seed = seed * 1103515245 + 12345;
                v[i] = (double)((seed >> 16) & 0x7fff) / 32768.0 - 0.5;
            }
#endif

            // Orthogonalize against ALL existing components (0..comp-1)
#ifdef SSA_USE_MKL
            cblas_dgemv(CblasColMajor, CblasTrans, K, comp,
                        1.0, ssa->V, K, v, 1, 0.0, ssa->ws_proj, 1);
            cblas_dgemv(CblasColMajor, CblasNoTrans, K, comp,
                        -1.0, ssa->V, K, ssa->ws_proj, 1, 1.0, v, 1);
#else
            for (int j = 0; j < comp; j++)
            {
                double *v_j = &ssa->V[j * K];
                double dot = ssa_opt_dot(v, v_j, K);
                ssa_opt_axpy(v, v_j, -dot, K);
            }
#endif
            ssa_opt_normalize(v, K);

            // Power iteration
            for (int iter = 0; iter < max_iter; iter++)
            {
                // u = H @ v
                ssa_opt_hankel_matvec(ssa, v, u);

                // Orthogonalize u against all previous u's
#ifdef SSA_USE_MKL
                cblas_dgemv(CblasColMajor, CblasTrans, L, comp,
                            1.0, ssa->U, L, u, 1, 0.0, ssa->ws_proj, 1);
                cblas_dgemv(CblasColMajor, CblasNoTrans, L, comp,
                            -1.0, ssa->U, L, ssa->ws_proj, 1, 1.0, u, 1);
#else
                for (int j = 0; j < comp; j++)
                {
                    double *u_j = &ssa->U[j * L];
                    double dot = ssa_opt_dot(u, u_j, L);
                    ssa_opt_axpy(u, u_j, -dot, L);
                }
#endif
                ssa_opt_normalize(u, L);

                // v_new = H^T @ u
                ssa_opt_hankel_matvec_T(ssa, u, v_new);

                // Orthogonalize v_new against all previous v's
#ifdef SSA_USE_MKL
                cblas_dgemv(CblasColMajor, CblasTrans, K, comp,
                            1.0, ssa->V, K, v_new, 1, 0.0, ssa->ws_proj, 1);
                cblas_dgemv(CblasColMajor, CblasNoTrans, K, comp,
                            -1.0, ssa->V, K, ssa->ws_proj, 1, 1.0, v_new, 1);
#else
                for (int j = 0; j < comp; j++)
                {
                    double *v_j = &ssa->V[j * K];
                    double dot = ssa_opt_dot(v_new, v_j, K);
                    ssa_opt_axpy(v_new, v_j, -dot, K);
                }
#endif
                ssa_opt_normalize(v_new, K);

                // Convergence check
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

            // =========================================================
            // Final orthogonalization and SVD consistency
            // Same procedure as in ssa_opt_decompose
            // =========================================================

            // Step 1: Final orthogonalization of v
#ifdef SSA_USE_MKL
            cblas_dgemv(CblasColMajor, CblasTrans, K, comp,
                        1.0, ssa->V, K, v, 1, 0.0, ssa->ws_proj, 1);
            cblas_dgemv(CblasColMajor, CblasNoTrans, K, comp,
                        -1.0, ssa->V, K, ssa->ws_proj, 1, 1.0, v, 1);
#else
            for (int j = 0; j < comp; j++)
            {
                double *v_j = &ssa->V[j * K];
                double dot = ssa_opt_dot(v, v_j, K);
                ssa_opt_axpy(v, v_j, -dot, K);
            }
#endif
            ssa_opt_normalize(v, K);

            // Step 2: Compute u = H @ v
            ssa_opt_hankel_matvec(ssa, v, u);

            // Step 3: Orthogonalize u and extract sigma
#ifdef SSA_USE_MKL
            cblas_dgemv(CblasColMajor, CblasTrans, L, comp,
                        1.0, ssa->U, L, u, 1, 0.0, ssa->ws_proj, 1);
            cblas_dgemv(CblasColMajor, CblasNoTrans, L, comp,
                        -1.0, ssa->U, L, ssa->ws_proj, 1, 1.0, u, 1);
#else
            for (int j = 0; j < comp; j++)
            {
                double *u_j = &ssa->U[j * L];
                double dot = ssa_opt_dot(u, u_j, L);
                ssa_opt_axpy(u, u_j, -dot, L);
            }
#endif
            double sigma = ssa_opt_normalize(u, L);

            // Step 4: Recompute v = H^T @ u for SVD consistency
            ssa_opt_hankel_matvec_T(ssa, u, v);

            // Step 5: Orthogonalize v and scale by 1/sigma
#ifdef SSA_USE_MKL
            cblas_dgemv(CblasColMajor, CblasTrans, K, comp,
                        1.0, ssa->V, K, v, 1, 0.0, ssa->ws_proj, 1);
            cblas_dgemv(CblasColMajor, CblasNoTrans, K, comp,
                        -1.0, ssa->V, K, ssa->ws_proj, 1, 1.0, v, 1);
#else
            for (int j = 0; j < comp; j++)
            {
                double *v_j = &ssa->V[j * K];
                double dot = ssa_opt_dot(v, v_j, K);
                ssa_opt_axpy(v, v_j, -dot, K);
            }
#endif

            // Scale v by 1/sigma to maintain SVD relationship
            if (sigma > 1e-15)
            {
                ssa_opt_scal(v, K, 1.0 / sigma);
            }

            // Store results
            ssa_opt_copy(u, &ssa->U[comp * L], L);
            ssa_opt_copy(v, &ssa->V[comp * K], K);
            ssa->sigma[comp] = sigma;
            ssa->eigenvalues[comp] = sigma * sigma;
            ssa->total_variance += sigma * sigma;
        }

        // Sort NEW components by descending singular value
        // (Keep existing components in place, sort only the new ones among themselves,
        // then merge into correct positions)
        // Actually, for simplicity, just re-sort everything - it's O(k^2) but k is small

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

#ifdef SSA_USE_MKL
                    cblas_dswap(L, &ssa->U[i * L], 1, &ssa->U[j * L], 1);
                    cblas_dswap(K, &ssa->V[i * K], 1, &ssa->V[j * K], 1);
#else
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
#endif
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
#ifndef SSA_USE_MKL
        ssa_opt_free_ptr(u);
        ssa_opt_free_ptr(v);
#endif

        return 0;
    }

    // ============================================================================
    // BLOCK POWER METHOD DECOMPOSITION (Phase 2)
    // ============================================================================
    //
    // MATHEMATICAL OPERATION:
    //   Compute k singular triplets (σᵢ, uᵢ, vᵢ) of the Hankel matrix H.
    //
    // TEXTBOOK APPROACH (Sequential Power Iteration):
    //   for i = 1 to k:
    //       v = random()
    //       for iter = 1 to max_iter:
    //           u = H @ v;  u = u / ‖u‖
    //           v = Hᵀ @ u; v = v / ‖v‖
    //       σᵢ = ‖H @ v‖
    //       deflate H
    //
    //   Problem: Each component needs separate iteration to converge.
    //   For k=32, max_iter=100: 6400 matvec calls.
    //
    // THIS IMPLEMENTATION (Block Subspace Iteration):
    //   Process b vectors simultaneously:
    //       V = random(K, b)
    //       for iter = 1 to max_iter:
    //           U = H @ V;    U = QR(U)
    //           V = Hᵀ @ U;   V = QR(V)
    //       [σ, U, V] = Rayleigh-Ritz(U, V)
    //
    //   For k=32, b=32: All components converge together in ~100 iterations.
    //   Matvec calls: 200 batched operations (each processes 32 vectors).
    //
    // KEY OPTIMIZATIONS:
    //
    //   [1] BATCHED FFT (see ssa_opt_hankel_matvec_block)
    //       Instead of 32 individual FFT calls, one batched MKL call.
    //       Amortizes descriptor overhead, enables cross-transform SIMD.
    //
    //   [2] PERIODIC QR (every 5 iterations instead of every iteration)
    //       QR factorization is O(n × b²), expensive for large n.
    //       Subspace angles drift slowly for well-separated singular values.
    //       Doing QR every 5 iterations reduces QR cost by 5x with <0.1% accuracy loss.
    //       Math: drift bounded by (σ_{b+1}/σ_b)^interval, negligible for decaying spectrum.
    //
    //   [3] GEMM-BASED ORTHOGONALIZATION (instead of Gram-Schmidt loops)
    //       Deflation requires orthogonalizing against previous components.
    //       Textbook: loop of dot products and axpy operations.
    //       Optimized: proj = V_prevᵀ @ V_block (one GEMM), V -= V_prev @ proj (one GEMM).
    //       GEMM is cache-optimized and SIMD-vectorized by MKL.
    //
    //   [4] RAYLEIGH-RITZ REFINEMENT (instead of reporting ‖Av‖ as σ)
    //       After iteration, form M = Uᵀ @ H @ V (small b×b matrix).
    //       SVD of M gives accurate σ even for clustered/repeated singular values.
    //       Textbook power iteration fails for repeated eigenvalues; Rayleigh-Ritz handles them.
    //
    // ============================================================================

#ifdef SSA_USE_MKL

    int ssa_opt_decompose_block(SSA_Opt *ssa, int k, int block_size, int max_iter)
    {
        if (!ssa || !ssa->initialized || k < 1 || block_size < 1)
        {
            return -1;
        }

        // Invalidate any cached FFTs from previous decomposition
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

        // Allocate results
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

        // Block workspace
        double *V_block = (double *)ssa_opt_alloc(K * b * sizeof(double));
        double *U_block = (double *)ssa_opt_alloc(L * b * sizeof(double));
        double *U_block2 = (double *)ssa_opt_alloc(L * b * sizeof(double)); // For Rayleigh-Ritz
        double *tau_u = (double *)ssa_opt_alloc(b * sizeof(double));        // Householder coefficients for QR
        double *tau_v = (double *)ssa_opt_alloc(b * sizeof(double));

        // Rayleigh-Ritz workspace: M = Uᵀ @ H @ V is b×b
        double *M = (double *)ssa_opt_alloc(b * b * sizeof(double));
        double *U_small = (double *)ssa_opt_alloc(b * b * sizeof(double));  // SVD left vectors
        double *Vt_small = (double *)ssa_opt_alloc(b * b * sizeof(double)); // SVD right vectors (transposed)
        double *S_small = (double *)ssa_opt_alloc(b * sizeof(double));      // SVD singular values
        double *superb = (double *)ssa_opt_alloc(b * sizeof(double));       // LAPACK workspace

        // Workspace for GEMM-based orthogonalization
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

        int comp = 0; // Components computed so far

        while (comp < k)
        {
            int cur_b = ssa_opt_min(b, k - comp); // Last block may be smaller

            // Initialize V_block with random values (MKL VSL)
            vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, ssa->rng, K * cur_b, V_block, -0.5, 0.5);

            // OPTIMIZATION [3]: GEMM-based orthogonalization against previous V's
            // Textbook would use a loop: for each prev vector, subtract projection
            // GEMM does it all at once: V_block -= V_prev @ (V_prevᵀ @ V_block)
            if (comp > 0)
            {
                // proj = V_prevᵀ @ V_block  (comp × cur_b matrix)
                cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                            comp, cur_b, K,
                            1.0, ssa->V, K, V_block, K,
                            0.0, work, comp);
                // V_block = V_block - V_prev @ proj
                cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                            K, cur_b, comp,
                            -1.0, ssa->V, K, work, comp,
                            1.0, V_block, K);
            }

            // Initial QR to get orthonormal starting basis
            LAPACKE_dgeqrf(LAPACK_COL_MAJOR, K, cur_b, V_block, K, tau_v);
            LAPACKE_dorgqr(LAPACK_COL_MAJOR, K, cur_b, cur_b, V_block, K, tau_v);

            // OPTIMIZATION [2]: Periodic QR instead of every-iteration QR
            // Subspace drift is bounded by (σ_{next}/σ_current)^interval
            // For SSA with decaying spectrum, this is negligible over 5 iterations
            const int QR_INTERVAL = 5;

            for (int iter = 0; iter < max_iter; iter++)
            {
                // U_block = X @ V_block
                ssa_opt_hankel_matvec_block(ssa, V_block, U_block, cur_b);

                // Orthogonalize U against previously computed U's
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

                // Periodic QR for U_block
                if ((iter % QR_INTERVAL == 0) || (iter == max_iter - 1))
                {
                    LAPACKE_dgeqrf(LAPACK_COL_MAJOR, L, cur_b, U_block, L, tau_u);
                    LAPACKE_dorgqr(LAPACK_COL_MAJOR, L, cur_b, cur_b, U_block, L, tau_u);
                }

                // V_block = X^T @ U_block
                ssa_opt_hankel_matvec_T_block(ssa, U_block, V_block, cur_b);

                // Orthogonalize V against previously computed V's
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

                // Periodic QR for V_block
                // Skip iter=0: initial QR already orthonormalized V_block, and we just
                // recomputed it. The iter=0 U_block QR ensures U is orthonormal, so
                // V_block = H^T @ U_block is reasonably well-conditioned for iter 1-4.
                if ((iter > 0 && iter % QR_INTERVAL == 0) || (iter == max_iter - 1))
                {
                    LAPACKE_dgeqrf(LAPACK_COL_MAJOR, K, cur_b, V_block, K, tau_v);
                    LAPACKE_dorgqr(LAPACK_COL_MAJOR, K, cur_b, cur_b, V_block, K, tau_v);
                }
            }

            // Final QR to ensure clean orthogonality before Rayleigh-Ritz
            // (Already done as iter == max_iter - 1 triggers QR above)

            // =====================================================================
            // Rayleigh-Ritz: Extract singular values and refine singular vectors
            //
            // After convergence, U_block and V_block span the dominant subspaces.
            // The optimal singular triplets within these subspaces are found by:
            //   1. Compute M = U_block^T @ (X @ V_block)  (cur_b × cur_b)
            //   2. SVD(M) = U_small @ Σ @ Vt_small
            //   3. Rotate: U_final = U_block @ U_small
            //              V_final = V_block @ V_small  (V_small = Vt_small^T)
            // =====================================================================

            // Step 1: U_block2 = X @ V_block
            ssa_opt_hankel_matvec_block(ssa, V_block, U_block2, cur_b);

            // Orthogonalize U_block2 against previous U's (for numerical stability)
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

            // Step 2: M = U_block^T @ U_block2  (cur_b × cur_b)
            cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                        cur_b, cur_b, L,
                        1.0, U_block, L, U_block2, L,
                        0.0, M, cur_b);

            // Step 3: SVD of M → U_small, S_small, Vt_small
            // M = U_small @ diag(S_small) @ Vt_small
            int svd_info = LAPACKE_dgesvd(LAPACK_COL_MAJOR, 'A', 'A',
                                          cur_b, cur_b, M, cur_b,
                                          S_small, U_small, cur_b, Vt_small, cur_b,
                                          superb);
            if (svd_info != 0)
            {
                // SVD failed - fall back to column norm extraction
                for (int i = 0; i < cur_b; i++)
                {
                    S_small[i] = cblas_dnrm2(L, &U_block2[i * L], 1);
                }
                // Set U_small and Vt_small to identity
                memset(U_small, 0, cur_b * cur_b * sizeof(double));
                memset(Vt_small, 0, cur_b * cur_b * sizeof(double));
                for (int i = 0; i < cur_b; i++)
                {
                    U_small[i + i * cur_b] = 1.0;
                    Vt_small[i + i * cur_b] = 1.0;
                }
            }

            // Step 4: Rotate U_block by U_small
            // U_rotated = U_block @ U_small  (L × cur_b)
            // Store in U_block2 temporarily
            cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                        L, cur_b, cur_b,
                        1.0, U_block, L, U_small, cur_b,
                        0.0, U_block2, L);

            // Step 5: Rotate V_block by V_small = Vt_small^T
            // V_rotated = V_block @ Vt_small^T  (K × cur_b)
            // Store in work temporarily (need K × cur_b space)
            cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                        K, cur_b, cur_b,
                        1.0, V_block, K, Vt_small, cur_b,
                        0.0, work, K);

            // Copy rotated vectors to final results
            for (int i = 0; i < cur_b; i++)
            {
                double sigma = S_small[i];

                ssa->sigma[comp + i] = sigma;
                ssa->eigenvalues[comp + i] = sigma * sigma;
                ssa->total_variance += sigma * sigma;

                // Copy U (already normalized by Rayleigh-Ritz)
                cblas_dcopy(L, &U_block2[i * L], 1, &ssa->U[(comp + i) * L], 1);

                // Copy V (already normalized by Rayleigh-Ritz)
                cblas_dcopy(K, &work[i * K], 1, &ssa->V[(comp + i) * K], 1);
            }

            // =====================================================================
            // Final V refinement: Recompute V = (1/σ) * X^T @ U for SVD consistency
            // Use batched GEMM for orthogonalization instead of per-vector GEMV
            // =====================================================================

            // Compute V_block = X^T @ U_block (for all cur_b vectors at once)
            // U vectors are now in ssa->U[(comp)..(comp+cur_b)]
            // First, pack them into U_block for batched operation
            for (int i = 0; i < cur_b; i++)
            {
                cblas_dcopy(L, &ssa->U[(comp + i) * L], 1, &U_block[i * L], 1);
            }

            // V_block = X^T @ U_block
            ssa_opt_hankel_matvec_T_block(ssa, U_block, V_block, cur_b);

            // Orthogonalize all V_block columns against previous V's using GEMM
            if (comp > 0)
            {
                // proj = V_prev^T @ V_block  (comp × cur_b)
                cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                            comp, cur_b, K,
                            1.0, ssa->V, K, V_block, K,
                            0.0, work, comp);
                // V_block = V_block - V_prev @ proj
                cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                            K, cur_b, comp,
                            -1.0, ssa->V, K, work, comp,
                            1.0, V_block, K);
            }

            // Scale each V column by 1/σ and copy to results
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
                    // Swap scalars
                    double tmp = ssa->sigma[i];
                    ssa->sigma[i] = ssa->sigma[j];
                    ssa->sigma[j] = tmp;

                    tmp = ssa->eigenvalues[i];
                    ssa->eigenvalues[i] = ssa->eigenvalues[j];
                    ssa->eigenvalues[j] = tmp;

                    // Swap vectors
                    cblas_dswap(L, &ssa->U[i * L], 1, &ssa->U[j * L], 1);
                    cblas_dswap(K, &ssa->V[i * K], 1, &ssa->V[j * K], 1);
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
                ssa_opt_scal(&ssa->U[i * L], L, -1.0);
                ssa_opt_scal(&ssa->V[i * K], K, -1.0);
            }
        }

        // Cleanup
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
    // RANDOMIZED SVD DECOMPOSITION (Phase 3)
    // ============================================================================
    //
    // MATHEMATICAL OPERATION:
    //   Compute k singular triplets (σᵢ, uᵢ, vᵢ) of the Hankel matrix H.
    //
    // TEXTBOOK APPROACH (Power Iteration):
    //   - Iterate many times until convergence
    //   - For k=32, max_iter=100: ~6400 matvec calls
    //
    // THIS IMPLEMENTATION (Halko-Martinsson-Tropp Randomized SVD):
    //   Only TWO passes over the data:
    //
    //   Pass 1: Range finding
    //       Ω = randn(K, k+p)         — Gaussian random test matrix
    //       Y = H @ Ω                 — Sample the range of H
    //       Q = orth(Y)               — Orthonormal basis via QR
    //
    //   Pass 2: Projection
    //       B = Hᵀ @ Q                — Project H onto Q
    //       [Ũ, Σ, Ṽ] = SVD(B)        — Small dense SVD
    //       U = Q @ Ṽ, V = Ũ          — Recover full vectors
    //
    //   For k=32, p=8: only 80 matvec calls (40 forward + 40 transpose)
    //
    // WHY THIS WORKS:
    //   The random matrix Ω "probes" the column space of H.
    //   For matrices with decaying singular values (like Hankel from time series),
    //   Y = H @ Ω captures the dominant singular vectors with high probability.
    //   Oversampling (p extra columns) provides a safety margin.
    //
    //   Theoretical guarantee: With p ≥ 4, the error is bounded by O(σ_{k+1})
    //   which is the best possible (the next singular value we didn't compute).
    //
    // WHY NOT ALWAYS USE THIS?
    //   - Requires 2 passes (streaming scenarios may prefer 1-pass methods)
    //   - For very small k, overhead of QR + dense SVD may dominate
    //   - Accuracy depends on singular value decay; slow decay → use power iteration
    //
    // REFERENCE:
    //   Halko, Martinsson, Tropp. "Finding structure with randomness:
    //   Probabilistic algorithms for constructing approximate matrix decompositions."
    //   SIAM Review, 2011. (Algorithm 5.1)
    //
    // ============================================================================

    int ssa_opt_decompose_randomized(SSA_Opt *ssa, int k, int oversampling)
    {
        if (!ssa || !ssa->initialized || k < 1)
        {
            return -1;
        }

        // Invalidate any cached FFTs from previous decomposition
        ssa_opt_free_cached_ffts(ssa);

        int L = ssa->L;
        int K = ssa->K;

        // Oversampling p: extra columns for numerical stability
        // p=8 is typical; larger p → more accurate but slower
        int p = (oversampling <= 0) ? 8 : oversampling;
        int kp = k + p;

        kp = ssa_opt_min(kp, ssa_opt_min(L, K));
        k = ssa_opt_min(k, kp);

        // Allocate results
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

        // Workspace allocation
        double *Omega = (double *)ssa_opt_alloc(K * kp * sizeof(double)); // Random test matrix
        double *Y = (double *)ssa_opt_alloc(L * kp * sizeof(double));     // Y = H @ Ω
        double *Q = (double *)ssa_opt_alloc(L * kp * sizeof(double));     // Orthonormal basis
        double *B = (double *)ssa_opt_alloc(K * kp * sizeof(double));     // B = Hᵀ @ Q
        double *tau = (double *)ssa_opt_alloc(kp * sizeof(double));       // Householder coefficients

        // SVD workspace (divide-and-conquer via dgesdd is faster than dgesvd)
        double *U_svd = (double *)ssa_opt_alloc(K * kp * sizeof(double));
        double *Vt_svd = (double *)ssa_opt_alloc(kp * kp * sizeof(double));
        double *S_svd = (double *)ssa_opt_alloc(kp * sizeof(double));

        // Query optimal LAPACK workspace
        double work_query;
        int *iwork = (int *)ssa_opt_alloc(8 * kp * sizeof(int));
        int lwork = -1;
        int info;

        // Workspace query
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
        vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER, ssa->rng, K * kp, Omega, 0.0, 1.0);

        // =========================================================================
        // Step 2: Y = X @ Ω (range sampling via batched FFT matvec)
        // =========================================================================
        ssa_opt_hankel_matvec_block(ssa, Omega, Y, kp);

        // =========================================================================
        // Step 3: Q = orth(Y) via QR factorization
        // =========================================================================
        // Copy Y to Q (dgeqrf overwrites input)
        cblas_dcopy(L * kp, Y, 1, Q, 1);

        // QR factorization: Q = Q * R (Q stored in-place, R discarded)
        LAPACKE_dgeqrf(LAPACK_COL_MAJOR, L, kp, Q, L, tau);

        // Extract orthonormal Q by applying Householder reflectors
        LAPACKE_dorgqr(LAPACK_COL_MAJOR, L, kp, kp, Q, L, tau);

        // =========================================================================
        // Step 4: B = X^T @ Q (projection via batched FFT matvec transpose)
        // =========================================================================
        ssa_opt_hankel_matvec_T_block(ssa, Q, B, kp);

        // =========================================================================
        // Step 5: SVD of small matrix B (K × kp)
        // B = U_svd @ diag(S_svd) @ Vt_svd
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
        // From B = U_svd @ Σ @ Vt_svd and B = X^T @ Q:
        //   X^T @ Q = U_svd @ Σ @ Vt_svd
        //   X ≈ Q @ V_svd @ Σ @ U_svd^T
        //
        // So: U_final = Q @ V_svd = Q @ Vt_svd^T
        //     V_final = U_svd
        //     σ_final = S_svd
        // =========================================================================

        // U = Q @ Vt_svd^T (only first k columns)
        // Q is L × kp, Vt_svd is kp × kp, we want L × k
        // Vt_svd^T[:, :k] = first k rows of Vt_svd transposed = first k columns of V_svd
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                    L, k, kp,
                    1.0, Q, L, Vt_svd, kp,
                    0.0, ssa->U, L);

        // V = U_svd[:, :k]
        // U_svd is K × kp, we want K × k
        for (int i = 0; i < k; i++)
        {
            cblas_dcopy(K, &U_svd[i * K], 1, &ssa->V[i * K], 1);
        }

        // Copy singular values and compute eigenvalues
        for (int i = 0; i < k; i++)
        {
            ssa->sigma[i] = S_svd[i];
            ssa->eigenvalues[i] = S_svd[i] * S_svd[i];
            ssa->total_variance += ssa->eigenvalues[i];
        }

        // =========================================================================
        // Fix sign convention: make sum(U) positive
        // =========================================================================
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

        // =========================================================================
        // Cleanup
        // =========================================================================
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

#else
    // Non-MKL fallback: use sequential decomposition
    int ssa_opt_decompose_block(SSA_Opt *ssa, int k, int block_size, int max_iter)
    {
        (void)block_size; // Unused
        return ssa_opt_decompose(ssa, k, max_iter);
    }

    int ssa_opt_decompose_randomized(SSA_Opt *ssa, int k, int oversampling)
    {
        (void)oversampling;                    // Unused
        return ssa_opt_decompose(ssa, k, 200); // Fall back to power iteration
    }
#endif // SSA_USE_MKL

    // ============================================================================
    // FFT CACHING FOR RECONSTRUCTION
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

        // Free any existing cached FFTs
        ssa_opt_free_cached_ffts(ssa);

        int L = ssa->L;
        int K = ssa->K;
        int fft_n = ssa->fft_len;
        int k = ssa->n_components;
        size_t fft_size = 2 * fft_n; // Interleaved complex

        // Allocate FFT cache
        ssa->U_fft = (double *)ssa_opt_alloc(fft_size * k * sizeof(double));
        ssa->V_fft = (double *)ssa_opt_alloc(fft_size * k * sizeof(double));

        if (!ssa->U_fft || !ssa->V_fft)
        {
            ssa_opt_free_cached_ffts(ssa);
            return -1;
        }

#ifdef SSA_USE_MKL
        // Compute and cache FFT of each u_i (with sigma pre-scaling)
        for (int i = 0; i < k; i++)
        {
            double sigma = ssa->sigma[i];
            const double *u_vec = &ssa->U[i * L];
            double *dst = &ssa->U_fft[i * fft_size];

            // Zero-pad and pack into complex format
            memset(dst, 0, fft_size * sizeof(double));
            for (int j = 0; j < L; j++)
            {
                dst[2 * j] = sigma * u_vec[j]; // Pre-scale by sigma
            }

            // In-place FFT
            DftiComputeForward(ssa->fft_handle, dst);
        }

        // Compute and cache FFT of each v_i
        for (int i = 0; i < k; i++)
        {
            const double *v_vec = &ssa->V[i * K];
            double *dst = &ssa->V_fft[i * fft_size];

            // Zero-pad and pack into complex format
            memset(dst, 0, fft_size * sizeof(double));
            for (int j = 0; j < K; j++)
            {
                dst[2 * j] = v_vec[j];
            }

            // In-place FFT
            DftiComputeForward(ssa->fft_handle, dst);
        }
#else
        // Non-MKL fallback
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
            ssa_opt_fft_builtin(dst, fft_n, -1);
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
            ssa_opt_fft_builtin(dst, fft_n, -1);
        }
#endif

        ssa->fft_cached = true;
        return 0;
    }

    // ============================================================================
    // SIGNAL RECONSTRUCTION
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
        size_t fft_size = 2 * fft_n;

        ssa_opt_zero(output, N);

#ifdef SSA_USE_MKL
        double *ws_u = ssa->ws_batch_u;
        double *ws_v = ssa->ws_batch_v;
        size_t stride = fft_size;

        // Check if we can use cached FFTs
        if (ssa->fft_cached && ssa->U_fft && ssa->V_fft)
        {
            // =========================================================
            // FAST PATH: Use cached FFTs (skip forward FFT entirely)
            // =========================================================
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

                // Copy cached FFTs to workspace and multiply
                for (int b = 0; b < batch_count; b++)
                {
                    int idx = batch_indices[b];
                    const double *u_fft_cached = &ssa->U_fft[idx * fft_size];
                    const double *v_fft_cached = &ssa->V_fft[idx * fft_size];
                    double *dst = ws_u + b * stride;

                    // Element-wise complex multiply directly into workspace
                    vzMul(fft_n, (const MKL_Complex16 *)u_fft_cached,
                          (const MKL_Complex16 *)v_fft_cached,
                          (MKL_Complex16 *)dst);
                }

                // Zero unused slots for batched IFFT
                if (batch_count < SSA_BATCH_SIZE)
                {
                    memset(ws_u + batch_count * stride, 0,
                           (SSA_BATCH_SIZE - batch_count) * stride * sizeof(double));
                }

                // Batched inverse FFT
                DftiComputeBackward(ssa->fft_batch_inplace, ws_u);

                // Accumulate results
                for (int b = 0; b < batch_count; b++)
                {
                    double *conv = ws_u + b * stride;
                    cblas_daxpy(N, 1.0, conv, 2, output, 1);
                }
            }
        }
        else
        {
            // =========================================================
            // STANDARD PATH: Compute FFTs on the fly
            // =========================================================
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

                // Single memset for entire workspace
                memset(ws_u, 0, SSA_BATCH_SIZE * stride * sizeof(double));
                memset(ws_v, 0, SSA_BATCH_SIZE * stride * sizeof(double));

                // Pack u vectors with σ pre-scaling
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

                // Pack v vectors
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

                // Batched forward FFT
                DftiComputeForward(ssa->fft_batch_inplace, ws_u);
                DftiComputeForward(ssa->fft_batch_inplace, ws_v);

                // Element-wise complex multiply
                for (int b = 0; b < batch_count; b++)
                {
                    double *u_fft = ws_u + b * stride;
                    double *v_fft = ws_v + b * stride;
                    vzMul(fft_n, (const MKL_Complex16 *)u_fft, (const MKL_Complex16 *)v_fft,
                          (MKL_Complex16 *)u_fft);
                }

                // Batched inverse FFT
                DftiComputeBackward(ssa->fft_batch_inplace, ws_u);

                // Accumulate results
                for (int b = 0; b < batch_count; b++)
                {
                    double *conv = ws_u + b * stride;
                    cblas_daxpy(N, 1.0, conv, 2, output, 1);
                }
            }
        }

#else
        // Non-MKL path
        double *ws1 = ssa->ws_fft1;
        double *ws2 = ssa->ws_fft2;

        if (ssa->fft_cached && ssa->U_fft && ssa->V_fft)
        {
            // FAST PATH: Use cached FFTs
            for (int g = 0; g < n_group; g++)
            {
                int idx = group[g];
                if (idx < 0 || idx >= ssa->n_components)
                    continue;

                const double *u_fft_cached = &ssa->U_fft[idx * fft_size];
                const double *v_fft_cached = &ssa->V_fft[idx * fft_size];

                // Copy and multiply
                ssa_opt_complex_mul(u_fft_cached, v_fft_cached, ws1, fft_n);
                ssa_opt_fft_builtin(ws1, fft_n, 1);

                for (int t = 0; t < N; t++)
                {
                    output[t] += ws1[2 * t];
                }
            }
        }
        else
        {
            // STANDARD PATH: Compute FFTs on the fly
            for (int g = 0; g < n_group; g++)
            {
                int idx = group[g];
                if (idx < 0 || idx >= ssa->n_components)
                    continue;

                double sigma = ssa->sigma[idx];
                const double *u_vec = &ssa->U[idx * L];
                const double *v_vec = &ssa->V[idx * K];

                ssa_opt_zero(ws1, 2 * fft_n);
                for (int i = 0; i < L; i++)
                    ws1[2 * i] = sigma * u_vec[i];
                ssa_opt_fft_builtin(ws1, fft_n, -1);

                ssa_opt_zero(ws2, 2 * fft_n);
                for (int i = 0; i < K; i++)
                    ws2[2 * i] = v_vec[i];
                ssa_opt_fft_builtin(ws2, fft_n, -1);

                ssa_opt_complex_mul(ws1, ws2, ws1, fft_n);
                ssa_opt_fft_builtin(ws1, fft_n, 1);

                for (int t = 0; t < N; t++)
                {
                    output[t] += ws1[2 * t];
                }
            }
        }
#endif

        // Diagonal averaging using precomputed inverse weights
        // Uses multiplication instead of division (10-20x faster per element)
#ifdef SSA_USE_MKL
        // Vectorized multiply: output[t] *= inv_diag_count[t]
        vdMul(N, output, ssa->inv_diag_count, output);
#else
        for (int t = 0; t < N; t++)
        {
            output[t] *= ssa->inv_diag_count[t];
        }
#endif

        return 0;
    }

    // ============================================================================
    // W-CORRELATION
    // ============================================================================

    int ssa_opt_wcorr_matrix(const SSA_Opt *ssa, double *W)
    {
        if (!ssa || !ssa->decomposed || !W)
            return -1;

        int N = ssa->N;
        int L = ssa->L;
        int K = ssa->K;
        int n = ssa->n_components;

        // Precompute weights
        double *weights = (double *)ssa_opt_alloc(N * sizeof(double));
        if (!weights)
            return -1;

        for (int t = 0; t < N; t++)
        {
            weights[t] = (double)ssa_opt_min(ssa_opt_min(t + 1, L),
                                             ssa_opt_min(K, N - t));
        }

        // Reconstruct all components
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

        // Compute weighted norms
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

    // ============================================================================
    // COMPONENT STATISTICS
    // ============================================================================

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

        // Copy singular values and compute log
        for (int i = 0; i < n; i++)
        {
            stats->singular_values[i] = ssa->sigma[i];
            stats->log_sv[i] = log(ssa->sigma[i] + 1e-300);
        }

        // Compute gap ratios
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

        // Cumulative explained variance
        double cumsum = 0.0;
        for (int i = 0; i < n; i++)
        {
            cumsum += ssa->eigenvalues[i];
            stats->cumulative_var[i] = cumsum / ssa->total_variance;
        }

        // Second differences for elbow detection
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

                // Check singular value similarity
                double sv_ratio = ssa->sigma[j] / (ssa->sigma[i] + 1e-300);
                if (fabs(1.0 - sv_ratio) > sv_tol)
                    continue;

                // Check W-correlation
                double wcorr = fabs(ssa_opt_wcorr_pair(ssa, i, j));
                if (wcorr < wcorr_thresh)
                    continue;

                // Found a pair
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

    // ============================================================================
    // CONVENIENCE FUNCTIONS
    // ============================================================================

    void ssa_opt_free(SSA_Opt *ssa)
    {
        if (!ssa)
            return;

#ifdef SSA_USE_MKL
        if (ssa->fft_handle)
            DftiFreeDescriptor(&ssa->fft_handle);
        if (ssa->fft_batch_c2c)
            DftiFreeDescriptor(&ssa->fft_batch_c2c);
        if (ssa->fft_batch_inplace)
            DftiFreeDescriptor(&ssa->fft_batch_inplace);
        if (ssa->rng)
            vslDeleteStream(&ssa->rng);
        ssa_opt_free_ptr(ssa->ws_real);
        ssa_opt_free_ptr(ssa->ws_v);
        ssa_opt_free_ptr(ssa->ws_u);
        ssa_opt_free_ptr(ssa->ws_proj);
        ssa_opt_free_ptr(ssa->ws_batch_u);
        ssa_opt_free_ptr(ssa->ws_batch_v);
        ssa_opt_free_ptr(ssa->ws_batch_out);
#endif

        ssa_opt_free_ptr(ssa->fft_x);
        ssa_opt_free_ptr(ssa->ws_fft1);
        ssa_opt_free_ptr(ssa->ws_fft2);
        ssa_opt_free_ptr(ssa->U);
        ssa_opt_free_ptr(ssa->V);
        ssa_opt_free_ptr(ssa->sigma);
        ssa_opt_free_ptr(ssa->eigenvalues);

        // Free reconstruction optimization buffers
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