/*
 * SSA with Intel MKL - Optimized Version
 *
 * Optimizations over ssa_mkl.h:
 *   1. Pre-allocated workspace (zero allocations in hot paths)
 *   2. Vectorized glue code using MKL VML
 *   3. Cache-friendly memory access patterns
 *   4. Fused operations where possible
 *   5. Aligned memory throughout
 *   6. Batch FFT for reconstruction
 *
 * Compile:
 *   source /opt/intel/oneapi/setvars.sh
 *   gcc -O3 -march=native -DSSA_USE_MKL -I${MKLROOT}/include \
 *       -o ssa_test ssa_opt.c \
 *       -L${MKLROOT}/lib/intel64 -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core \
 *       -liomp5 -lpthread -lm
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

#ifndef SSA_CONVERGENCE_TOL
#define SSA_CONVERGENCE_TOL 1e-12
#endif

// Memory alignment for AVX-512
#define SSA_ALIGN 64

// Batch size for reconstruction FFTs
#ifndef SSA_BATCH_SIZE
#define SSA_BATCH_SIZE 32
#endif

    // ============================================================================
    // Data Structures - Optimized Layout
    // ============================================================================

    typedef struct
    {
        // Dimensions
        int N;       // Series length
        int L;       // Window length
        int K;       // K = N - L + 1
        int fft_len; // FFT length (power of 2)

#ifdef SSA_USE_MKL
        // MKL FFT descriptors
        DFTI_DESCRIPTOR_HANDLE fft_handle;        // Single C2C FFT
        DFTI_DESCRIPTOR_HANDLE fft_batch_c2c;     // Batched C2C for block matvec (NOT_INPLACE)
        DFTI_DESCRIPTOR_HANDLE fft_batch_inplace; // Batched C2C for reconstruction (INPLACE)

        // MKL Random Number Generator
        VSLStreamStatePtr rng;

        // Precomputed FFT of signal x (interleaved complex)
        double *fft_x; // C2C FFT(x), length 2*fft_len

        // Pre-allocated workspace (NO allocations in hot paths)
        double *ws_fft1; // FFT workspace 1, length 2*fft_len
        double *ws_fft2; // FFT workspace 2, length 2*fft_len
        double *ws_real; // Real workspace, length fft_len
        double *ws_v;    // Vector workspace, length max(L, K)
        double *ws_u;    // Vector workspace, length max(L, K)
        double *ws_proj; // Projection coefficients for GEMV orthog, length k

        // Batch FFT workspace
        double *ws_batch_u;   // Batch workspace, length 2*fft_len*SSA_BATCH_SIZE
        double *ws_batch_v;   // Batch workspace, length 2*fft_len*SSA_BATCH_SIZE
        double *ws_batch_out; // Output buffer for NOT_INPLACE, length 2*fft_len*SSA_BATCH_SIZE
#else
    double *fft_x;
    double *ws_fft1;
    double *ws_fft2;
#endif

        // Results (column-major for BLAS compatibility)
        double *U;           // Left singular vectors, L × k
        double *V;           // Right singular vectors, K × k
        double *sigma;       // Singular values, k
        double *eigenvalues; // σ², k
        int n_components;

        // State
        bool initialized;
        bool decomposed;
        double total_variance;
    } SSA_Opt;

    // ============================================================================
    // API
    // ============================================================================

    int ssa_opt_init(SSA_Opt *ssa, const double *x, int N, int L);
    int ssa_opt_decompose(SSA_Opt *ssa, int k, int max_iter);
    int ssa_opt_decompose_block(SSA_Opt *ssa, int k, int block_size, int max_iter); // Phase 2
    int ssa_opt_decompose_randomized(SSA_Opt *ssa, int k, int oversampling);        // Phase 3
    int ssa_opt_reconstruct(const SSA_Opt *ssa, const int *group, int n_group, double *output);
    void ssa_opt_free(SSA_Opt *ssa);

    int ssa_opt_get_trend(const SSA_Opt *ssa, double *output);
    int ssa_opt_get_noise(const SSA_Opt *ssa, int noise_start, double *output);
    double ssa_opt_variance_explained(const SSA_Opt *ssa, int start, int end);

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

    // ----------------------------------------------------------------------------
    // Hankel Matrix-Vector Products (OPTIMIZED - no allocations)
    // ----------------------------------------------------------------------------

    // y = X @ v  where X is L×K Hankel matrix
    // Uses pre-allocated workspace
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

        // Zero workspace and REVERSE v into real parts
        // X @ v = conv(x, reverse(v))[K-1 : K-1+L]
        ssa_opt_zero(ws, 2 * n);

        // Note: cblas_dcopy with negative stride is broken on MKL Windows
        // Use simple loop for portability
        for (int i = 0; i < K; i++)
        {
            ws[2 * i] = v[K - 1 - i];
        }

        // FFT of reversed v
#ifdef SSA_USE_MKL
        DftiComputeForward(ssa->fft_handle, ws);
#else
        ssa_opt_fft_builtin(ws, n, -1);
#endif

        // Complex multiply with precomputed FFT(x)
        ssa_opt_complex_mul(ssa->fft_x, ws, ws, n);

        // Inverse FFT
#ifdef SSA_USE_MKL
        DftiComputeBackward(ssa->fft_handle, ws);
#else
        ssa_opt_fft_builtin(ws, n, 1);
#endif

        // Extract result: conv[K-1 : K-1+L]
#ifdef SSA_USE_MKL
        cblas_dcopy(L, ws + 2 * (K - 1), 2, y, 1);
#else
        for (int i = 0; i < L; i++)
        {
            y[i] = ws[2 * (K - 1 + i)];
        }
#endif
    }

    // y = X^T @ u
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

        // X^T @ u = conv(x, reverse(u))[L-1 : L-1+K]

        // Zero workspace and REVERSE u into real parts
        ssa_opt_zero(ws, 2 * n);

        // Note: cblas_dcopy with negative stride is broken on MKL Windows
        // Use simple loop for portability
        for (int i = 0; i < L; i++)
        {
            ws[2 * i] = u[L - 1 - i];
        }

        // FFT of reversed u
#ifdef SSA_USE_MKL
        DftiComputeForward(ssa->fft_handle, ws);
#else
        ssa_opt_fft_builtin(ws, n, -1);
#endif

        // Complex multiply with FFT(x)
        ssa_opt_complex_mul(ssa->fft_x, ws, ws, n);

        // Inverse FFT
#ifdef SSA_USE_MKL
        DftiComputeBackward(ssa->fft_handle, ws);
#else
        ssa_opt_fft_builtin(ws, n, 1);
#endif

        // Extract: conv[L-1 : L-1+K]
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
    // Block Hankel Matrix-Vector Products (Phase 2 - Block Power Method)
    // ============================================================================

#ifdef SSA_USE_MKL

    // Y = X @ V_block  where V_block is K × b, Y is L × b
    // Uses batched C2C FFT with NOT_INPLACE for better performance
    static void ssa_opt_hankel_matvec_block(
        SSA_Opt *ssa,
        const double *V_block, // Input: K × b (column-major)
        double *Y_block,       // Output: L × b (column-major)
        int b                  // Block size
    )
    {
        int K = ssa->K;
        int L = ssa->L;
        int n = ssa->fft_len;
        size_t stride = 2 * n; // Complex interleaved stride

        double *ws_in = ssa->ws_batch_u;    // Input buffer
        double *ws_out = ssa->ws_batch_out; // Output buffer for NOT_INPLACE

        // Process in batches of SSA_BATCH_SIZE
        int col = 0;
        while (col < b)
        {
            int batch_count = (b - col < SSA_BATCH_SIZE) ? (b - col) : SSA_BATCH_SIZE;

            // Pack reversed vectors into input workspace (complex format)
            for (int i = 0; i < batch_count; i++)
            {
                const double *v = &V_block[(col + i) * K];
                double *dst = ws_in + i * stride;

                memset(dst, 0, stride * sizeof(double));
                for (int j = 0; j < K; j++)
                {
                    dst[2 * j] = v[K - 1 - j]; // Real part only
                }
            }

            // Zero remaining slots for partial batch
            if (batch_count < SSA_BATCH_SIZE)
            {
                memset(ws_in + batch_count * stride, 0,
                       (SSA_BATCH_SIZE - batch_count) * stride * sizeof(double));
            }

            // Batch forward FFT (NOT_INPLACE: ws_in -> ws_out)
            DftiComputeForward(ssa->fft_batch_c2c, ws_in, ws_out);

            // Complex multiply each with precomputed FFT(x) using MKL vzMul
            for (int i = 0; i < batch_count; i++)
            {
                double *fft_v = ws_out + i * stride;
                vzMul(n, (const MKL_Complex16 *)ssa->fft_x, (const MKL_Complex16 *)fft_v,
                      (MKL_Complex16 *)fft_v);
            }

            // Batch inverse FFT (NOT_INPLACE: ws_out -> ws_in)
            DftiComputeBackward(ssa->fft_batch_c2c, ws_out, ws_in);

            // Extract results: conv[K-1 : K-1+L] for each column
            for (int i = 0; i < batch_count; i++)
            {
                double *conv = ws_in + i * stride;
                double *y = &Y_block[(col + i) * L];
                cblas_dcopy(L, conv + 2 * (K - 1), 2, y, 1);
            }

            col += batch_count;
        }
    }

    // Y = X^T @ U_block  where U_block is L × b, Y is K × b
    static void ssa_opt_hankel_matvec_T_block(
        SSA_Opt *ssa,
        const double *U_block, // Input: L × b (column-major)
        double *Y_block,       // Output: K × b (column-major)
        int b                  // Block size
    )
    {
        int K = ssa->K;
        int L = ssa->L;
        int n = ssa->fft_len;
        size_t stride = 2 * n;

        double *ws_in = ssa->ws_batch_u;
        double *ws_out = ssa->ws_batch_out;

        int col = 0;
        while (col < b)
        {
            int batch_count = (b - col < SSA_BATCH_SIZE) ? (b - col) : SSA_BATCH_SIZE;

            // Pack reversed vectors
            for (int i = 0; i < batch_count; i++)
            {
                const double *u = &U_block[(col + i) * L];
                double *dst = ws_in + i * stride;

                memset(dst, 0, stride * sizeof(double));
                for (int j = 0; j < L; j++)
                {
                    dst[2 * j] = u[L - 1 - j];
                }
            }

            // Zero remaining slots
            if (batch_count < SSA_BATCH_SIZE)
            {
                memset(ws_in + batch_count * stride, 0,
                       (SSA_BATCH_SIZE - batch_count) * stride * sizeof(double));
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

        // FFT length must accommodate full linear convolution:
        // conv(x, v) has length N + K - 1 = N + (N - L + 1) - 1 = 2N - L
        // Must use next power of 2 >= 2N - L to avoid circular convolution artifacts
        int conv_len = N + ssa->K - 1; // = 2*N - L
        int fft_n = ssa_opt_next_pow2(conv_len);
        ssa->fft_len = fft_n;

        // Allocate all workspace upfront
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

        // Batch workspace - all C2C sized (2 * fft_n per transform)
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

        // =========================================================================
        // Create C2C FFT descriptor for single transforms
        // =========================================================================
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

        // =========================================================================
        // Create batched C2C FFT descriptor - NOT_INPLACE for block matvec
        // =========================================================================
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

        // =========================================================================
        // Create batched C2C FFT descriptor - INPLACE for reconstruction
        // =========================================================================
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

        // Initialize VSL random number generator (Mersenne Twister)
        status = vslNewStream(&ssa->rng, VSL_BRNG_MT2203, 42);
        if (status != VSL_STATUS_OK)
        {
            ssa_opt_free(ssa);
            return -1;
        }
#endif

        // Precompute FFT of x (used by matvec and reconstruction)
        ssa_opt_fft_r2c(ssa, x, N, ssa->fft_x);

        ssa->initialized = true;
        return 0;
    }

    int ssa_opt_decompose(SSA_Opt *ssa, int k, int max_iter)
    {
        if (!ssa || !ssa->initialized || k < 1)
        {
            return -1;
        }

        int L = ssa->L;
        int K = ssa->K;

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

        // Use pre-allocated workspace for iteration vectors
#ifdef SSA_USE_MKL
        double *u = ssa->ws_u;
        double *v = ssa->ws_v;
        double *v_new = (double *)ssa_opt_alloc(K * sizeof(double));

        // Allocate projection workspace for GEMV orthogonalization
        ssa->ws_proj = (double *)ssa_opt_alloc(k * sizeof(double));
        if (!ssa->ws_proj)
            return -1;
#else
        double *u = (double *)ssa_opt_alloc(L * sizeof(double));
        double *v = (double *)ssa_opt_alloc(K * sizeof(double));
        double *v_new = (double *)ssa_opt_alloc(K * sizeof(double));
#endif

        if (!v_new)
            return -1;

#ifndef SSA_USE_MKL
        // Fallback: simple LCG for non-MKL builds
        unsigned int seed = 42;
#endif

        ssa->total_variance = 0.0;

        for (int comp = 0; comp < k; comp++)
        {
#ifdef SSA_USE_MKL
            // VSL random initialization (vectorized, better distribution)
            vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, ssa->rng, K, v, -0.5, 0.5);
#else
            // Fallback: scalar LCG
            for (int i = 0; i < K; i++)
            {
                seed = seed * 1103515245 + 12345;
                v[i] = (double)((seed >> 16) & 0x7fff) / 32768.0 - 0.5;
            }
#endif

            // Orthogonalize against previous v's using GEMV
            if (comp > 0)
            {
#ifdef SSA_USE_MKL
                // proj = V^T @ v  (all dot products at once)
                cblas_dgemv(CblasColMajor, CblasTrans, K, comp,
                            1.0, ssa->V, K, v, 1, 0.0, ssa->ws_proj, 1);
                // v = v - V @ proj  (all axpy at once)
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

            // Power iteration
            for (int iter = 0; iter < max_iter; iter++)
            {
                // u = X @ v
                ssa_opt_hankel_matvec(ssa, v, u);

                // Orthogonalize u against previous u's using GEMV
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

                // v_new = X^T @ u
                ssa_opt_hankel_matvec_T(ssa, u, v_new);

                // Orthogonalize v_new against previous v's using GEMV
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

                // Convergence check (sign-invariant)
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

            // =====================================================================
            // Final orthogonalization
            //
            // To ensure SVD consistency (X @ v = σ*u AND X^T @ u = σ*v),
            // we must recompute v after finalizing u.
            // =====================================================================

            // Step 1: Orthogonalize v against previous components
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

            // Step 2: Compute u = X @ v
            ssa_opt_hankel_matvec(ssa, v, u);

            // Step 3: Orthogonalize u and extract sigma
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

            // Step 4: Recompute v = X^T @ u to ensure SVD consistency
            // Since u is now unit length, X^T @ u gives a vector with ||v|| ≈ σ
            // We scale by 1/σ to get unit v: v = (1/σ) * X^T @ u
            ssa_opt_hankel_matvec_T(ssa, u, v);

            // Step 5: Orthogonalize v against previous components
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

            // Scale v by 1/σ to maintain SVD relationship: X^T @ u = σ * v
            // This ensures v is unit length AND satisfies the SVD equations
            if (sigma > 1e-15)
            {
                ssa_opt_scal(v, K, 1.0 / sigma);
            }

            // Store final u, v, sigma
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
                    // Swap scalars
                    double tmp = ssa->sigma[i];
                    ssa->sigma[i] = ssa->sigma[j];
                    ssa->sigma[j] = tmp;

                    tmp = ssa->eigenvalues[i];
                    ssa->eigenvalues[i] = ssa->eigenvalues[j];
                    ssa->eigenvalues[j] = tmp;

                    // Swap vectors
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

        ssa_opt_free_ptr(v_new);
#ifndef SSA_USE_MKL
        ssa_opt_free_ptr(u);
        ssa_opt_free_ptr(v);
#endif

        ssa->decomposed = true;
        return 0;
    }

    // ============================================================================
    // Phase 2: Block Power Method Decomposition
    // ============================================================================
    // Computes k singular components using subspace iteration with block size b.
    // This is much faster than sequential decomposition for large k.
    //
    // Algorithm:
    //   1. Initialize V_block (K × b) with random orthonormal columns
    //   2. Iterate:
    //      a. U_block = X @ V_block          (batched FFT)
    //      b. U_block = QR(U_block)          (orthonormalize)
    //      c. V_block = X^T @ U_block        (batched FFT)
    //      d. V_block = QR(V_block)          (orthonormalize)
    //   3. Extract singular values via Rayleigh-Ritz
    //   4. Process remaining components in next block
    // ============================================================================

#ifdef SSA_USE_MKL

    int ssa_opt_decompose_block(SSA_Opt *ssa, int k, int block_size, int max_iter)
    {
        if (!ssa || !ssa->initialized || k < 1 || block_size < 1)
        {
            return -1;
        }

        int L = ssa->L;
        int K = ssa->K;

        // Match block_size to SSA_BATCH_SIZE for FFT efficiency
        // If user passes 0 or negative, use SSA_BATCH_SIZE
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

        // Allocate block workspace
        double *V_block = (double *)ssa_opt_alloc(K * b * sizeof(double));
        double *U_block = (double *)ssa_opt_alloc(L * b * sizeof(double));
        double *U_block2 = (double *)ssa_opt_alloc(L * b * sizeof(double)); // For Rayleigh-Ritz
        double *tau_u = (double *)ssa_opt_alloc(b * sizeof(double));        // QR workspace
        double *tau_v = (double *)ssa_opt_alloc(b * sizeof(double));

        // Rayleigh-Ritz workspace
        double *M = (double *)ssa_opt_alloc(b * b * sizeof(double));        // b×b Ritz matrix
        double *U_small = (double *)ssa_opt_alloc(b * b * sizeof(double));  // Left singular vectors of M
        double *Vt_small = (double *)ssa_opt_alloc(b * b * sizeof(double)); // Right singular vectors of M (transposed)
        double *S_small = (double *)ssa_opt_alloc(b * sizeof(double));      // Singular values of M
        double *superb = (double *)ssa_opt_alloc(b * sizeof(double));       // SVD workspace

        // General workspace
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
            // Current block size (may be smaller for last block)
            int cur_b = ssa_opt_min(b, k - comp);

            // Initialize V_block with random values
            vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, ssa->rng, K * cur_b, V_block, -0.5, 0.5);

            // Orthogonalize against previously computed V's
            if (comp > 0)
            {
                // V_block = V_block - V_prev @ (V_prev^T @ V_block)
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

            // Initial QR factorization to orthonormalize V_block
            LAPACKE_dgeqrf(LAPACK_COL_MAJOR, K, cur_b, V_block, K, tau_v);
            LAPACKE_dorgqr(LAPACK_COL_MAJOR, K, cur_b, cur_b, V_block, K, tau_v);

            // Power iteration with periodic QR (not every iteration)
            // QR is expensive; only do it every QR_INTERVAL iterations
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
                if ((iter % QR_INTERVAL == 0) || (iter == max_iter - 1))
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
    // Phase 3: Randomized SVD Decomposition
    // ============================================================================
    //
    // Algorithm (Halko-Martinsson-Tropp):
    //   1. Ω = randn(K, k+p)        - Random test matrix
    //   2. Y = X @ Ω                - Sample range of X (uses batched FFT)
    //   3. Q = orth(Y)              - Orthonormal basis via QR
    //   4. B = X^T @ Q              - Project onto Q (uses batched FFT)
    //   5. [Ũ, Σ, Ṽ] = svd(B)       - Small SVD of K × (k+p)
    //   6. U = Q @ Ṽ, V = Ũ, σ = Σ  - Recover full singular vectors
    //
    // Complexity: O((k+p) × N log N) vs O(k × iter × N log N) for power iteration
    // For k=32, p=8, iter=100: ~80x faster than sequential, ~2.5x faster than block
    //
    int ssa_opt_decompose_randomized(SSA_Opt *ssa, int k, int oversampling)
    {
        if (!ssa || !ssa->initialized || k < 1)
        {
            return -1;
        }

        int L = ssa->L;
        int K = ssa->K;

        // Oversampling parameter (typically 5-10)
        int p = (oversampling <= 0) ? 8 : oversampling;
        int kp = k + p; // Total columns to compute

        // Clamp to valid range
        kp = ssa_opt_min(kp, ssa_opt_min(L, K));
        k = ssa_opt_min(k, kp);

        // =========================================================================
        // Allocate results
        // =========================================================================
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

        // =========================================================================
        // Allocate workspace
        // =========================================================================
        double *Omega = (double *)ssa_opt_alloc(K * kp * sizeof(double)); // Random matrix K × kp
        double *Y = (double *)ssa_opt_alloc(L * kp * sizeof(double));     // Y = X @ Omega, L × kp
        double *Q = (double *)ssa_opt_alloc(L * kp * sizeof(double));     // Q from QR(Y), L × kp
        double *B = (double *)ssa_opt_alloc(K * kp * sizeof(double));     // B = X^T @ Q, K × kp
        double *tau = (double *)ssa_opt_alloc(kp * sizeof(double));       // QR workspace

        // SVD workspace for B (K × kp)
        // Using dgesdd (divide and conquer - faster for medium matrices)
        double *U_svd = (double *)ssa_opt_alloc(K * kp * sizeof(double));   // Left singular vectors
        double *Vt_svd = (double *)ssa_opt_alloc(kp * kp * sizeof(double)); // Right singular vectors (transposed)
        double *S_svd = (double *)ssa_opt_alloc(kp * sizeof(double));       // Singular values

        // Query optimal workspace size for dgesdd
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

        ssa_opt_zero(output, N);

#ifdef SSA_USE_MKL
        // ==========================================================================
        // MKL Batched FFT Reconstruction (Optimized)
        //
        // Optimizations:
        // 1. Pre-scale u by σ during packing (saves N multiplies per component)
        // 2. Always use batch FFT (zero unused slots for partial batches)
        // 3. Use cblas_daxpy for vectorized accumulation with stride-2 access
        // ==========================================================================

        double *ws_u = ssa->ws_batch_u;
        double *ws_v = ssa->ws_batch_v;
        size_t stride = 2 * fft_n; // Stride between transforms (complex interleaved)

        int g = 0;
        while (g < n_group)
        {
            // Determine batch size for this chunk
            int batch_count = 0;
            int batch_indices[SSA_BATCH_SIZE];

            // Collect valid indices for this batch
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

            // ---------------------------------------------------------------------
            // Pack u vectors with σ pre-scaling
            // σ*IFFT(FFT(u)*FFT(v)) = IFFT(FFT(σ*u)*FFT(v)) by FFT linearity
            // This moves σ from the N-length accumulation to L-length packing
            // ---------------------------------------------------------------------
            for (int b = 0; b < batch_count; b++)
            {
                int idx = batch_indices[b];
                double sigma = ssa->sigma[idx];
                const double *u_vec = &ssa->U[idx * L];
                double *dst = ws_u + b * stride;

                ssa_opt_zero(dst, stride);
                for (int i = 0; i < L; i++)
                {
                    dst[2 * i] = sigma * u_vec[i]; // Pre-scale by σ
                }
            }

            // Pack v vectors (no scaling needed)
            for (int b = 0; b < batch_count; b++)
            {
                int idx = batch_indices[b];
                const double *v_vec = &ssa->V[idx * K];
                double *dst = ws_v + b * stride;

                ssa_opt_zero(dst, stride);
                for (int i = 0; i < K; i++)
                {
                    dst[2 * i] = v_vec[i];
                }
            }

            // ---------------------------------------------------------------------
            // Zero unused slots for partial batches (single memset per array)
            // This allows always using batch FFT descriptor
            // ---------------------------------------------------------------------
            int pad = SSA_BATCH_SIZE - batch_count;
            if (pad > 0)
            {
                ssa_opt_zero(ws_u + batch_count * stride, pad * stride);
                ssa_opt_zero(ws_v + batch_count * stride, pad * stride);
            }

            // ---------------------------------------------------------------------
            // Batch forward FFT (use in-place handle for reconstruction)
            // ---------------------------------------------------------------------
            DftiComputeForward(ssa->fft_batch_inplace, ws_u);
            DftiComputeForward(ssa->fft_batch_inplace, ws_v);

            // ---------------------------------------------------------------------
            // Batch complex multiply: ws_u = ws_u * ws_v
            // Only process valid batch elements
            // ---------------------------------------------------------------------
            for (int b = 0; b < batch_count; b++)
            {
                double *u_fft = ws_u + b * stride;
                double *v_fft = ws_v + b * stride;
                vzMul(fft_n, (const MKL_Complex16 *)u_fft, (const MKL_Complex16 *)v_fft,
                      (MKL_Complex16 *)u_fft);
            }

            // ---------------------------------------------------------------------
            // Batch inverse FFT (use in-place handle)
            // ---------------------------------------------------------------------
            DftiComputeBackward(ssa->fft_batch_inplace, ws_u);

            // ---------------------------------------------------------------------
            // Vectorized accumulation using cblas_daxpy
            // output += conv (σ already applied during packing)
            // daxpy with incx=2 handles stride-2 access to real parts
            // ---------------------------------------------------------------------
            for (int b = 0; b < batch_count; b++)
            {
                double *conv = ws_u + b * stride;
                cblas_daxpy(N, 1.0, conv, 2, output, 1);
            }
        }

#else
        // ==========================================================================
        // Non-MKL Fallback: Sequential FFT (with pre-scaling optimization)
        // ==========================================================================

        double *ws1 = ssa->ws_fft1;
        double *ws2 = ssa->ws_fft2;

        for (int g = 0; g < n_group; g++)
        {
            int idx = group[g];
            if (idx < 0 || idx >= ssa->n_components)
                continue;

            double sigma = ssa->sigma[idx];
            const double *u_vec = &ssa->U[idx * L];
            const double *v_vec = &ssa->V[idx * K];

            // FFT of u with σ pre-scaling
            ssa_opt_zero(ws1, 2 * fft_n);
            for (int i = 0; i < L; i++)
                ws1[2 * i] = sigma * u_vec[i];
            ssa_opt_fft_builtin(ws1, fft_n, -1);

            // FFT of v
            ssa_opt_zero(ws2, 2 * fft_n);
            for (int i = 0; i < K; i++)
                ws2[2 * i] = v_vec[i];
            ssa_opt_fft_builtin(ws2, fft_n, -1);

            // Complex multiply
            ssa_opt_complex_mul(ws1, ws2, ws1, fft_n);

            // Inverse FFT
            ssa_opt_fft_builtin(ws1, fft_n, 1);

            // Accumulate (no σ multiply needed - already applied)
            for (int t = 0; t < N; t++)
            {
                output[t] += ws1[2 * t];
            }
        }
#endif

        // Hankel averaging (diagonal averaging)
        for (int t = 0; t < N; t++)
        {
            int count = ssa_opt_min(ssa_opt_min(t + 1, L), ssa_opt_min(K, N - t));
            if (count > 0)
                output[t] /= count;
        }

        return 0;
    }

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