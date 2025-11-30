/*
 * MKL-specific diagnostic test for SSA
 *
 * Compile (Windows, Intel oneAPI command prompt):
 *   cl /O2 /DSSA_USE_MKL /DSSA_OPT_IMPLEMENTATION /I"%MKLROOT%\include" ^
 *      ssa_mkl_debug.c /link /LIBPATH:"%MKLROOT%\lib" mkl_intel_lp64.lib ^
 *      mkl_intel_thread.lib mkl_core.lib libiomp5md.lib
 */

#define _USE_MATH_DEFINES
#define SSA_OPT_IMPLEMENTATION
#include "ssa_opt.h"
#include <stdio.h>

// Direct Hankel matvec (ground truth - no FFT)
static void hankel_matvec_direct(const double *x, int N, int L, const double *v, double *y)
{
    int K = N - L + 1;
    for (int i = 0; i < L; i++)
    {
        y[i] = 0;
        for (int j = 0; j < K; j++)
        {
            y[i] += x[i + j] * v[j];
        }
    }
}

// Direct X^T @ u (ground truth)
static void hankel_matvec_T_direct(const double *x, int N, int L, const double *u, double *y)
{
    int K = N - L + 1;
    for (int j = 0; j < K; j++)
    {
        y[j] = 0;
        for (int i = 0; i < L; i++)
        {
            y[j] += x[i + j] * u[i];
        }
    }
}

// Direct reconstruction (ground truth - no FFT)
static void reconstruct_direct(int N, int L, int K, double sigma,
                               const double *u, const double *v, double *output)
{
    for (int t = 0; t < N; t++)
    {
        double sum = 0;
        int count = 0;
        for (int i = 0; i < L; i++)
        {
            int j = t - i;
            if (j >= 0 && j < K)
            {
                sum += sigma * u[i] * v[j];
                count++;
            }
        }
        output[t] = (count > 0) ? sum / count : 0;
    }
}

static double vec_diff(const double *a, const double *b, int n)
{
    double sum = 0;
    for (int i = 0; i < n; i++)
    {
        double d = a[i] - b[i];
        sum += d * d;
    }
    return sqrt(sum);
}

static double correlation(const double *a, const double *b, int n)
{
    double sum_a = 0, sum_b = 0, sum_ab = 0, sum_a2 = 0, sum_b2 = 0;
    for (int i = 0; i < n; i++)
    {
        sum_a += a[i];
        sum_b += b[i];
        sum_ab += a[i] * b[i];
        sum_a2 += a[i] * a[i];
        sum_b2 += b[i] * b[i];
    }
    double num = n * sum_ab - sum_a * sum_b;
    double den = sqrt((n * sum_a2 - sum_a * sum_a) * (n * sum_b2 - sum_b * sum_b));
    return num / (den + 1e-15);
}

int main()
{
    // Force unbuffered output for Windows
    setvbuf(stdout, NULL, _IONBF, 0);
    setvbuf(stderr, NULL, _IONBF, 0);

    fprintf(stderr, "=== SSA MKL Diagnostic Test ===\n\n");
    printf("=== SSA MKL Diagnostic Test ===\n\n");
    fflush(stdout);

    // Test parameters
    int N = 100;
    int L = 40;
    int K = N - L + 1; // K = 61

    // Create test signal
    double *x = (double *)malloc(N * sizeof(double));
    for (int i = 0; i < N; i++)
    {
        x[i] = 0.1 * i + 5.0 * sin(2 * M_PI * i / 20.0);
    }

    printf("Parameters: N=%d, L=%d, K=%d\n\n", N, L, K);

    // Initialize SSA
    SSA_Opt ssa;
    if (ssa_opt_init(&ssa, x, N, L) != 0)
    {
        printf("ERROR: ssa_opt_init failed!\n");
        return 1;
    }
    printf("FFT length: %d\n\n", ssa.fft_len);

    // =========================================================================
    // TEST 1: Verify matvec (X @ v)
    // =========================================================================
    printf("--- TEST 1: Hankel matvec (X @ v) ---\n");

    double *v_test = (double *)malloc(K * sizeof(double));
    double *y_fft = (double *)malloc(L * sizeof(double));
    double *y_direct = (double *)malloc(L * sizeof(double));

    // Create test vector
    for (int i = 0; i < K; i++)
        v_test[i] = sin(0.1 * i);

    // FFT-based matvec
    ssa_opt_hankel_matvec(&ssa, v_test, y_fft);

    // Direct matvec (ground truth)
    hankel_matvec_direct(x, N, L, v_test, y_direct);

    double diff1 = vec_diff(y_fft, y_direct, L);
    printf("||y_fft - y_direct|| = %.6e\n", diff1);
    printf("y_fft[0..4]:    %.4f %.4f %.4f %.4f %.4f\n",
           y_fft[0], y_fft[1], y_fft[2], y_fft[3], y_fft[4]);
    printf("y_direct[0..4]: %.4f %.4f %.4f %.4f %.4f\n",
           y_direct[0], y_direct[1], y_direct[2], y_direct[3], y_direct[4]);
    printf("Result: %s\n\n", diff1 < 1e-10 ? "PASS" : "FAIL");

    // =========================================================================
    // TEST 2: Verify matvec_T (X^T @ u)
    // =========================================================================
    printf("--- TEST 2: Hankel matvec_T (X^T @ u) ---\n");

    double *u_test = (double *)malloc(L * sizeof(double));
    double *z_fft = (double *)malloc(K * sizeof(double));
    double *z_direct = (double *)malloc(K * sizeof(double));

    // Create test vector
    for (int i = 0; i < L; i++)
        u_test[i] = cos(0.15 * i);

    // FFT-based matvec_T
    ssa_opt_hankel_matvec_T(&ssa, u_test, z_fft);

    // Direct matvec_T (ground truth)
    hankel_matvec_T_direct(x, N, L, u_test, z_direct);

    double diff2 = vec_diff(z_fft, z_direct, K);
    printf("||z_fft - z_direct|| = %.6e\n", diff2);
    printf("z_fft[0..4]:    %.4f %.4f %.4f %.4f %.4f\n",
           z_fft[0], z_fft[1], z_fft[2], z_fft[3], z_fft[4]);
    printf("z_direct[0..4]: %.4f %.4f %.4f %.4f %.4f\n",
           z_direct[0], z_direct[1], z_direct[2], z_direct[3], z_direct[4]);
    printf("Result: %s\n\n", diff2 < 1e-10 ? "PASS" : "FAIL");

    // =========================================================================
    // TEST 3: Verify adjoint property <Xv, u> = <v, X^T u>
    // =========================================================================
    printf("--- TEST 3: Adjoint property ---\n");

    // <Xv, u> using y_fft from test 1 and u_test
    double inner1 = 0;
    for (int i = 0; i < L; i++)
        inner1 += y_fft[i] * u_test[i];

    // <v, X^T u> using v_test and z_fft from test 2
    double inner2 = 0;
    for (int i = 0; i < K; i++)
        inner2 += v_test[i] * z_fft[i];

    printf("<Xv, u> = %.10f\n", inner1);
    printf("<v, X^Tu> = %.10f\n", inner2);
    printf("Difference: %.6e\n", fabs(inner1 - inner2));
    printf("Result: %s\n\n", fabs(inner1 - inner2) < 1e-10 ? "PASS" : "FAIL");

    // =========================================================================
    // TEST 4: Decomposition
    // =========================================================================
    printf("--- TEST 4: Decomposition ---\n");

    if (ssa_opt_decompose(&ssa, 5, 100) != 0)
    {
        printf("ERROR: ssa_opt_decompose failed!\n");
        return 1;
    }

    printf("Singular values:\n");
    for (int i = 0; i < 5; i++)
    {
        printf("  sigma[%d] = %.6f\n", i, ssa.sigma[i]);
    }

    // Verify SVD property: X @ v = sigma * u
    printf("\nSVD verification (X @ v = sigma * u):\n");
    double *Xv = (double *)malloc(L * sizeof(double));
    for (int comp = 0; comp < 3; comp++)
    {
        const double *v_comp = &ssa.V[comp * K];
        const double *u_comp = &ssa.U[comp * L];
        double sigma = ssa.sigma[comp];

        ssa_opt_hankel_matvec(&ssa, v_comp, Xv);

        double diff = 0, norm = 0;
        for (int i = 0; i < L; i++)
        {
            double expected = sigma * u_comp[i];
            diff += (Xv[i] - expected) * (Xv[i] - expected);
            norm += expected * expected;
        }
        double rel_err = sqrt(diff) / (sqrt(norm) + 1e-15);
        printf("  Component %d: ||Xv - σu|| / ||σu|| = %.6e %s\n",
               comp, rel_err, rel_err < 1e-6 ? "OK" : "BAD");
    }

    // Verify vector norms
    printf("\nVector norms (should be 1.0):\n");
    for (int comp = 0; comp < 3; comp++)
    {
        const double *u_comp = &ssa.U[comp * L];
        const double *v_comp = &ssa.V[comp * K];
        double norm_u = 0, norm_v = 0;
        for (int i = 0; i < L; i++)
            norm_u += u_comp[i] * u_comp[i];
        for (int i = 0; i < K; i++)
            norm_v += v_comp[i] * v_comp[i];
        printf("  Component %d: ||u|| = %.6f, ||v|| = %.6f\n",
               comp, sqrt(norm_u), sqrt(norm_v));
    }
    printf("\n");

    // =========================================================================
    // TEST 5: Reconstruction
    // =========================================================================
    printf("--- TEST 5: Reconstruction ---\n");

    // Full reconstruction
    int *all_group = (int *)malloc(5 * sizeof(int));
    for (int i = 0; i < 5; i++)
        all_group[i] = i;

    double *recon_fft = (double *)malloc(N * sizeof(double));
    ssa_opt_reconstruct(&ssa, all_group, 5, recon_fft);

    // Direct reconstruction for comparison
    double *recon_direct = (double *)malloc(N * sizeof(double));
    memset(recon_direct, 0, N * sizeof(double));

    for (int comp = 0; comp < 5; comp++)
    {
        double sigma = ssa.sigma[comp];
        const double *u_comp = &ssa.U[comp * L];
        const double *v_comp = &ssa.V[comp * K];

        double *temp = (double *)malloc(N * sizeof(double));
        reconstruct_direct(N, L, K, sigma, u_comp, v_comp, temp);

        for (int t = 0; t < N; t++)
            recon_direct[t] += temp[t];
        free(temp);
    }

    double corr_fft_x = correlation(recon_fft, x, N);
    double corr_direct_x = correlation(recon_direct, x, N);
    double corr_fft_direct = correlation(recon_fft, recon_direct, N);

    printf("correlation(recon_fft, x) = %.6f\n", corr_fft_x);
    printf("correlation(recon_direct, x) = %.6f\n", corr_direct_x);
    printf("correlation(recon_fft, recon_direct) = %.6f\n", corr_fft_direct);

    printf("\nFirst 5 values:\n");
    printf("  x:            %.4f %.4f %.4f %.4f %.4f\n", x[0], x[1], x[2], x[3], x[4]);
    printf("  recon_fft:    %.4f %.4f %.4f %.4f %.4f\n",
           recon_fft[0], recon_fft[1], recon_fft[2], recon_fft[3], recon_fft[4]);
    printf("  recon_direct: %.4f %.4f %.4f %.4f %.4f\n",
           recon_direct[0], recon_direct[1], recon_direct[2], recon_direct[3], recon_direct[4]);

    printf("\nResult: %s\n\n", corr_fft_x > 0.99 ? "PASS" : "FAIL");

    // =========================================================================
    // TEST 6: Single component reconstruction comparison
    // =========================================================================
    printf("--- TEST 6: Single component reconstruction ---\n");

    int single_group[1] = {0};
    double *recon_single_fft = (double *)malloc(N * sizeof(double));
    double *recon_single_direct = (double *)malloc(N * sizeof(double));

    ssa_opt_reconstruct(&ssa, single_group, 1, recon_single_fft);
    reconstruct_direct(N, L, K, ssa.sigma[0], &ssa.U[0], &ssa.V[0], recon_single_direct);

    double diff_single = vec_diff(recon_single_fft, recon_single_direct, N);
    double corr_single = correlation(recon_single_fft, recon_single_direct, N);

    printf("||recon_fft - recon_direct|| = %.6e\n", diff_single);
    printf("correlation = %.6f\n", corr_single);
    printf("First 5: FFT=%.4f,%.4f,%.4f  Direct=%.4f,%.4f,%.4f\n",
           recon_single_fft[0], recon_single_fft[1], recon_single_fft[2],
           recon_single_direct[0], recon_single_direct[1], recon_single_direct[2]);
    printf("Result: %s\n\n", corr_single > 0.999 ? "PASS" : "FAIL");

    // Cleanup
    ssa_opt_free(&ssa);
    free(x);
    free(v_test);
    free(y_fft);
    free(y_direct);
    free(u_test);
    free(z_fft);
    free(z_direct);
    free(Xv);
    free(all_group);
    free(recon_fft);
    free(recon_direct);
    free(recon_single_fft);
    free(recon_single_direct);

    printf("=== Diagnostic Complete ===\n");
    return 0;
}