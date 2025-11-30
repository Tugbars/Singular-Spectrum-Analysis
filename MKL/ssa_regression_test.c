// ============================================================================
// SSA Regression Test Suite
// Complete test coverage for correctness, stability, and consistency
// ============================================================================

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "../ssa_opt.h"   // your full optimized SSA header

#ifndef EPS
#define EPS 1e-9
#endif

static double max(double a, double b) { return a > b ? a : b; }
static double rand01() { return rand() / (double)RAND_MAX; }

// -----------------------------------------------------------------------------
// Helper: compute correlation between two vectors
// -----------------------------------------------------------------------------
static double correlation(const double *a, const double *b, int n)
{
    double ma = 0, mb = 0;
    for (int i = 0; i < n; i++) { ma += a[i]; mb += b[i]; }
    ma /= n; mb /= n;

    double num = 0, da = 0, db = 0;
    for (int i = 0; i < n; i++) {
        double xa = a[i] - ma;
        double xb = b[i] - mb;
        num += xa * xb;
        da += xa * xa;
        db += xb * xb;
    }
    return num / sqrt(da * db + 1e-15);
}

// -----------------------------------------------------------------------------
// Helper: explicit Hankel multiply (reference slow path)
// X is L × K Hankel from x
// -----------------------------------------------------------------------------
static void hankel_explicit_mv(const double* x, int N, int L, const double* v, double* y)
{
    int K = N - L + 1;
    for (int i = 0; i < L; i++) {
        double sum = 0;
        for (int j = 0; j < K; j++) {
            sum += x[i + j] * v[j];
        }
        y[i] = sum;
    }
}

// -----------------------------------------------------------------------------
// Test 1: Basic decomposition correctness
// -----------------------------------------------------------------------------
static int test_basic_svd()
{
    printf("[TEST] basic sequential decomposition\n");

    int N = 300, L = 120;
    double* x = malloc(N * sizeof(double));

    // simple test signal: trend + sinusoid
    for (int i = 0; i < N; i++)
        x[i] = 0.01*i + sin(0.05*i);

    SSA_Opt ssa = {0};
    if (ssa_opt_init(&ssa, x, N, L) != 0) return -1;

    if (ssa_opt_decompose(&ssa, 6, 150) != 0) return -1;

    // 1) orthogonality checks
    for (int i = 0; i < 6; i++) {
        double nU = 0, nV = 0;
        for (int j = 0; j < L; j++) nU += ssa.U[i*L + j] * ssa.U[i*L + j];
        for (int j = 0; j < ssa.K; j++) nV += ssa.V[i*ssa.K + j] * ssa.V[i*ssa.K + j];
        if (fabs(nU - 1.0) > 1e-6 || fabs(nV - 1.0) > 1e-6) {
            printf("  FAIL: non-unit vectors\n");
            return -1;
        }
    }

    // 2) decreasing singular values
    for (int i = 1; i < 6; i++) {
        if (ssa.sigma[i] > ssa.sigma[i-1] + 1e-12) {
            printf("  FAIL: singular values not sorted\n");
            return -1;
        }
    }

    printf("  PASS\n");
    ssa_opt_free(&ssa);
    free(x);
    return 0;
}

// -----------------------------------------------------------------------------
// Test 2: matvec correctness vs explicit Hankel
// -----------------------------------------------------------------------------
static int test_matvec_correctness()
{
    printf("[TEST] hankel matvec correctness\n");

    int N = 200, L = 80;
    int K = N - L + 1;
    double* x = malloc(N*sizeof(double));
    for (int i = 0; i < N; i++) x[i] = rand01();

    double* v = malloc(K*sizeof(double));
    for (int i = 0; i < K; i++) v[i] = rand01() - 0.5;

    SSA_Opt ssa = {0};
    if (ssa_opt_init(&ssa, x, N, L) != 0) return -1;

    double* y1 = malloc(L*sizeof(double));
    double* y2 = malloc(L*sizeof(double));

    ssa_opt_hankel_matvec(&ssa, v, y1);
    hankel_explicit_mv(x, N, L, v, y2);

    double corr = correlation(y1, y2, L);
    if (corr < 0.999999) {
        printf("  FAIL: matvec correlation = %.12f\n", corr);
        return -1;
    }

    printf("  PASS\n");
    free(x); free(v); free(y1); free(y2);
    ssa_opt_free(&ssa);
    return 0;
}

// -----------------------------------------------------------------------------
// Test 3: reconstruction correctness (correlation)
// -----------------------------------------------------------------------------
static int test_reconstruction()
{
    printf("[TEST] reconstruction accuracy\n");

    int N = 500;
    int L = 200;
    double* x = malloc(N*sizeof(double));
    for (int i = 0; i < N; i++)
        x[i] = sin(0.02*i) + 0.001*i;

    SSA_Opt ssa = {0};
    ssa_opt_init(&ssa, x, N, L);
    ssa_opt_decompose(&ssa, 6, 120);

    double* recon = calloc(N, sizeof(double));
    int group[] = {0,1};       // trend + first component
    ssa_opt_reconstruct(&ssa, group, 2, recon);

    double corr = correlation(x, recon, N);
    if (corr < 0.9995) {
        printf("  FAIL: reconstruction corr = %.12f\n", corr);
        return -1;
    }

    printf("  PASS\n");
    ssa_opt_free(&ssa);
    free(x); free(recon);
    return 0;
}

// -----------------------------------------------------------------------------
// Test 4: block-power vs sequential (correlation of singular vectors)
// -----------------------------------------------------------------------------
static int test_block_vs_sequential()
{
    printf("[TEST] block vs sequential correlation\n");

    int N = 2000, L = 900;
    double* x = malloc(N*sizeof(double));
    for (int i = 0; i < N; i++) x[i] = sin(0.03*i) + 0.3*sin(0.05*i);

    SSA_Opt A = {0}, B = {0};
    ssa_opt_init(&A, x, N, L);
    ssa_opt_init(&B, x, N, L);

    ssa_opt_decompose(&A, 8, 120);             // sequential
    ssa_opt_decompose_block(&B, 8, 8, 80);     // block

    for (int c = 0; c < 8; c++) {
        double corrU = correlation(&A.U[c*L], &B.U[c*L], L);
        if (fabs(corrU) < 0.999) {
            printf("  FAIL comp=%d U corr=%.6f\n", c, corrU);
            return -1;
        }
    }

    printf("  PASS\n");
    ssa_opt_free(&A);
    ssa_opt_free(&B);
    free(x);
    return 0;
}

// -----------------------------------------------------------------------------
// Test 5: randomized SVD vs exact
// -----------------------------------------------------------------------------
static int test_randomized_svd()
{
    printf("[TEST] randomized SVD correctness\n");

    int N = 1500, L = 700;
    double* x = malloc(N*sizeof(double));
    for (int i = 0; i < N; i++)
        x[i] = 0.4*sin(0.02*i) + 0.3*sin(0.05*i);

    SSA_Opt A = {0}, B = {0};
    ssa_opt_init(&A, x, N, L);
    ssa_opt_init(&B, x, N, L);

    ssa_opt_decompose(&A, 6, 120);
    ssa_opt_decompose_randomized(&B, 6, 8);

    for (int c = 0; c < 6; c++) {
        double corrU = correlation(&A.U[c*L], &B.U[c*L], L);
        if (fabs(corrU) < 0.995) {
            printf("  FAIL: randomized U corr=%.6f\n", corrU);
            return -1;
        }
    }

    printf("  PASS\n");
    ssa_opt_free(&A);
    ssa_opt_free(&B);
    free(x);
    return 0;
}

// -----------------------------------------------------------------------------
// Test 6: variance explained monotonicity
// -----------------------------------------------------------------------------
static int test_variance_explained()
{
    printf("[TEST] variance explained monotonicity\n");

    int N = 300, L = 150;
    double* x = malloc(N*sizeof(double));
    for (int i = 0; i < N; i++) x[i] = sin(0.04*i);

    SSA_Opt ssa = {0};
    ssa_opt_init(&ssa, x, N, L);
    ssa_opt_decompose(&ssa, 10, 100);

    double prev = -1;
    for (int i = 0; i < 10; i++) {
        double v = ssa_opt_variance_explained(&ssa, 0, i);
        if (v < prev - 1e-12) {
            printf("  FAIL: non-monotonic variance\n");
            return -1;
        }
        prev = v;
    }

    printf("  PASS\n");
    ssa_opt_free(&ssa);
    free(x);
    return 0;
}

// -----------------------------------------------------------------------------
// MAIN
// -----------------------------------------------------------------------------
int main()
{
    printf("==========================================\n");
    printf("   SSA Regression Test Suite\n");
    printf("==========================================\n\n");

    int fails = 0;

    fails += test_basic_svd();
    fails += test_matvec_correctness();
    fails += test_reconstruction();
    fails += test_block_vs_sequential();
    fails += test_randomized_svd();
    fails += test_variance_explained();

    printf("\n==========================================\n");
    if (fails == 0)
        printf("   ALL TESTS PASSED ✓✓✓\n");
    else
        printf("   %d TEST(S) FAILED ✗\n", fails);
    printf("==========================================\n");

    return fails;
}
