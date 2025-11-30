/*
 * Test for Phase 2: Block Power Method Decomposition
 *
 * Compile (Windows, Intel oneAPI command prompt):
 *   cl /O2 /DSSA_USE_MKL /DSSA_OPT_IMPLEMENTATION /D_USE_MATH_DEFINES ^
 *      /I"%MKLROOT%\include" ssa_block_test.c /link /LIBPATH:"%MKLROOT%\lib" ^
 *      mkl_intel_lp64.lib mkl_intel_thread.lib mkl_core.lib libiomp5md.lib
 */

#define _USE_MATH_DEFINES
#define SSA_OPT_IMPLEMENTATION
#include "ssa_opt.h"
#include <stdio.h>
#include <time.h>

#ifdef _WIN32
#include <windows.h>
static double get_time_ms() {
    LARGE_INTEGER freq, counter;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart * 1000.0 / freq.QuadPart;
}
#else
static double get_time_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}
#endif

static double correlation(const double* a, const double* b, int n) {
    double sum_a = 0, sum_b = 0, sum_ab = 0, sum_a2 = 0, sum_b2 = 0;
    for (int i = 0; i < n; i++) {
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

int main() {
    printf("=== Phase 2: Block Power Method Test ===\n\n");
    
    // Test parameters
    int N = 1000;      // Larger signal for meaningful benchmark
    int L = 400;
    int k = 20;        // Number of components
    int block_size = 8;
    int max_iter = 100;
    
    printf("Parameters: N=%d, L=%d, K=%d, k=%d, block_size=%d\n\n", 
           N, L, N - L + 1, k, block_size);
    
    // Create test signal: trend + periodic + noise
    double* x = (double*)malloc(N * sizeof(double));
    for (int i = 0; i < N; i++) {
        x[i] = 0.01 * i                           // trend
             + 10.0 * sin(2 * M_PI * i / 50.0)    // period 50
             + 5.0 * sin(2 * M_PI * i / 20.0)     // period 20
             + 0.5 * ((double)rand() / RAND_MAX - 0.5);  // noise
    }
    
    // =========================================================================
    // Test 1: Sequential decomposition (baseline)
    // =========================================================================
    printf("--- Sequential Decomposition (baseline) ---\n");
    
    SSA_Opt ssa_seq;
    if (ssa_opt_init(&ssa_seq, x, N, L) != 0) {
        printf("ERROR: ssa_opt_init failed\n");
        return 1;
    }
    
    double t0 = get_time_ms();
    if (ssa_opt_decompose(&ssa_seq, k, max_iter) != 0) {
        printf("ERROR: ssa_opt_decompose failed\n");
        return 1;
    }
    double t_seq = get_time_ms() - t0;
    
    printf("Time: %.2f ms\n", t_seq);
    printf("Singular values: ");
    for (int i = 0; i < 5; i++) printf("%.2f ", ssa_seq.sigma[i]);
    printf("...\n");
    
    // Reconstruction test
    double* recon_seq = (double*)malloc(N * sizeof(double));
    int* group_all = (int*)malloc(k * sizeof(int));
    for (int i = 0; i < k; i++) group_all[i] = i;
    ssa_opt_reconstruct(&ssa_seq, group_all, k, recon_seq);
    double corr_seq = correlation(recon_seq, x, N);
    printf("Reconstruction correlation: %.6f\n\n", corr_seq);
    
    // =========================================================================
    // Test 2: Block decomposition (Phase 2)
    // =========================================================================
    printf("--- Block Decomposition (Phase 2) ---\n");
    
    SSA_Opt ssa_block;
    if (ssa_opt_init(&ssa_block, x, N, L) != 0) {
        printf("ERROR: ssa_opt_init failed\n");
        return 1;
    }
    
    t0 = get_time_ms();
    if (ssa_opt_decompose_block(&ssa_block, k, block_size, max_iter) != 0) {
        printf("ERROR: ssa_opt_decompose_block failed\n");
        return 1;
    }
    double t_block = get_time_ms() - t0;
    
    printf("Time: %.2f ms\n", t_block);
    printf("Singular values: ");
    for (int i = 0; i < 5; i++) printf("%.2f ", ssa_block.sigma[i]);
    printf("...\n");
    
    // Reconstruction test
    double* recon_block = (double*)malloc(N * sizeof(double));
    ssa_opt_reconstruct(&ssa_block, group_all, k, recon_block);
    double corr_block = correlation(recon_block, x, N);
    printf("Reconstruction correlation: %.6f\n\n", corr_block);
    
    // =========================================================================
    // Comparison
    // =========================================================================
    printf("--- Comparison ---\n");
    printf("Speedup: %.2fx\n", t_seq / t_block);
    
    // Compare singular values
    printf("\nSingular value comparison (seq vs block):\n");
    double max_sigma_diff = 0;
    for (int i = 0; i < k; i++) {
        double diff = fabs(ssa_seq.sigma[i] - ssa_block.sigma[i]);
        double rel_diff = diff / (ssa_seq.sigma[i] + 1e-15);
        if (rel_diff > max_sigma_diff) max_sigma_diff = rel_diff;
        if (i < 5) {
            printf("  sigma[%d]: seq=%.4f, block=%.4f, rel_diff=%.2e\n",
                   i, ssa_seq.sigma[i], ssa_block.sigma[i], rel_diff);
        }
    }
    printf("Max relative sigma difference: %.2e\n", max_sigma_diff);
    
    // Compare reconstructions
    double recon_corr = correlation(recon_seq, recon_block, N);
    printf("Reconstruction correlation (seq vs block): %.6f\n", recon_corr);
    
    // =========================================================================
    // Verdict
    // =========================================================================
    printf("\n--- Verdict ---\n");
    int pass = 1;
    
    if (corr_seq < 0.99) {
        printf("[FAIL] Sequential reconstruction correlation < 0.99\n");
        pass = 0;
    }
    if (corr_block < 0.99) {
        printf("[FAIL] Block reconstruction correlation < 0.99\n");
        pass = 0;
    }
    if (recon_corr < 0.999) {
        printf("[FAIL] Seq vs Block reconstruction correlation < 0.999\n");
        pass = 0;
    }
    if (max_sigma_diff > 0.01) {
        printf("[FAIL] Singular value relative difference > 1%%\n");
        pass = 0;
    }
    
    if (pass) {
        printf("[PASS] Block decomposition matches sequential!\n");
        printf("[PASS] Speedup: %.2fx\n", t_seq / t_block);
    }
    
    // Cleanup
    ssa_opt_free(&ssa_seq);
    ssa_opt_free(&ssa_block);
    free(x);
    free(recon_seq);
    free(recon_block);
    free(group_all);
    
    printf("\n=== Test Complete ===\n");
    return pass ? 0 : 1;
}
