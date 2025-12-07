/*
 * Test for SSA Decomposition Methods with MKL Configuration
 *
 * Tests all three decomposition methods and compares performance:
 *   1. Sequential power iteration (baseline)
 *   2. Block power iteration (batched FFTs)
 *   3. Randomized SVD (fastest for k << L)
 *
 * Also tests the cached FFT optimization for W-correlation.
 *
 * Compile (Windows, Intel oneAPI command prompt):
 *   cl /O2 /DSSA_USE_MKL /DSSA_OPT_IMPLEMENTATION /D_USE_MATH_DEFINES ^
 *      /I"%MKLROOT%\include" ssa_perf_test.c /link /LIBPATH:"%MKLROOT%\lib" ^
 *      mkl_intel_lp64.lib mkl_intel_thread.lib mkl_core.lib libiomp5md.lib
 *
 * Compile (Linux):
 *   gcc -O2 -DSSA_USE_MKL -DSSA_OPT_IMPLEMENTATION -D_USE_MATH_DEFINES \
 *       -I${MKLROOT}/include ssa_perf_test.c -o ssa_perf_test \
 *       -L${MKLROOT}/lib/intel64 -lmkl_intel_lp64 -lmkl_intel_thread \
 *       -lmkl_core -liomp5 -lpthread -lm
 */

#define _USE_MATH_DEFINES
#define SSA_USE_MKL

//#define SSA_USE_FLOAT  // optional, for float mode

#define SSA_OPT_IMPLEMENTATION
#include "ssa_opt_r2c.h"

#define SSA_OPT_ANALYSIS_IMPLEMENTATION
#include "ssa_opt_analysis.h"

#define SSA_OPT_FORECAST_IMPLEMENTATION
#include "ssa_opt_forecast.h"

#define SSA_OPT_ADVANCED_IMPLEMENTATION
#include "ssa_opt_advanced.h"

#include "mkl_config.h"
#include "ssa_opt.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _WIN32
#include <windows.h>
static double get_time_ms(void)
{
    LARGE_INTEGER freq, counter;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart * 1000.0 / freq.QuadPart;
}
#else
static double get_time_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}
#endif

static double correlation(const ssa_real *a, const ssa_real *b, int n)
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

static double rmse(const ssa_real *a, const ssa_real *b, int n)
{
    double sum = 0;
    for (int i = 0; i < n; i++)
    {
        double d = a[i] - b[i];
        sum += d * d;
    }
    return sqrt(sum / n);
}

int main(void)
{
    printf("============================================================\n");
    printf("    SSA Performance Test with MKL Configuration\n");
    printf("============================================================\n\n");

    // =========================================================================
    // Step 1: Initialize MKL with optimal settings
    // =========================================================================
    printf("--- MKL Configuration ---\n");
    mkl_config_ssa_full(1); // Verbose output

    // =========================================================================
    // Step 2: Setup test parameters
    // =========================================================================
    int N = 10000;
    int L = 2500;
    int k = 15;
    int block_size = 32;
    int max_iter = 200;
    int oversampling = 10;

    printf("--- Test Parameters ---\n");
    printf("N = %d (signal length)\n", N);
    printf("L = %d (window length)\n", L);
    printf("K = %d (trajectory matrix columns)\n", N - L + 1);
    printf("k = %d (components to extract)\n", k);
    printf("block_size = %d\n", block_size);
    printf("max_iter = %d\n", max_iter);
    printf("oversampling = %d\n", oversampling);

    // Check if problem fits in cache
    size_t mem_est = mkl_config_ssa_memory_estimate(N, L, k);
    printf("\nEstimated memory: %.2f MB\n", mem_est / (1024.0 * 1024.0));
    printf("Fits in L3 cache: %s\n", mkl_config_fits_in_cache(N, L, k) ? "Yes" : "No");
    printf("\n");

    // =========================================================================
    // Step 3: Create test signal
    // =========================================================================
    ssa_real *x = (ssa_real *)mkl_malloc(N * sizeof(ssa_real), 64);
    if (!x)
    {
        printf("ERROR: Memory allocation failed\n");
        return 1;
    }

    // Signal: trend + two periodic components + noise
    srand(42);
    for (int i = 0; i < N; i++)
    {
        x[i] = 0.01 * i                                   // linear trend
               + 10.0 * sin(2.0 * M_PI * i / 50.0)        // period 50
               + 5.0 * sin(2.0 * M_PI * i / 20.0)         // period 20
               + 0.5 * ((double)rand() / RAND_MAX - 0.5); // noise
    }

    // Allocate buffers for reconstructions
    ssa_real *recon_seq = (ssa_real *)mkl_malloc(N * sizeof(ssa_real), 64);
    ssa_real *recon_block = (ssa_real *)mkl_malloc(N * sizeof(ssa_real), 64);
    ssa_real *recon_rand = (ssa_real *)mkl_malloc(N * sizeof(ssa_real), 64);
    int *group_all = (int *)malloc(k * sizeof(int));
    for (int i = 0; i < k; i++)
        group_all[i] = i;

    // =========================================================================
    // Test 1: Sequential Decomposition
    // =========================================================================
    printf("=== Test 1: Sequential Power Iteration ===\n");

    SSA_Opt ssa_seq = {0};
    ssa_opt_init(&ssa_seq, x, N, L);

    double t0 = get_time_ms();
    ssa_opt_decompose(&ssa_seq, k, max_iter);
    double t_seq = get_time_ms() - t0;

    printf("Time: %.2f ms\n", t_seq);
    printf("Top 5 singular values: ");
    for (int i = 0; i < 5; i++)
        printf("%.2f ", ssa_seq.sigma[i]);
    printf("...\n");

    ssa_opt_reconstruct(&ssa_seq, group_all, k, recon_seq);
    double corr_seq = correlation(recon_seq, x, N);
    printf("Reconstruction correlation: %.6f\n\n", corr_seq);

    // =========================================================================
    // Test 2: Block Decomposition
    // =========================================================================
    printf("=== Test 2: Block Power Iteration ===\n");

    SSA_Opt ssa_block = {0};
    ssa_opt_init(&ssa_block, x, N, L);

    t0 = get_time_ms();
    ssa_opt_decompose_block(&ssa_block, k, block_size, max_iter);
    double t_block = get_time_ms() - t0;

    printf("Time: %.2f ms\n", t_block);
    printf("Top 5 singular values: ");
    for (int i = 0; i < 5; i++)
        printf("%.2f ", ssa_block.sigma[i]);
    printf("...\n");

    ssa_opt_reconstruct(&ssa_block, group_all, k, recon_block);
    double corr_block = correlation(recon_block, x, N);
    printf("Reconstruction correlation: %.6f\n", corr_block);
    printf("Speedup vs sequential: %.2fx\n\n", t_seq / t_block);

    // =========================================================================
    // Test 3: Randomized SVD
    // =========================================================================
    printf("=== Test 3: Randomized SVD ===\n");

    SSA_Opt ssa_rand = {0};
    ssa_opt_init(&ssa_rand, x, N, L);
    ssa_opt_prepare(&ssa_rand, k, oversampling); // Required for randomized SVD

    t0 = get_time_ms();
    ssa_opt_decompose_randomized(&ssa_rand, k, oversampling);
    double t_rand = get_time_ms() - t0;

    printf("Time: %.2f ms\n", t_rand);
    printf("Top 5 singular values: ");
    for (int i = 0; i < 5; i++)
        printf("%.2f ", ssa_rand.sigma[i]);
    printf("...\n");

    ssa_opt_reconstruct(&ssa_rand, group_all, k, recon_rand);
    double corr_rand = correlation(recon_rand, x, N);
    printf("Reconstruction correlation: %.6f\n", corr_rand);
    printf("Speedup vs sequential: %.2fx\n\n", t_seq / t_rand);

    // =========================================================================
    // Test 4: W-Correlation with/without cached FFTs
    // =========================================================================
    printf("=== Test 4: W-Correlation (Cached FFT Optimization) ===\n");

    ssa_real *W = (ssa_real *)mkl_malloc(k * k * sizeof(ssa_real), 64);

    // Without cache
    t0 = get_time_ms();
    ssa_opt_wcorr_matrix(&ssa_rand, W);
    double t_wcorr_nocache = get_time_ms() - t0;
    printf("W-correlation (no cache): %.2f ms\n", t_wcorr_nocache);

    // With cache
    t0 = get_time_ms();
    ssa_opt_cache_ffts((SSA_Opt *)&ssa_rand);
    double t_cache = get_time_ms() - t0;
    printf("Cache FFTs: %.2f ms\n", t_cache);

    t0 = get_time_ms();
    ssa_opt_wcorr_matrix(&ssa_rand, W);
    double t_wcorr_cached = get_time_ms() - t0;
    printf("W-correlation (cached): %.2f ms\n", t_wcorr_cached);
    printf("Speedup with cache: %.2fx\n", t_wcorr_nocache / t_wcorr_cached);

    // Verify W-correlation diagonal is 1.0
    double max_diag_err = 0;
    for (int i = 0; i < k; i++)
    {
        double err = fabs(W[i * k + i] - 1.0);
        if (err > max_diag_err)
            max_diag_err = err;
    }
    printf("W-correlation diagonal max error: %.2e\n\n", max_diag_err);

    // =========================================================================
    // Test 5: Multiple W-correlation calls (cache benefit)
    // =========================================================================
    printf("=== Test 5: Repeated W-Correlation (Cache Stress Test) ===\n");

    int n_repeats = 10;

    // Fresh SSA without cache
    SSA_Opt ssa_fresh = {0};
    ssa_opt_init(&ssa_fresh, x, N, L);
    ssa_opt_prepare(&ssa_fresh, k, oversampling); // Required for randomized SVD
    ssa_opt_decompose_randomized(&ssa_fresh, k, oversampling);

    t0 = get_time_ms();
    for (int r = 0; r < n_repeats; r++)
    {
        ssa_opt_wcorr_matrix(&ssa_fresh, W);
    }
    double t_nocache_total = get_time_ms() - t0;

    // With cache
    ssa_opt_cache_ffts(&ssa_fresh);

    t0 = get_time_ms();
    for (int r = 0; r < n_repeats; r++)
    {
        ssa_opt_wcorr_matrix(&ssa_fresh, W);
    }
    double t_cached_total = get_time_ms() - t0;

    printf("%d W-correlation calls (no cache): %.2f ms\n", n_repeats, t_nocache_total);
    printf("%d W-correlation calls (cached):   %.2f ms\n", n_repeats, t_cached_total);
    printf("Total speedup: %.2fx\n\n", t_nocache_total / t_cached_total);

    ssa_opt_free(&ssa_fresh);

    // =========================================================================
    // Test 6: Streaming Updates (Malloc-Free Hot Path)
    // =========================================================================
    printf("=== Test 6: Streaming Updates ===\n");

    int n_hot_iterations = 100;
    ssa_real *x_copy = (ssa_real *)mkl_malloc(N * sizeof(ssa_real), 64);
    memcpy(x_copy, x, N * sizeof(ssa_real));

    // Setup once with prepare
    SSA_Opt ssa_hot = {0};
    ssa_opt_init(&ssa_hot, x_copy, N, L);
    ssa_opt_prepare(&ssa_hot, k, oversampling); // Pre-allocate workspace

    t0 = get_time_ms();
    for (int iter = 0; iter < n_hot_iterations; iter++)
    {
        // Simulate new data arriving
        x_copy[0] += 0.001;

        ssa_opt_update_signal(&ssa_hot, x_copy);                 // Just memcpy + 1 FFT
        ssa_opt_decompose_randomized(&ssa_hot, k, oversampling); // Reuses workspace
        ssa_opt_reconstruct(&ssa_hot, group_all, k, recon_rand);
    }
    double t_streaming = get_time_ms() - t0;
    printf("Streaming updates (%d iterations): %.2f ms (%.3f ms/iter)\n",
           n_hot_iterations, t_streaming, t_streaming / n_hot_iterations);
    printf("Throughput: %.1f updates/sec\n\n", n_hot_iterations / (t_streaming / 1000.0));

    ssa_opt_free(&ssa_hot);
    mkl_free(x_copy);

    // =========================================================================
    // Comparison Summary
    // =========================================================================
    printf("============================================================\n");
    printf("                    PERFORMANCE SUMMARY\n");
    printf("============================================================\n");
    printf("%-25s %10s %10s %10s\n", "Method", "Time (ms)", "Speedup", "Corr");
    printf("------------------------------------------------------------\n");
    printf("%-25s %10.2f %10s %10.6f\n", "Sequential", t_seq, "1.00x", corr_seq);
    printf("%-25s %10.2f %10.2fx %10.6f\n", "Block", t_block, t_seq / t_block, corr_block);
    printf("%-25s %10.2f %10.2fx %10.6f\n", "Randomized", t_rand, t_seq / t_rand, corr_rand);
    printf("------------------------------------------------------------\n");
    printf("%-25s %10.2f %10s\n", "W-corr (no cache)", t_wcorr_nocache, "-");
    printf("%-25s %10.2f %10.2fx\n", "W-corr (cached)", t_wcorr_cached, t_wcorr_nocache / t_wcorr_cached);
    printf("------------------------------------------------------------\n");
    printf("%-25s %10.2f %10s\n", "Streaming (per iter)", t_streaming / n_hot_iterations, "-");
    printf("============================================================\n\n");

    // =========================================================================
    // Verification
    // =========================================================================
    printf("--- Verification ---\n");

    int pass = 1;

    // Check reconstruction correlations
    if (corr_seq < 0.99)
    {
        printf("[FAIL] Sequential reconstruction correlation < 0.99\n");
        pass = 0;
    }
    if (corr_block < 0.99)
    {
        printf("[FAIL] Block reconstruction correlation < 0.99\n");
        pass = 0;
    }
    if (corr_rand < 0.99)
    {
        printf("[FAIL] Randomized reconstruction correlation < 0.99\n");
        pass = 0;
    }

    // Check method agreement
    double corr_seq_block = correlation(recon_seq, recon_block, N);
    double corr_seq_rand = correlation(recon_seq, recon_rand, N);

    if (corr_seq_block < 0.999)
    {
        printf("[FAIL] Sequential vs Block correlation < 0.999 (%.6f)\n", corr_seq_block);
        pass = 0;
    }
    if (corr_seq_rand < 0.99) // Randomized has more variance
    {
        printf("[FAIL] Sequential vs Randomized correlation < 0.99 (%.6f)\n", corr_seq_rand);
        pass = 0;
    }

    // Check W-correlation validity
    if (max_diag_err > 1e-6)
    {
        printf("[FAIL] W-correlation diagonal error > 1e-6\n");
        pass = 0;
    }

    // Check speedups are positive
    if (t_seq / t_rand < 1.5)
    {
        printf("[WARN] Randomized SVD speedup < 1.5x (may be expected for small k)\n");
    }


    if (pass)
    {
        printf("[PASS] All tests passed!\n");
    }

    // =========================================================================
    // Cleanup
    // =========================================================================
    ssa_opt_free(&ssa_seq);
    ssa_opt_free(&ssa_block);
    ssa_opt_free(&ssa_rand);

    mkl_free(x);
    mkl_free(recon_seq);
    mkl_free(recon_block);
    mkl_free(recon_rand);
    mkl_free(W);
    free(group_all);

    printf("\n=== Test Complete ===\n");
    return pass ? 0 : 1;
}