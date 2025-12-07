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

#define SSA_USE_FLOAT // optional, for float mode

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

// Pin current thread to a specific CPU core (P-core 0)
static void pin_thread_to_core(int core_id)
{
    DWORD_PTR mask = (DWORD_PTR)1 << core_id;
    SetThreadAffinityMask(GetCurrentThread(), mask);
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_TIME_CRITICAL);
}

#else
#include <sched.h>
#include <pthread.h>
static double get_time_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

static void pin_thread_to_core(int core_id)
{
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
}
#endif

// ============================================================================
// TLB Warmup - Pre-touch all memory pages to avoid TLB miss storms
// ============================================================================
static void tlb_warmup_ssa(SSA_Opt *ssa)
{
    volatile char dummy = 0;
    const size_t PAGE_SIZE = 4096;

    // Touch workspace pages
    if (ssa->ws_real)
    {
        for (size_t i = 0; i < (size_t)ssa->fft_len * sizeof(ssa_real); i += PAGE_SIZE)
            dummy += ((volatile char *)ssa->ws_real)[i];
    }
    if (ssa->ws_real2)
    {
        for (size_t i = 0; i < (size_t)ssa->fft_len * sizeof(ssa_real); i += PAGE_SIZE)
            dummy += ((volatile char *)ssa->ws_real2)[i];
    }
    if (ssa->ws_complex)
    {
        for (size_t i = 0; i < (size_t)(2 * ssa->r2c_len) * sizeof(ssa_real); i += PAGE_SIZE)
            dummy += ((volatile char *)ssa->ws_complex)[i];
    }
    if (ssa->ws_batch_real)
    {
        for (size_t i = 0; i < (size_t)(ssa->batch_size * ssa->fft_len) * sizeof(ssa_real); i += PAGE_SIZE)
            dummy += ((volatile char *)ssa->ws_batch_real)[i];
    }
    if (ssa->ws_batch_complex)
    {
        for (size_t i = 0; i < (size_t)(ssa->batch_size * 2 * ssa->r2c_len) * sizeof(ssa_real); i += PAGE_SIZE)
            dummy += ((volatile char *)ssa->ws_batch_complex)[i];
    }

    // Touch U, V, sigma arrays
    if (ssa->U)
    {
        for (size_t i = 0; i < (size_t)(ssa->n_components * ssa->L) * sizeof(ssa_real); i += PAGE_SIZE)
            dummy += ((volatile char *)ssa->U)[i];
    }
    if (ssa->V)
    {
        for (size_t i = 0; i < (size_t)(ssa->n_components * ssa->K) * sizeof(ssa_real); i += PAGE_SIZE)
            dummy += ((volatile char *)ssa->V)[i];
    }

    // Touch decomposition workspaces if allocated
    if (ssa->decomp_Omega)
    {
        int kp = ssa->n_components + 10; // oversampling estimate
        for (size_t i = 0; i < (size_t)(ssa->K * kp) * sizeof(ssa_real); i += PAGE_SIZE)
            dummy += ((volatile char *)ssa->decomp_Omega)[i];
    }
    if (ssa->decomp_Y)
    {
        int kp = ssa->n_components + 10;
        for (size_t i = 0; i < (size_t)(ssa->L * kp) * sizeof(ssa_real); i += PAGE_SIZE)
            dummy += ((volatile char *)ssa->decomp_Y)[i];
    }

    (void)dummy; // Suppress unused warning
}

// Warmup buffer for output
static void tlb_warmup_buffer(void *buf, size_t size)
{
    volatile char dummy = 0;
    const size_t PAGE_SIZE = 4096;
    for (size_t i = 0; i < size; i += PAGE_SIZE)
        dummy += ((volatile char *)buf)[i];
    (void)dummy;
}

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
    // Test 7: PRODUCTION LATENCY TEST (N=500, L=375, k=15)
    // Maximum speed configuration with TLB warmup and thread pinning
    // =========================================================================
    printf("=== Test 7: PRODUCTION LATENCY TEST (N=500) ===\n");
    printf("Configuration: Thread pinned to P-core 0, TLB pre-warmed\n\n");

    // Pin to P-core 0 for consistent latency
    pin_thread_to_core(0);

    // Production parameters
    const int N_PROD = 500;
    const int L_PROD = 375; // 0.75 * N - good for financial data
    const int k_PROD = 15;
    const int oversample_PROD = 10;
    const int n_warmup = 50;      // Warmup iterations (not timed)
    const int n_benchmark = 1000; // Benchmark iterations

    // Allocate production buffers
    ssa_real *x_prod = (ssa_real *)mkl_malloc(N_PROD * sizeof(ssa_real), 64);
    ssa_real *recon_prod = (ssa_real *)mkl_malloc(N_PROD * sizeof(ssa_real), 64);
    int *group_prod = (int *)malloc(k_PROD * sizeof(int));
    for (int i = 0; i < k_PROD; i++)
        group_prod[i] = i;

    // Create test signal
    for (int i = 0; i < N_PROD; i++)
    {
        x_prod[i] = (ssa_real)(0.01 * i + 10.0 * sin(2.0 * M_PI * i / 50.0) +
                               5.0 * sin(2.0 * M_PI * i / 20.0) +
                               0.5 * ((double)rand() / RAND_MAX - 0.5));
    }

    // Initialize SSA
    SSA_Opt ssa_prod = {0};
    ssa_opt_init(&ssa_prod, x_prod, N_PROD, L_PROD);
    ssa_opt_prepare(&ssa_prod, k_PROD, oversample_PROD);

    // TLB warmup - pre-touch all memory pages
    printf("Pre-warming TLB and caches...\n");
    tlb_warmup_ssa(&ssa_prod);
    tlb_warmup_buffer(x_prod, N_PROD * sizeof(ssa_real));
    tlb_warmup_buffer(recon_prod, N_PROD * sizeof(ssa_real));

    // CPU warmup iterations (not timed)
    printf("Running %d warmup iterations...\n", n_warmup);
    for (int i = 0; i < n_warmup; i++)
    {
        x_prod[0] += (ssa_real)0.0001;
        ssa_opt_update_signal(&ssa_prod, x_prod);
        ssa_opt_decompose_randomized(&ssa_prod, k_PROD, oversample_PROD);
        ssa_opt_reconstruct(&ssa_prod, group_prod, k_PROD, recon_prod);
    }

    // Re-warm TLB after warmup (in case of eviction)
    tlb_warmup_ssa(&ssa_prod);

    // Benchmark: measure individual iteration times
    printf("Running %d benchmark iterations...\n\n", n_benchmark);

    double *latencies = (double *)malloc(n_benchmark * sizeof(double));
    double total_time = 0;
    double min_latency = 1e9, max_latency = 0;

    for (int i = 0; i < n_benchmark; i++)
    {
        // Simulate new tick
        x_prod[i % N_PROD] += (ssa_real)0.0001;

        t0 = get_time_ms();

        ssa_opt_update_signal(&ssa_prod, x_prod);
        ssa_opt_decompose_randomized(&ssa_prod, k_PROD, oversample_PROD);
        ssa_opt_reconstruct(&ssa_prod, group_prod, k_PROD, recon_prod);

        double latency = get_time_ms() - t0;
        latencies[i] = latency;
        total_time += latency;

        if (latency < min_latency)
            min_latency = latency;
        if (latency > max_latency)
            max_latency = latency;
    }

    // Calculate statistics
    double mean_latency = total_time / n_benchmark;

    // Sort for percentiles
    for (int i = 0; i < n_benchmark - 1; i++)
    {
        for (int j = i + 1; j < n_benchmark; j++)
        {
            if (latencies[j] < latencies[i])
            {
                double tmp = latencies[i];
                latencies[i] = latencies[j];
                latencies[j] = tmp;
            }
        }
    }

    double p50 = latencies[n_benchmark / 2];
    double p95 = latencies[(int)(n_benchmark * 0.95)];
    double p99 = latencies[(int)(n_benchmark * 0.99)];
    double p999 = latencies[(int)(n_benchmark * 0.999)];

    // Calculate standard deviation
    double variance = 0;
    for (int i = 0; i < n_benchmark; i++)
    {
        double diff = latencies[i] - mean_latency;
        variance += diff * diff;
    }
    double stddev = sqrt(variance / n_benchmark);

    printf("╔═══════════════════════════════════════════════════════════╗\n");
    printf("║           PRODUCTION LATENCY RESULTS (N=500)              ║\n");
    printf("╠═══════════════════════════════════════════════════════════╣\n");
    printf("║  Parameters: N=%d, L=%d, k=%d, oversampling=%d          ║\n",
           N_PROD, L_PROD, k_PROD, oversample_PROD);
    printf("║  Iterations: %d (after %d warmup)                       ║\n",
           n_benchmark, n_warmup);
    printf("╠═══════════════════════════════════════════════════════════╣\n");
    printf("║  Mean latency:    %8.2f μs                             ║\n", mean_latency * 1000);
    printf("║  Std deviation:   %8.2f μs                             ║\n", stddev * 1000);
    printf("║  Min latency:     %8.2f μs                             ║\n", min_latency * 1000);
    printf("║  Max latency:     %8.2f μs                             ║\n", max_latency * 1000);
    printf("╠═══════════════════════════════════════════════════════════╣\n");
    printf("║  P50 (median):    %8.2f μs                             ║\n", p50 * 1000);
    printf("║  P95:             %8.2f μs                             ║\n", p95 * 1000);
    printf("║  P99:             %8.2f μs                             ║\n", p99 * 1000);
    printf("║  P99.9:           %8.2f μs                             ║\n", p999 * 1000);
    printf("╠═══════════════════════════════════════════════════════════╣\n");
    printf("║  Throughput:      %8.0f updates/sec                    ║\n",
           1000.0 / mean_latency);
    printf("╚═══════════════════════════════════════════════════════════╝\n\n");

    // Check if we hit 200 μs target
    if (mean_latency * 1000 < 200.0)
    {
        printf("✓ TARGET MET: Mean latency < 200 μs\n\n");
    }
    else if (mean_latency * 1000 < 300.0)
    {
        printf("◐ CLOSE: Mean latency < 300 μs (target: 200 μs)\n\n");
    }
    else
    {
        printf("✗ TARGET MISSED: Mean latency > 300 μs\n\n");
    }

    // Verify correctness
    double corr_prod = correlation(recon_prod, x_prod, N_PROD);
    printf("Reconstruction correlation: %.6f\n\n", corr_prod);

    // Cleanup production test
    free(latencies);
    free(group_prod);
    mkl_free(x_prod);
    mkl_free(recon_prod);
    ssa_opt_free(&ssa_prod);

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
    printf("%-25s %10.2f %10s\n", "Production N=500 (mean)", mean_latency, "-");
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
    if (corr_prod < 0.99)
    {
        printf("[FAIL] Production reconstruction correlation < 0.99\n");
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