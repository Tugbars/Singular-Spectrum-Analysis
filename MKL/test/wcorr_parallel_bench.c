/*
 * Benchmark: W-Correlation Parallelization
 * =========================================
 * 
 * Compares:
 *   - Original: Sequential loop of n IFFTs
 *   - Fast: OpenMP complex multiply + batched IFFT + OpenMP G build
 *
 * Build: cmake --build . --config Release
 * Run:   .\Release\wcorr_parallel_bench.exe
 */

#define _USE_MATH_DEFINES
#define SSA_OPT_IMPLEMENTATION
#define SSA_OPT_IMPLEMENTATION
#include "ssa_opt_r2c.h"

#define SSA_OPT_ANALYSIS_IMPLEMENTATION
#include "ssa_opt_analysis.h"

#define SSA_OPT_FORECAST_IMPLEMENTATION
#include "ssa_opt_forecast.h"

#define SSA_OPT_ADVANCED_IMPLEMENTATION
#include "ssa_opt_advanced.h"
#include "mkl_config.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

/* ============================================================================
 * Timing
 * ============================================================================ */

#ifdef _WIN32
#include <windows.h>
static ssa_real get_time_ms(void)
{
    LARGE_INTEGER freq, count;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&count);
    return (ssa_real)count.QuadPart / freq.QuadPart * 1000.0;
}
#else
#include <time.h>
static ssa_real get_time_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}
#endif

/* ============================================================================
 * Benchmark
 * ============================================================================ */

typedef int (*wcorr_fn)(const SSA_Opt*, ssa_real*);

static ssa_real benchmark_wcorr(wcorr_fn fn, const SSA_Opt *ssa, int runs)
{
    int n = ssa->n_components;
    int r;
    ssa_real t0, t1;
    ssa_real *W = (ssa_real *)mkl_malloc(n * n * sizeof(ssa_real), 64);

    /* Warmup */
    fn(ssa, W);

    t0 = get_time_ms();
    for (r = 0; r < runs; r++)
        fn(ssa, W);
    t1 = get_time_ms();

    mkl_free(W);
    return (t1 - t0) / runs;
}

static ssa_real verify_results(const SSA_Opt *ssa)
{
    int n = ssa->n_components;
    int i;
    ssa_real max_diff = 0.0;
    ssa_real *W1 = (ssa_real *)mkl_malloc(n * n * sizeof(ssa_real), 64);
    ssa_real *W2 = (ssa_real *)mkl_malloc(n * n * sizeof(ssa_real), 64);

    ssa_opt_wcorr_matrix(ssa, W1);
    ssa_opt_wcorr_matrix_fast(ssa, W2);

    for (i = 0; i < n * n; i++)
    {
        ssa_real d = fabs(W1[i] - W2[i]);
        if (d > max_diff) max_diff = d;
    }

    mkl_free(W1);
    mkl_free(W2);
    return max_diff;
}

int main(void)
{
    int configs[][3] = {
        {1000, 250, 30},
        {5000, 1250, 50},
        {10000, 2500, 50},
        {20000, 5000, 50},
        {10000, 2500, 100}
    };
    int n_configs = sizeof(configs) / sizeof(configs[0]);
    int runs = 10;
    int c, i;

    printf("=============================================================\n");
    printf("W-Correlation Parallelization Benchmark\n");
    printf("=============================================================\n\n");

    printf("OpenMP threads: %d\n", omp_get_max_threads());
    mkl_config_ssa_full(1);
    printf("\n");

    for (c = 0; c < n_configs; c++)
    {
        int N = configs[c][0];
        int L = configs[c][1];
        int k = configs[c][2];
        ssa_real *x;
        SSA_Opt ssa = {0};
        ssa_real t_orig, t_fast, max_diff;

        printf("-------------------------------------------------------------\n");
        printf("N=%d, L=%d, k=%d\n", N, L, k);
        printf("-------------------------------------------------------------\n");

        /* Generate test signal */
        x = (ssa_real *)mkl_malloc(N * sizeof(ssa_real), 64);
        for (i = 0; i < N; i++)
            x[i] = sin(2.0 * M_PI * i / 50.0) + 0.3 * ((ssa_real)rand() / RAND_MAX - 0.5);

        /* Decompose */
        ssa_opt_init(&ssa, x, N, L);
        ssa_opt_decompose_randomized(&ssa, k, 8);
        ssa_opt_cache_ffts(&ssa);

        /* Benchmark */
        t_orig = benchmark_wcorr(ssa_opt_wcorr_matrix, &ssa, runs);
        t_fast = benchmark_wcorr(ssa_opt_wcorr_matrix_fast, &ssa, runs);

        printf("  Original (sequential): %7.3f ms\n", t_orig);
        printf("  Fast (batched+OpenMP): %7.3f ms\n", t_fast);
        printf("  Speedup:               %7.2fx\n", t_orig / t_fast);

        /* Verify */
        max_diff = verify_results(&ssa);
        printf("  Max difference:        %.2e %s\n", max_diff,
               max_diff < 1e-10 ? "(OK)" : "(MISMATCH!)");

        ssa_opt_free(&ssa);
        mkl_free(x);
        printf("\n");
    }

    printf("=============================================================\n");
    return 0;
}
