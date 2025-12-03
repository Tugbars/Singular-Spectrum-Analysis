/*
 * SSA Optimized Test Suite (R2C Version)
 *
 * Compares ssa_opt_r2c.h (R2C optimized) performance.
 *
 * Compile with MKL:
 *   source /opt/intel/oneapi/setvars.sh
 *   gcc -O3 -march=native -I${MKLROOT}/include \
 *       -o ssa_opt_test ssa_opt_test.c \
 *       -L${MKLROOT}/lib/intel64 -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core \
 *       -liomp5 -lpthread -lm
 *
 * Run:
 *   ./ssa_opt_test           # Tests only
 *   ./ssa_opt_test --bench   # Include benchmarks
 */

#define SSA_OPT_IMPLEMENTATION
#include "ssa_opt.h"
#include "mkl_config.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

// ============================================================================
// Test Framework
// ============================================================================

#define TEST_PASS "\033[32mPASS\033[0m"
#define TEST_FAIL "\033[31mFAIL\033[0m"

static int g_total = 0, g_passed = 0, g_failed = 0;
static int g_benchmarks = 0;

#define ASSERT_TRUE(cond, msg)                                         \
    do                                                                 \
    {                                                                  \
        g_total++;                                                     \
        if (cond)                                                      \
        {                                                              \
            g_passed++;                                                \
        }                                                              \
        else                                                           \
        {                                                              \
            g_failed++;                                                \
            printf("  [%s] %s (line %d)\n", TEST_FAIL, msg, __LINE__); \
        }                                                              \
    } while (0)

#define ASSERT_EQ(a, b, msg) ASSERT_TRUE((a) == (b), msg)
#define ASSERT_GT(a, b, msg) ASSERT_TRUE((a) > (b), msg)
#define ASSERT_LT(a, b, msg) ASSERT_TRUE((a) < (b), msg)

#define RUN_TEST(fn)                  \
    do                                \
    {                                 \
        printf("\n[TEST] %s\n", #fn); \
        fn();                         \
    } while (0)

// Timing
#ifdef __linux__
#include <time.h>
static inline double get_time_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}
#elif defined(_WIN32)
#include <windows.h>
static inline double get_time_ms(void)
{
    LARGE_INTEGER freq, count;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&count);
    return (double)count.QuadPart * 1000.0 / (double)freq.QuadPart;
}
#else
static inline double get_time_ms(void)
{
    return (double)clock() * 1000.0 / CLOCKS_PER_SEC;
}
#endif

// ============================================================================
// Signal Generation
// ============================================================================

static unsigned int g_seed = 42;

static double randn(void)
{
    static int have = 0;
    static double spare;
    if (have)
    {
        have = 0;
        return spare;
    }
    double u, v, s;
    do
    {
        g_seed = g_seed * 1103515245 + 12345;
        u = (double)((g_seed >> 16) & 0x7fff) / 16384.0 - 1.0;
        g_seed = g_seed * 1103515245 + 12345;
        v = (double)((g_seed >> 16) & 0x7fff) / 16384.0 - 1.0;
        s = u * u + v * v;
    } while (s >= 1 || s == 0);
    s = sqrt(-2 * log(s) / s);
    spare = v * s;
    have = 1;
    return u * s;
}

static void generate_signal(double *x, int N)
{
    for (int i = 0; i < N; i++)
    {
        double t = (double)i / N;
        double trend = 100.0 + 50.0 * t;
        double cycle1 = 20.0 * sin(2.0 * M_PI * i / 100.0);
        double cycle2 = 10.0 * sin(2.0 * M_PI * i / 25.0);
        double noise = 5.0 * randn();
        x[i] = trend + cycle1 + cycle2 + noise;
    }
}

static double correlation(const double *a, const double *b, int n)
{
    double ma = 0, mb = 0;
    for (int i = 0; i < n; i++)
    {
        ma += a[i];
        mb += b[i];
    }
    ma /= n;
    mb /= n;
    double cov = 0, va = 0, vb = 0;
    for (int i = 0; i < n; i++)
    {
        double da = a[i] - ma, db = b[i] - mb;
        cov += da * db;
        va += da * da;
        vb += db * db;
    }
    return cov / sqrt(va * vb + 1e-15);
}

// ============================================================================
// Tests
// ============================================================================

void test_initialization(void)
{
    int N = 1000, L = 250;
    double *x = (double *)malloc(N * sizeof(double));
    generate_signal(x, N);

    SSA_Opt ssa;
    int ret = ssa_opt_init(&ssa, x, N, L);

    ASSERT_EQ(ret, 0, "init returns 0");
    ASSERT_TRUE(ssa.initialized, "initialized flag set");
    ASSERT_EQ(ssa.N, N, "N stored correctly");
    ASSERT_EQ(ssa.L, L, "L stored correctly");
    ASSERT_EQ(ssa.K, N - L + 1, "K computed correctly");
    ASSERT_TRUE(ssa.fft_x != NULL, "fft_x allocated");
    ASSERT_TRUE(ssa.ws_real != NULL, "ws_real allocated");      // R2C field name
    ASSERT_TRUE(ssa.ws_complex != NULL, "ws_complex allocated"); // R2C field name

    ssa_opt_free(&ssa);
    free(x);
}

void test_decomposition(void)
{
    int N = 500, L = 100, k = 10;
    g_seed = 12345;

    double *x = (double *)malloc(N * sizeof(double));
    generate_signal(x, N);

    SSA_Opt ssa;
    ssa_opt_init(&ssa, x, N, L);
    int ret = ssa_opt_decompose(&ssa, k, 150);

    ASSERT_EQ(ret, 0, "decompose returns 0");
    ASSERT_TRUE(ssa.decomposed, "decomposed flag set");
    ASSERT_EQ(ssa.n_components, k, "n_components correct");

    // Singular values should be positive
    for (int i = 0; i < k; i++)
    {
        ASSERT_GT(ssa.sigma[i] + 1e-10, 0, "sigma positive");
    }

    // Should be sorted descending
    for (int i = 0; i < k - 1; i++)
    {
        ASSERT_GT(ssa.sigma[i] + 1e-10, ssa.sigma[i + 1], "sigma sorted");
    }

    ssa_opt_free(&ssa);
    free(x);
}

void test_decomposition_block(void)
{
    int N = 500, L = 100, k = 10;
    g_seed = 12345;

    double *x = (double *)malloc(N * sizeof(double));
    generate_signal(x, N);

    SSA_Opt ssa;
    ssa_opt_init(&ssa, x, N, L);
    int ret = ssa_opt_decompose_block(&ssa, k, 4, 50);

    ASSERT_EQ(ret, 0, "decompose_block returns 0");
    ASSERT_TRUE(ssa.decomposed, "decomposed flag set");
    ASSERT_EQ(ssa.n_components, k, "n_components correct");

    // Singular values should be positive and sorted
    for (int i = 0; i < k; i++)
    {
        ASSERT_GT(ssa.sigma[i] + 1e-10, 0, "sigma positive");
    }
    for (int i = 0; i < k - 1; i++)
    {
        ASSERT_GT(ssa.sigma[i] + 1e-10, ssa.sigma[i + 1], "sigma sorted");
    }

    ssa_opt_free(&ssa);
    free(x);
}

void test_decomposition_randomized(void)
{
    int N = 500, L = 100, k = 10;
    g_seed = 12345;

    double *x = (double *)malloc(N * sizeof(double));
    generate_signal(x, N);

    SSA_Opt ssa;
    ssa_opt_init(&ssa, x, N, L);
    ssa_opt_prepare(&ssa, k, 8);  // Pre-allocate workspace
    int ret = ssa_opt_decompose_randomized(&ssa, k, 8);

    ASSERT_EQ(ret, 0, "decompose_randomized returns 0");
    ASSERT_TRUE(ssa.decomposed, "decomposed flag set");
    ASSERT_TRUE(ssa.prepared, "prepared flag set");
    ASSERT_EQ(ssa.n_components, k, "n_components correct");

    // Singular values should be positive and sorted
    for (int i = 0; i < k; i++)
    {
        ASSERT_GT(ssa.sigma[i] + 1e-10, 0, "sigma positive");
    }
    for (int i = 0; i < k - 1; i++)
    {
        ASSERT_GT(ssa.sigma[i] + 1e-10, ssa.sigma[i + 1], "sigma sorted");
    }

    ssa_opt_free(&ssa);
    free(x);
}

void test_trend_extraction(void)
{
    int N = 1000, L = 200;
    g_seed = 22222;

    double *x = (double *)malloc(N * sizeof(double));
    double *true_trend = (double *)malloc(N * sizeof(double));

    // Linear trend + small noise
    for (int i = 0; i < N; i++)
    {
        double t = (double)i / N;
        true_trend[i] = 2.0 * t + 0.5 * t * t;
        x[i] = true_trend[i] + 0.05 * randn();
    }

    SSA_Opt ssa;
    ssa_opt_init(&ssa, x, N, L);
    ssa_opt_decompose(&ssa, 10, 200);

    double *extracted = (double *)malloc(N * sizeof(double));
    ssa_opt_get_trend(&ssa, extracted);

    double corr = fabs(correlation(extracted, true_trend, N));
    ASSERT_GT(corr, 0.95, "trend |correlation| > 0.95");

    ssa_opt_free(&ssa);
    free(x);
    free(true_trend);
    free(extracted);
}

void test_cycle_detection(void)
{
    int N = 1000, L = 250;

    double *x = (double *)malloc(N * sizeof(double));
    for (int i = 0; i < N; i++)
    {
        x[i] = sin(0.1 * i); // Pure sinusoid
    }

    SSA_Opt ssa;
    ssa_opt_init(&ssa, x, N, L);
    ssa_opt_decompose(&ssa, 10, 100);

    // First two eigenvalues should be paired (sin/cos)
    double ratio = ssa.eigenvalues[0] / (ssa.eigenvalues[1] + 1e-15);
    ASSERT_LT(fabs(ratio - 1.0), 0.15, "first two eigenvalues paired");

    ssa_opt_free(&ssa);
    free(x);
}

void test_reconstruction(void)
{
    int N = 500, L = 100, k = 30;
    g_seed = 33333;

    double *x = (double *)malloc(N * sizeof(double));
    generate_signal(x, N);

    SSA_Opt ssa;
    ssa_opt_init(&ssa, x, N, L);
    ssa_opt_decompose(&ssa, k, 150);

    int *group = (int *)malloc(k * sizeof(int));
    for (int i = 0; i < k; i++)
        group[i] = i;

    double *recon = (double *)malloc(N * sizeof(double));
    ssa_opt_reconstruct(&ssa, group, k, recon);

    double corr = fabs(correlation(recon, x, N));
    ASSERT_GT(corr, 0.95, "reconstruction |correlation| > 0.95");

    ssa_opt_free(&ssa);
    free(x);
    free(group);
    free(recon);
}

void test_variance_explained(void)
{
    int N = 500, L = 100, k = 20;
    g_seed = 44444;

    double *x = (double *)malloc(N * sizeof(double));
    generate_signal(x, N);

    SSA_Opt ssa;
    ssa_opt_init(&ssa, x, N, L);
    ssa_opt_decompose(&ssa, k, 150);

    double total = ssa_opt_variance_explained(&ssa, 0, k - 1);
    ASSERT_GT(total, 0.8, "total variance > 80%");

    double first = ssa_opt_variance_explained(&ssa, 0, 0);
    ASSERT_GT(first, 0.3, "first component > 30%");

    ssa_opt_free(&ssa);
    free(x);
}

void test_no_allocation_in_matvec(void)
{
    // This test verifies that repeated matvec operations don't leak memory
    // by checking that workspace is reused

    int N = 1000, L = 250;
    double *x = (double *)malloc(N * sizeof(double));
    generate_signal(x, N);

    SSA_Opt ssa;
    ssa_opt_init(&ssa, x, N, L);

    // Store workspace pointer (R2C uses ws_real instead of ws_fft1)
    double *ws_before = ssa.ws_real;

    // Do decomposition (many matvec calls)
    ssa_opt_decompose(&ssa, 20, 100);

    // Workspace pointer should be same (reused, not reallocated)
    ASSERT_TRUE(ssa.ws_real == ws_before, "workspace reused (no reallocation)");

    ssa_opt_free(&ssa);
    free(x);
}

void test_wcorr_matrix(void)
{
    int N = 500, L = 100, k = 6;
    g_seed = 55555;

    // Create signal with clear periodic component
    double *x = (double *)malloc(N * sizeof(double));
    for (int i = 0; i < N; i++)
    {
        x[i] = sin(0.1 * i) + 0.1 * randn();
    }

    SSA_Opt ssa;
    ssa_opt_init(&ssa, x, N, L);
    ssa_opt_decompose(&ssa, k, 100);

    double *W = (double *)malloc(k * k * sizeof(double));
    int ret = ssa_opt_wcorr_matrix(&ssa, W);

    ASSERT_EQ(ret, 0, "wcorr_matrix returns 0");

    // Diagonal should be 1.0
    for (int i = 0; i < k; i++)
    {
        ASSERT_LT(fabs(W[i * k + i] - 1.0), 0.01, "diagonal is 1.0");
    }

    // Matrix should be symmetric
    for (int i = 0; i < k; i++)
    {
        for (int j = i + 1; j < k; j++)
        {
            ASSERT_LT(fabs(W[i * k + j] - W[j * k + i]), 1e-10, "matrix symmetric");
        }
    }

    // First two components (sin/cos pair) should be highly correlated
    double wcorr_01 = fabs(W[0 * k + 1]);
    ASSERT_GT(wcorr_01, 0.5, "sin/cos pair has high W-correlation");

    ssa_opt_free(&ssa);
    free(x);
    free(W);
}

void test_component_stats(void)
{
    int N = 500, L = 100, k = 10;
    g_seed = 66666;

    double *x = (double *)malloc(N * sizeof(double));
    generate_signal(x, N);

    SSA_Opt ssa;
    ssa_opt_init(&ssa, x, N, L);
    ssa_opt_decompose(&ssa, k, 100);

    SSA_ComponentStats stats;
    int ret = ssa_opt_component_stats(&ssa, &stats);

    ASSERT_EQ(ret, 0, "component_stats returns 0");
    ASSERT_EQ(stats.n, k, "stats.n matches k");
    ASSERT_TRUE(stats.singular_values != NULL, "singular_values allocated");
    ASSERT_TRUE(stats.cumulative_var != NULL, "cumulative_var allocated");
    ASSERT_TRUE(stats.gaps != NULL, "gaps allocated");

    // Cumulative variance should be monotonically increasing
    for (int i = 1; i < k; i++)
    {
        ASSERT_GT(stats.cumulative_var[i] + 1e-10, stats.cumulative_var[i - 1],
                  "cumulative_var increasing");
    }

    // Last cumulative variance should be close to 1.0
    ASSERT_LT(fabs(stats.cumulative_var[k - 1] - 1.0), 0.01,
              "cumulative_var ends near 1.0");

    ssa_opt_component_stats_free(&stats);
    ssa_opt_free(&ssa);
    free(x);
}

void test_forecasting(void)
{
    int N = 500, L = 100;
    g_seed = 77777;

    // Create predictable signal: trend + periodic
    double *x = (double *)malloc(N * sizeof(double));
    for (int i = 0; i < N; i++)
    {
        x[i] = 0.01 * i + 10.0 * sin(2.0 * M_PI * i / 50.0);
    }

    SSA_Opt ssa;
    ssa_opt_init(&ssa, x, N, L);
    ssa_opt_decompose(&ssa, 10, 100);

    // Forecast using first 3 components (trend + periodic pair)
    int group[] = {0, 1, 2};
    int n_forecast = 50;
    double *forecast = (double *)malloc(n_forecast * sizeof(double));

    int ret = ssa_opt_forecast(&ssa, group, 3, n_forecast, forecast);
    ASSERT_EQ(ret, 0, "forecast returns 0");

    // Generate true future values
    double *true_future = (double *)malloc(n_forecast * sizeof(double));
    for (int i = 0; i < n_forecast; i++)
    {
        true_future[i] = 0.01 * (N + i) + 10.0 * sin(2.0 * M_PI * (N + i) / 50.0);
    }

    // Correlation should be high for short-term forecast
    double corr = fabs(correlation(forecast, true_future, n_forecast));
    ASSERT_GT(corr, 0.8, "forecast correlation > 0.8");

    ssa_opt_free(&ssa);
    free(x);
    free(forecast);
    free(true_future);
}

void test_mssa_basic(void)
{
    int M = 3;   // 3 series
    int N = 300; // Length of each
    int L = 75;
    int k = 6;
    g_seed = 88888;

    // Create M correlated series
    double *X = (double *)malloc(M * N * sizeof(double));
    for (int m = 0; m < M; m++)
    {
        for (int i = 0; i < N; i++)
        {
            // Common trend + series-specific periodic + noise
            double common = 0.01 * i;
            double periodic = 5.0 * sin(2.0 * M_PI * i / 50.0 + m * 0.5);
            double noise = 0.5 * randn();
            X[m * N + i] = common + periodic + noise;
        }
    }

    MSSA_Opt mssa;
    int ret = mssa_opt_init(&mssa, X, M, N, L);
    ASSERT_EQ(ret, 0, "mssa_opt_init returns 0");
    ASSERT_TRUE(mssa.initialized, "mssa initialized flag set");

    ret = mssa_opt_decompose(&mssa, k, 8);
    ASSERT_EQ(ret, 0, "mssa_opt_decompose returns 0");
    ASSERT_TRUE(mssa.decomposed, "mssa decomposed flag set");
    ASSERT_EQ(mssa.n_components, k, "mssa n_components correct");

    // Reconstruct first series
    int group[] = {0, 1, 2};
    double *output = (double *)malloc(N * sizeof(double));
    ret = mssa_opt_reconstruct(&mssa, 0, group, 3, output);
    ASSERT_EQ(ret, 0, "mssa_opt_reconstruct returns 0");

    // Should correlate with original
    double corr = fabs(correlation(output, X, N));
    ASSERT_GT(corr, 0.8, "MSSA reconstruction correlation > 0.8");

    mssa_opt_free(&mssa);
    free(X);
    free(output);
}

// ============================================================================
// Benchmarks
// ============================================================================

void benchmark_decomposition(void)
{
    if (!g_benchmarks)
        return;

    printf("\n  === Decomposition Benchmark (R2C Optimized) ===\n");
    printf("  Backend: Intel MKL with R2C FFT\n\n");

    int sizes[] = {1000, 5000, 10000, 50000, 100000};
    int n_sizes = sizeof(sizes) / sizeof(sizes[0]);

    printf("  %-10s  %-8s  %-12s  %-12s  %-12s  %-10s\n",
           "N", "L", "Init (ms)", "Decomp (ms)", "Recon (ms)", "Total (ms)");
    printf("  %-10s  %-8s  %-12s  %-12s  %-12s  %-10s\n",
           "---", "---", "---------", "-----------", "----------", "---------");

    for (int s = 0; s < n_sizes; s++)
    {
        int N = sizes[s];
        int L = N / 4;
        int k = 20;

        g_seed = 99999;
        double *x = (double *)malloc(N * sizeof(double));
        generate_signal(x, N);

        SSA_Opt ssa;

        // Warmup
        ssa_opt_init(&ssa, x, N, L);
        ssa_opt_decompose(&ssa, k, 50);
        ssa_opt_free(&ssa);

        // Timed run
        double t0 = get_time_ms();
        ssa_opt_init(&ssa, x, N, L);
        double t1 = get_time_ms();

        ssa_opt_decompose(&ssa, k, 100);
        double t2 = get_time_ms();

        int group[] = {0, 1, 2, 3, 4};
        double *output = (double *)malloc(N * sizeof(double));
        ssa_opt_reconstruct(&ssa, group, 5, output);
        double t3 = get_time_ms();

        printf("  %-10d  %-8d  %-12.1f  %-12.1f  %-12.1f  %-10.1f\n",
               N, L, t1 - t0, t2 - t1, t3 - t2, t3 - t0);

        ssa_opt_free(&ssa);
        free(x);
        free(output);
    }
}

void benchmark_matvec_throughput(void)
{
    if (!g_benchmarks)
        return;

    printf("\n  === Hankel Matvec Throughput (R2C) ===\n");

    int N = 10000;
    int L = 2500;
    int K = N - L + 1;
    int iterations = 1000;

    double *x = (double *)malloc(N * sizeof(double));
    double *v = (double *)malloc(K * sizeof(double));
    double *y = (double *)malloc(L * sizeof(double));

    generate_signal(x, N);
    for (int i = 0; i < K; i++)
        v[i] = sin(0.01 * i);

    SSA_Opt ssa;
    ssa_opt_init(&ssa, x, N, L);

    // Warmup
    for (int i = 0; i < 10; i++)
    {
        ssa_opt_hankel_matvec(&ssa, v, y);
    }

    // Timed
    double t0 = get_time_ms();
    for (int i = 0; i < iterations; i++)
    {
        ssa_opt_hankel_matvec(&ssa, v, y);
    }
    double t1 = get_time_ms();

    double ms_per_op = (t1 - t0) / iterations;
    double ops_per_sec = 1000.0 / ms_per_op;

    printf("  N=%d, L=%d, K=%d\n", N, L, K);
    printf("  %d iterations in %.1f ms\n", iterations, t1 - t0);
    printf("  %.3f ms/op (%.0f ops/sec)\n", ms_per_op, ops_per_sec);
    printf("  R2C FFT size: %d (r2c_len=%d)\n", ssa.fft_len, ssa.r2c_len);
    printf("  Memory savings: ~%.0f%% vs C2C\n",
           100.0 * (1.0 - (double)ssa.r2c_len / ssa.fft_len));

    ssa_opt_free(&ssa);
    free(x);
    free(v);
    free(y);
}

void benchmark_memory_efficiency(void)
{
    if (!g_benchmarks)
        return;

    printf("\n  === Memory Efficiency (R2C vs C2C) ===\n");

    int N = 100000;
    int L = N / 4;
    int fft_n = 1;
    while (fft_n < N)
        fft_n <<= 1;
    int r2c_len = fft_n / 2 + 1;

    // C2C memory (old)
    size_t c2c_fft_mem = 2 * fft_n * sizeof(double) * 3; // Complex interleaved

    // R2C memory (new)
    size_t r2c_fft_mem = fft_n * sizeof(double) +           // ws_real
                         2 * r2c_len * sizeof(double) +     // ws_complex
                         2 * r2c_len * sizeof(double);      // fft_x

    printf("  N=%d, L=%d, FFT_N=%d, R2C_LEN=%d\n", N, L, fft_n, r2c_len);
    printf("  C2C buffers: %.2f MB (old)\n", c2c_fft_mem / 1e6);
    printf("  R2C buffers: %.2f MB (new)\n", r2c_fft_mem / 1e6);
    printf("  Memory reduction: %.1f%%\n",
           100.0 * (1.0 - (double)r2c_fft_mem / c2c_fft_mem));
    printf("  (No additional allocations during decomposition)\n");
}

void benchmark_reconstruction_scaling(void)
{
    if (!g_benchmarks)
        return;

    printf("\n  === Reconstruction Scaling (k components) ===\n");

    int N = 10000;
    int L = N / 4;

    g_seed = 77777;
    double *x = (double *)malloc(N * sizeof(double));
    generate_signal(x, N);

    SSA_Opt ssa;
    ssa_opt_init(&ssa, x, N, L);
    ssa_opt_decompose(&ssa, 64, 100);

    printf("  N=%d, L=%d\n\n", N, L);
    printf("  %-8s  %-12s  %-12s\n", "k", "Time (ms)", "ms/component");
    printf("  %-8s  %-12s  %-12s\n", "---", "---------", "------------");

    int k_values[] = {1, 5, 10, 20, 32, 50, 64};
    int n_k = sizeof(k_values) / sizeof(k_values[0]);

    double *output = (double *)malloc(N * sizeof(double));
    int *group = (int *)malloc(64 * sizeof(int));

    for (int ki = 0; ki < n_k; ki++)
    {
        int k = k_values[ki];
        for (int i = 0; i < k; i++)
            group[i] = i;

        // Warmup
        ssa_opt_reconstruct(&ssa, group, k, output);

        // Timed
        int reps = 100;
        double t0 = get_time_ms();
        for (int r = 0; r < reps; r++)
        {
            ssa_opt_reconstruct(&ssa, group, k, output);
        }
        double t1 = get_time_ms();

        double ms_total = (t1 - t0) / reps;
        double ms_per_k = ms_total / k;

        printf("  %-8d  %-12.2f  %-12.3f\n", k, ms_total, ms_per_k);
    }

    ssa_opt_free(&ssa);
    free(x);
    free(output);
    free(group);
}

void benchmark_decomposition_methods(void)
{
    if (!g_benchmarks)
        return;

    printf("\n  === Decomposition Method Comparison ===\n");

    int N = 10000;
    int L = N / 4;
    int k = 20;

    g_seed = 11111;
    double *x = (double *)malloc(N * sizeof(double));
    generate_signal(x, N);

    printf("  N=%d, L=%d, k=%d\n\n", N, L, k);
    printf("  %-20s  %-12s  %-15s\n", "Method", "Time (ms)", "Var Explained");
    printf("  %-20s  %-12s  %-15s\n", "------", "---------", "-------------");

    // Sequential power iteration
    {
        SSA_Opt ssa;
        ssa_opt_init(&ssa, x, N, L);
        double t0 = get_time_ms();
        ssa_opt_decompose(&ssa, k, 100);
        double t1 = get_time_ms();
        double var = ssa_opt_variance_explained(&ssa, 0, k - 1);
        printf("  %-20s  %-12.1f  %-15.4f\n", "Sequential", t1 - t0, var);
        ssa_opt_free(&ssa);
    }

    // Block power iteration
    {
        SSA_Opt ssa;
        ssa_opt_init(&ssa, x, N, L);
        double t0 = get_time_ms();
        ssa_opt_decompose_block(&ssa, k, 8, 50);
        double t1 = get_time_ms();
        double var = ssa_opt_variance_explained(&ssa, 0, k - 1);
        printf("  %-20s  %-12.1f  %-15.4f\n", "Block (b=8)", t1 - t0, var);
        ssa_opt_free(&ssa);
    }

    // Randomized SVD (with prepared workspace)
    {
        SSA_Opt ssa;
        ssa_opt_init(&ssa, x, N, L);
        ssa_opt_prepare(&ssa, k, 8);  // Pre-allocate workspace
        double t0 = get_time_ms();
        ssa_opt_decompose_randomized(&ssa, k, 8);
        double t1 = get_time_ms();
        double var = ssa_opt_variance_explained(&ssa, 0, k - 1);
        printf("  %-20s  %-12.1f  %-15.4f\n", "Randomized (p=8)", t1 - t0, var);
        ssa_opt_free(&ssa);
    }

    free(x);
}

void benchmark_streaming(void)
{
    if (!g_benchmarks)
        return;

    printf("\n  === Streaming Updates (Malloc-Free Hot Path) ===\n");

    int N = 10000;
    int L = N / 4;
    int k = 30;
    int n_iterations = 100;

    g_seed = 22222;
    double *x = (double *)malloc(N * sizeof(double));
    generate_signal(x, N);
    double *output = (double *)malloc(N * sizeof(double));
    int *group = (int *)malloc(k * sizeof(int));
    for (int i = 0; i < k; i++) group[i] = i;

    printf("  N=%d, L=%d, k=%d, iterations=%d\n\n", N, L, k, n_iterations);

    // Setup once with prepare
    SSA_Opt ssa;
    ssa_opt_init(&ssa, x, N, L);
    ssa_opt_prepare(&ssa, k, 8);

    // Warmup
    ssa_opt_decompose_randomized(&ssa, k, 8);
    ssa_opt_reconstruct(&ssa, group, k, output);

    // Timed streaming loop
    double t0 = get_time_ms();
    for (int iter = 0; iter < n_iterations; iter++)
    {
        // Simulate new data arriving
        x[0] += 0.001;

        ssa_opt_update_signal(&ssa, x);
        ssa_opt_decompose_randomized(&ssa, k, 8);
        ssa_opt_reconstruct(&ssa, group, k, output);
    }
    double t1 = get_time_ms();

    double total_ms = t1 - t0;
    double per_iter = total_ms / n_iterations;
    double throughput = 1000.0 / per_iter;

    printf("  Total time: %.1f ms\n", total_ms);
    printf("  Per iteration: %.3f ms\n", per_iter);
    printf("  Throughput: %.0f updates/sec\n", throughput);

    ssa_opt_free(&ssa);
    free(x);
    free(output);
    free(group);
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char **argv)
{
    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "--bench") == 0 || strcmp(argv[i], "-b") == 0)
        {
            g_benchmarks = 1;
        }
    }

    printf("==========================================\n");
    printf("   SSA R2C Optimized Test (Intel MKL)\n");
    printf("==========================================\n");

    // Initialize MKL with optimal settings
    printf("\n--- MKL Configuration ---\n");
    mkl_config_ssa_full(1);  // Verbose output

    // Functional tests
    RUN_TEST(test_initialization);
    RUN_TEST(test_decomposition);
    RUN_TEST(test_decomposition_block);
    RUN_TEST(test_decomposition_randomized);
    RUN_TEST(test_trend_extraction);
    RUN_TEST(test_cycle_detection);
    RUN_TEST(test_reconstruction);
    RUN_TEST(test_variance_explained);
    RUN_TEST(test_no_allocation_in_matvec);
    RUN_TEST(test_wcorr_matrix);
    RUN_TEST(test_component_stats);
    RUN_TEST(test_forecasting);
    RUN_TEST(test_mssa_basic);

    // Benchmarks
    if (g_benchmarks)
    {
        printf("\n=== Benchmarks ===");
        benchmark_decomposition();
        benchmark_decomposition_methods();
        benchmark_streaming();
        benchmark_matvec_throughput();
        benchmark_reconstruction_scaling();
        benchmark_memory_efficiency();
    }

    printf("\n==========================================\n");
    printf("   RESULTS: %d/%d passed", g_passed, g_total);
    if (g_failed > 0)
        printf(" (%d FAILED)", g_failed);
    printf("\n==========================================\n");

#ifdef _WIN32
    printf("\nPress Enter to exit...\n");
    getchar();
#endif

    return g_failed > 0 ? 1 : 0;
}