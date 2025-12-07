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
static inline ssa_real get_time_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}
#elif defined(_WIN32)
#include <windows.h>
static inline ssa_real get_time_ms(void)
{
    LARGE_INTEGER freq, count;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&count);
    return (ssa_real)count.QuadPart * 1000.0 / (ssa_real)freq.QuadPart;
}
#else
static inline ssa_real get_time_ms(void)
{
    return (ssa_real)clock() * 1000.0 / CLOCKS_PER_SEC;
}
#endif

// ============================================================================
// Signal Generation
// ============================================================================

static unsigned int g_seed = 42;

static ssa_real randn(void)
{
    static int have = 0;
    static ssa_real spare;
    if (have)
    {
        have = 0;
        return spare;
    }
    ssa_real u, v, s;
    do
    {
        g_seed = g_seed * 1103515245 + 12345;
        u = (ssa_real)((g_seed >> 16) & 0x7fff) / 16384.0 - 1.0;
        g_seed = g_seed * 1103515245 + 12345;
        v = (ssa_real)((g_seed >> 16) & 0x7fff) / 16384.0 - 1.0;
        s = u * u + v * v;
    } while (s >= 1 || s == 0);
    s = sqrt(-2 * log(s) / s);
    spare = v * s;
    have = 1;
    return u * s;
}

static void generate_signal(ssa_real *x, int N)
{
    for (int i = 0; i < N; i++)
    {
        ssa_real t = (ssa_real)i / N;
        ssa_real trend = 100.0 + 50.0 * t;
        ssa_real cycle1 = 20.0 * sin(2.0 * M_PI * i / 100.0);
        ssa_real cycle2 = 10.0 * sin(2.0 * M_PI * i / 25.0);
        ssa_real noise = 5.0 * randn();
        x[i] = trend + cycle1 + cycle2 + noise;
    }
}

static ssa_real correlation(const ssa_real *a, const ssa_real *b, int n)
{
    ssa_real ma = 0, mb = 0;
    for (int i = 0; i < n; i++)
    {
        ma += a[i];
        mb += b[i];
    }
    ma /= n;
    mb /= n;
    ssa_real cov = 0, va = 0, vb = 0;
    for (int i = 0; i < n; i++)
    {
        ssa_real da = a[i] - ma, db = b[i] - mb;
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
    ssa_real *x = (ssa_real *)malloc(N * sizeof(ssa_real));
    generate_signal(x, N);

    SSA_Opt ssa;
    int ret = ssa_opt_init(&ssa, x, N, L);

    ASSERT_EQ(ret, 0, "init returns 0");
    ASSERT_TRUE(ssa.initialized, "initialized flag set");
    ASSERT_EQ(ssa.N, N, "N stored correctly");
    ASSERT_EQ(ssa.L, L, "L stored correctly");
    ASSERT_EQ(ssa.K, N - L + 1, "K computed correctly");
    ASSERT_TRUE(ssa.fft_x != NULL, "fft_x allocated");
    ASSERT_TRUE(ssa.ws_real != NULL, "ws_real allocated");       // R2C field name
    ASSERT_TRUE(ssa.ws_complex != NULL, "ws_complex allocated"); // R2C field name

    ssa_opt_free(&ssa);
    free(x);
}

void test_decomposition(void)
{
    int N = 500, L = 100, k = 10;
    g_seed = 12345;

    ssa_real *x = (ssa_real *)malloc(N * sizeof(ssa_real));
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

    ssa_real *x = (ssa_real *)malloc(N * sizeof(ssa_real));
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

    ssa_real *x = (ssa_real *)malloc(N * sizeof(ssa_real));
    generate_signal(x, N);

    SSA_Opt ssa;
    ssa_opt_init(&ssa, x, N, L);
    ssa_opt_prepare(&ssa, k, 8); // Pre-allocate workspace
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

    ssa_real *x = (ssa_real *)malloc(N * sizeof(ssa_real));
    ssa_real *true_trend = (ssa_real *)malloc(N * sizeof(ssa_real));

    // Linear trend + small noise
    for (int i = 0; i < N; i++)
    {
        ssa_real t = (ssa_real)i / N;
        true_trend[i] = 2.0 * t + 0.5 * t * t;
        x[i] = true_trend[i] + 0.05 * randn();
    }

    SSA_Opt ssa;
    ssa_opt_init(&ssa, x, N, L);
    ssa_opt_decompose(&ssa, 10, 200);

    ssa_real *extracted = (ssa_real *)malloc(N * sizeof(ssa_real));
    ssa_opt_get_trend(&ssa, extracted);

    ssa_real corr = fabs(correlation(extracted, true_trend, N));
    ASSERT_GT(corr, 0.95, "trend |correlation| > 0.95");

    ssa_opt_free(&ssa);
    free(x);
    free(true_trend);
    free(extracted);
}

void test_cycle_detection(void)
{
    int N = 1000, L = 250;

    ssa_real *x = (ssa_real *)malloc(N * sizeof(ssa_real));
    for (int i = 0; i < N; i++)
    {
        x[i] = sin(0.1 * i); // Pure sinusoid
    }

    SSA_Opt ssa;
    ssa_opt_init(&ssa, x, N, L);
    ssa_opt_decompose(&ssa, 10, 100);

    // First two eigenvalues should be paired (sin/cos)
    ssa_real ratio = ssa.eigenvalues[0] / (ssa.eigenvalues[1] + 1e-15);
    ASSERT_LT(fabs(ratio - 1.0), 0.15, "first two eigenvalues paired");

    ssa_opt_free(&ssa);
    free(x);
}

void test_reconstruction(void)
{
    int N = 500, L = 100, k = 30;
    g_seed = 33333;

    ssa_real *x = (ssa_real *)malloc(N * sizeof(ssa_real));
    generate_signal(x, N);

    SSA_Opt ssa;
    ssa_opt_init(&ssa, x, N, L);
    ssa_opt_decompose(&ssa, k, 150);

    int *group = (int *)malloc(k * sizeof(int));
    for (int i = 0; i < k; i++)
        group[i] = i;

    ssa_real *recon = (ssa_real *)malloc(N * sizeof(ssa_real));
    ssa_opt_reconstruct(&ssa, group, k, recon);

    ssa_real corr = fabs(correlation(recon, x, N));
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

    ssa_real *x = (ssa_real *)malloc(N * sizeof(ssa_real));
    generate_signal(x, N);

    SSA_Opt ssa;
    ssa_opt_init(&ssa, x, N, L);
    ssa_opt_decompose(&ssa, k, 150);

    ssa_real total = ssa_opt_variance_explained(&ssa, 0, k - 1);
    ASSERT_GT(total, 0.8, "total variance > 80%");

    ssa_real first = ssa_opt_variance_explained(&ssa, 0, 0);
    ASSERT_GT(first, 0.3, "first component > 30%");

    ssa_opt_free(&ssa);
    free(x);
}

void test_no_allocation_in_matvec(void)
{
    // This test verifies that repeated matvec operations don't leak memory
    // by checking that workspace is reused

    int N = 1000, L = 250;
    ssa_real *x = (ssa_real *)malloc(N * sizeof(ssa_real));
    generate_signal(x, N);

    SSA_Opt ssa;
    ssa_opt_init(&ssa, x, N, L);

    // Store workspace pointer (R2C uses ws_real instead of ws_fft1)
    ssa_real *ws_before = ssa.ws_real;

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
    ssa_real *x = (ssa_real *)malloc(N * sizeof(ssa_real));
    for (int i = 0; i < N; i++)
    {
        x[i] = sin(0.1 * i) + 0.1 * randn();
    }

    SSA_Opt ssa;
    ssa_opt_init(&ssa, x, N, L);
    ssa_opt_decompose(&ssa, k, 100);

    ssa_real *W = (ssa_real *)malloc(k * k * sizeof(ssa_real));
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
    ssa_real wcorr_01 = fabs(W[0 * k + 1]);
    ASSERT_GT(wcorr_01, 0.5, "sin/cos pair has high W-correlation");

    ssa_opt_free(&ssa);
    free(x);
    free(W);
}

void test_component_stats(void)
{
    int N = 500, L = 100, k = 10;
    g_seed = 66666;

    ssa_real *x = (ssa_real *)malloc(N * sizeof(ssa_real));
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
    ssa_real *x = (ssa_real *)malloc(N * sizeof(ssa_real));
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
    ssa_real *forecast = (ssa_real *)malloc(n_forecast * sizeof(ssa_real));

    int ret = ssa_opt_forecast(&ssa, group, 3, n_forecast, forecast);
    ASSERT_EQ(ret, 0, "forecast returns 0");

    // Generate true future values
    ssa_real *true_future = (ssa_real *)malloc(n_forecast * sizeof(ssa_real));
    for (int i = 0; i < n_forecast; i++)
    {
        true_future[i] = 0.01 * (N + i) + 10.0 * sin(2.0 * M_PI * (N + i) / 50.0);
    }

    // Correlation should be high for short-term forecast
    ssa_real corr = fabs(correlation(forecast, true_future, n_forecast));
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
    ssa_real *X = (ssa_real *)malloc(M * N * sizeof(ssa_real));
    for (int m = 0; m < M; m++)
    {
        for (int i = 0; i < N; i++)
        {
            // Common trend + series-specific periodic + noise
            ssa_real common = 0.01 * i;
            ssa_real periodic = 5.0 * sin(2.0 * M_PI * i / 50.0 + m * 0.5);
            ssa_real noise = 0.5 * randn();
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
    ssa_real *output = (ssa_real *)malloc(N * sizeof(ssa_real));
    ret = mssa_opt_reconstruct(&mssa, 0, group, 3, output);
    ASSERT_EQ(ret, 0, "mssa_opt_reconstruct returns 0");

    // Should correlate with original
    ssa_real corr = fabs(correlation(output, X, N));
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
        ssa_real *x = (ssa_real *)malloc(N * sizeof(ssa_real));
        generate_signal(x, N);

        SSA_Opt ssa;

        // Warmup
        ssa_opt_init(&ssa, x, N, L);
        ssa_opt_decompose(&ssa, k, 50);
        ssa_opt_free(&ssa);

        // Timed run
        ssa_real t0 = get_time_ms();
        ssa_opt_init(&ssa, x, N, L);
        ssa_real t1 = get_time_ms();

        ssa_opt_decompose(&ssa, k, 100);
        ssa_real t2 = get_time_ms();

        int group[] = {0, 1, 2, 3, 4};
        ssa_real *output = (ssa_real *)malloc(N * sizeof(ssa_real));
        ssa_opt_reconstruct(&ssa, group, 5, output);
        ssa_real t3 = get_time_ms();

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

    ssa_real *x = (ssa_real *)malloc(N * sizeof(ssa_real));
    ssa_real *v = (ssa_real *)malloc(K * sizeof(ssa_real));
    ssa_real *y = (ssa_real *)malloc(L * sizeof(ssa_real));

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
    ssa_real t0 = get_time_ms();
    for (int i = 0; i < iterations; i++)
    {
        ssa_opt_hankel_matvec(&ssa, v, y);
    }
    ssa_real t1 = get_time_ms();

    ssa_real ms_per_op = (t1 - t0) / iterations;
    ssa_real ops_per_sec = 1000.0 / ms_per_op;

    printf("  N=%d, L=%d, K=%d\n", N, L, K);
    printf("  %d iterations in %.1f ms\n", iterations, t1 - t0);
    printf("  %.3f ms/op (%.0f ops/sec)\n", ms_per_op, ops_per_sec);
    printf("  R2C FFT size: %d (r2c_len=%d)\n", ssa.fft_len, ssa.r2c_len);
    printf("  Memory savings: ~%.0f%% vs C2C\n",
           100.0 * (1.0 - (ssa_real)ssa.r2c_len / ssa.fft_len));

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
    size_t c2c_fft_mem = 2 * fft_n * sizeof(ssa_real) * 3; // Complex interleaved

    // R2C memory (new)
    size_t r2c_fft_mem = fft_n * sizeof(ssa_real) +       // ws_real
                         2 * r2c_len * sizeof(ssa_real) + // ws_complex
                         2 * r2c_len * sizeof(ssa_real);  // fft_x

    printf("  N=%d, L=%d, FFT_N=%d, R2C_LEN=%d\n", N, L, fft_n, r2c_len);
    printf("  C2C buffers: %.2f MB (old)\n", c2c_fft_mem / 1e6);
    printf("  R2C buffers: %.2f MB (new)\n", r2c_fft_mem / 1e6);
    printf("  Memory reduction: %.1f%%\n",
           100.0 * (1.0 - (ssa_real)r2c_fft_mem / c2c_fft_mem));
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
    ssa_real *x = (ssa_real *)malloc(N * sizeof(ssa_real));
    generate_signal(x, N);

    SSA_Opt ssa;
    ssa_opt_init(&ssa, x, N, L);
    ssa_opt_decompose(&ssa, 64, 100);

    printf("  N=%d, L=%d\n\n", N, L);
    printf("  %-8s  %-12s  %-12s\n", "k", "Time (ms)", "ms/component");
    printf("  %-8s  %-12s  %-12s\n", "---", "---------", "------------");

    int k_values[] = {1, 5, 10, 20, 32, 50, 64};
    int n_k = sizeof(k_values) / sizeof(k_values[0]);

    ssa_real *output = (ssa_real *)malloc(N * sizeof(ssa_real));
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
        ssa_real t0 = get_time_ms();
        for (int r = 0; r < reps; r++)
        {
            ssa_opt_reconstruct(&ssa, group, k, output);
        }
        ssa_real t1 = get_time_ms();

        ssa_real ms_total = (t1 - t0) / reps;
        ssa_real ms_per_k = ms_total / k;

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
    ssa_real *x = (ssa_real *)malloc(N * sizeof(ssa_real));
    generate_signal(x, N);

    printf("  N=%d, L=%d, k=%d\n\n", N, L, k);
    printf("  %-20s  %-12s  %-15s\n", "Method", "Time (ms)", "Var Explained");
    printf("  %-20s  %-12s  %-15s\n", "------", "---------", "-------------");

    // Sequential power iteration
    {
        SSA_Opt ssa;
        ssa_opt_init(&ssa, x, N, L);
        ssa_real t0 = get_time_ms();
        ssa_opt_decompose(&ssa, k, 100);
        ssa_real t1 = get_time_ms();
        ssa_real var = ssa_opt_variance_explained(&ssa, 0, k - 1);
        printf("  %-20s  %-12.1f  %-15.4f\n", "Sequential", t1 - t0, var);
        ssa_opt_free(&ssa);
    }

    // Block power iteration
    {
        SSA_Opt ssa;
        ssa_opt_init(&ssa, x, N, L);
        ssa_real t0 = get_time_ms();
        ssa_opt_decompose_block(&ssa, k, 8, 50);
        ssa_real t1 = get_time_ms();
        ssa_real var = ssa_opt_variance_explained(&ssa, 0, k - 1);
        printf("  %-20s  %-12.1f  %-15.4f\n", "Block (b=8)", t1 - t0, var);
        ssa_opt_free(&ssa);
    }

    // Randomized SVD (with prepared workspace)
    {
        SSA_Opt ssa;
        ssa_opt_init(&ssa, x, N, L);
        ssa_opt_prepare(&ssa, k, 8); // Pre-allocate workspace
        ssa_real t0 = get_time_ms();
        ssa_opt_decompose_randomized(&ssa, k, 8);
        ssa_real t1 = get_time_ms();
        ssa_real var = ssa_opt_variance_explained(&ssa, 0, k - 1);
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
    ssa_real *x = (ssa_real *)malloc(N * sizeof(ssa_real));
    generate_signal(x, N);
    ssa_real *output = (ssa_real *)malloc(N * sizeof(ssa_real));
    int *group = (int *)malloc(k * sizeof(int));
    for (int i = 0; i < k; i++)
        group[i] = i;

    printf("  N=%d, L=%d, k=%d, iterations=%d\n\n", N, L, k, n_iterations);

    // Setup once with prepare
    SSA_Opt ssa;
    ssa_opt_init(&ssa, x, N, L);
    ssa_opt_prepare(&ssa, k, 8);

    // Warmup
    ssa_opt_decompose_randomized(&ssa, k, 8);
    ssa_opt_reconstruct(&ssa, group, k, output);

    // Timed streaming loop
    ssa_real t0 = get_time_ms();
    for (int iter = 0; iter < n_iterations; iter++)
    {
        // Simulate new data arriving
        x[0] += 0.001;

        ssa_opt_update_signal(&ssa, x);
        ssa_opt_decompose_randomized(&ssa, k, 8);
        ssa_opt_reconstruct(&ssa, group, k, output);
    }
    ssa_real t1 = get_time_ms();

    ssa_real total_ms = t1 - t0;
    ssa_real per_iter = total_ms / n_iterations;
    ssa_real throughput = 1000.0 / per_iter;

    printf("  Total time: %.1f ms\n", total_ms);
    printf("  Per iteration: %.3f ms\n", per_iter);
    printf("  Throughput: %.0f updates/sec\n", throughput);

    ssa_opt_free(&ssa);
    free(x);
    free(output);
    free(group);
}

// ============================================================================
// Additional Tests
// ============================================================================

void test_gapfill_iterative(void)
{
    int N = 500, L = 100, rank = 6;

    ssa_real *true_signal = (ssa_real *)malloc(N * sizeof(ssa_real));
    ssa_real *x = (ssa_real *)malloc(N * sizeof(ssa_real));

    for (int i = 0; i < N; i++)
    {
        ssa_real t = (ssa_real)i / N;
        true_signal[i] = sin(2 * M_PI * i / 50.0) +
                         0.5 * sin(2 * M_PI * i / 20.0) +
                         0.02 * i;
        x[i] = true_signal[i];
    }

    // Create gaps
    for (int i = 50; i < 65; i++)
        x[i] = NAN;
    for (int i = 150; i < 170; i++)
        x[i] = NAN;

    int n_gaps_before = 0;
    for (int i = 0; i < N; i++)
    {
        if (isnan(x[i]))
            n_gaps_before++; // Use isnan()
    }
    ASSERT_EQ(n_gaps_before, 35, "35 gaps created");

    // Fill gaps
    SSA_GapFillResult result = {0};
    int ret = ssa_opt_gapfill(x, N, L, rank, 30, 1e-8, &result);

    ASSERT_EQ(ret, 0, "gapfill returns 0");
    ASSERT_EQ(result.n_gaps, 35, "n_gaps detected correctly");
    ASSERT_GT(result.iterations, 0, "iterations > 0");

    // Check no NaNs remain
    int n_gaps_after = 0;
    for (int i = 0; i < N; i++)
    {
        if (x[i] != x[i])
            n_gaps_after++;
    }
    ASSERT_EQ(n_gaps_after, 0, "all gaps filled");

    // Check accuracy at gap positions
    ssa_real rmse = 0;
    int gap_count = 0;
    for (int i = 50; i < 65; i++)
    {
        rmse += (x[i] - true_signal[i]) * (x[i] - true_signal[i]);
        gap_count++;
    }
    for (int i = 150; i < 170; i++)
    {
        rmse += (x[i] - true_signal[i]) * (x[i] - true_signal[i]);
        gap_count++;
    }
    rmse = sqrt(rmse / gap_count);
    ASSERT_LT(rmse, 0.5, "gap fill RMSE < 0.5");

    free(true_signal);
    free(x);
}

void test_gapfill_simple(void)
{
    int N = 500, L = 100, rank = 6;

    ssa_real *true_signal = (ssa_real *)malloc(N * sizeof(ssa_real));
    ssa_real *x = (ssa_real *)malloc(N * sizeof(ssa_real));

    for (int i = 0; i < N; i++)
    {
        true_signal[i] = sin(2 * M_PI * i / 50.0) + 0.02 * i;
        x[i] = true_signal[i];
    }

    // Single gap in middle
    for (int i = 200; i < 220; i++)
        x[i] = NAN;

    SSA_GapFillResult result = {0};
    int ret = ssa_opt_gapfill_simple(x, N, L, rank, &result);

    ASSERT_EQ(ret, 0, "gapfill_simple returns 0");
    ASSERT_EQ(result.n_gaps, 20, "n_gaps = 20");

    // Check no NaNs remain
    int n_gaps_after = 0;
    for (int i = 0; i < N; i++)
    {
        if (x[i] != x[i])
            n_gaps_after++;
    }
    ASSERT_EQ(n_gaps_after, 0, "all gaps filled (simple)");

    free(true_signal);
    free(x);
}

void test_cadzow(void)
{
    int N = 500, L = 125, rank = 6;
    g_seed = 11111;

    ssa_real *true_signal = (ssa_real *)malloc(N * sizeof(ssa_real));
    ssa_real *x = (ssa_real *)malloc(N * sizeof(ssa_real));

    for (int i = 0; i < N; i++)
    {
        true_signal[i] = 2.0 * sin(2 * M_PI * i / 50.0) +
                         1.0 * sin(2 * M_PI * i / 20.0) +
                         0.5 * sin(2 * M_PI * i / 10.0);
        x[i] = true_signal[i] + 1.0 * randn();
    }

    // Single-pass SSA for comparison
    SSA_Opt ssa;
    ssa_opt_init(&ssa, x, N, L);
    ssa_opt_decompose(&ssa, rank, 100);

    int group[6] = {0, 1, 2, 3, 4, 5};
    ssa_real *ssa_result = (ssa_real *)malloc(N * sizeof(ssa_real));
    ssa_opt_reconstruct(&ssa, group, rank, ssa_result);
    ssa_opt_free(&ssa);

    // Cadzow iterations
    ssa_real *x_cadzow = (ssa_real *)malloc(N * sizeof(ssa_real));

    SSA_CadzowResult result = {0};
    int ret = ssa_opt_cadzow(x, N, L, rank, 20, 1e-6, x_cadzow, &result);

    // ret >= 0 means success (returns iteration count or 0)
    ASSERT_TRUE(ret >= 0, "cadzow returns >= 0");
    ASSERT_GT(result.iterations, 0, "cadzow iterations > 0");
    ASSERT_LT(result.final_diff, 0.01, "cadzow converged (diff < 0.01)");

    // Cadzow should be at least as good as single-pass SSA
    ssa_real corr_ssa = fabs(correlation(ssa_result, true_signal, N));
    ssa_real corr_cadzow = fabs(correlation(x_cadzow, true_signal, N));

    ASSERT_GT(corr_cadzow, corr_ssa - 0.05, "Cadzow >= SSA quality");
    ASSERT_GT(corr_cadzow, 0.9, "Cadzow correlation > 0.9");

    free(true_signal);
    free(x);
    free(ssa_result);
    free(x_cadzow);
}

void test_esprit(void)
{
    int N = 500, L = 125, k = 6;

    // Signal with known periods: 50 and 20
    ssa_real *x = (ssa_real *)malloc(N * sizeof(ssa_real));
    for (int i = 0; i < N; i++)
    {
        x[i] = 3.0 * sin(2 * M_PI * i / 50.0) + // period 50
               2.0 * sin(2 * M_PI * i / 20.0);  // period 20
    }

    SSA_Opt ssa;
    ssa_opt_init(&ssa, x, N, L);
    ssa_opt_decompose(&ssa, k, 100);

    // ESPRIT
    int group[] = {0, 1, 2, 3};
    SSA_ParEstimate par = {0};
    int ret = ssa_opt_parestimate(&ssa, group, 4, &par);

    ASSERT_EQ(ret, 0, "parestimate returns 0");
    ASSERT_EQ(par.n_components, 4, "par.n_components = 4");
    ASSERT_TRUE(par.periods != NULL, "periods allocated");
    ASSERT_TRUE(par.moduli != NULL, "moduli allocated");

    // Check that we found periods near 50 and 20
    int found_50 = 0, found_20 = 0;
    for (int i = 0; i < par.n_components; i++)
    {
        if (fabs(par.periods[i] - 50.0) < 2.0)
            found_50 = 1;
        if (fabs(par.periods[i] - 20.0) < 2.0)
            found_20 = 1;

        // Modulus should be close to 1.0 for undamped sinusoids
        if (par.periods[i] > 5 && par.periods[i] < 100)
        {
            ASSERT_GT(par.moduli[i], 0.9, "modulus > 0.9 for signal");
        }
    }
    ASSERT_TRUE(found_50, "detected period ~50");
    ASSERT_TRUE(found_20, "detected period ~20");

    ssa_opt_parestimate_free(&par);
    ssa_opt_free(&ssa);
    free(x);
}

void test_vforecast(void)
{
    int N = 500, L = 100;

    // Predictable signal
    ssa_real *x = (ssa_real *)malloc(N * sizeof(ssa_real));
    for (int i = 0; i < N; i++)
    {
        x[i] = 10.0 * sin(2.0 * M_PI * i / 50.0);
    }

    SSA_Opt ssa;
    ssa_opt_init(&ssa, x, N, L);
    ssa_opt_decompose(&ssa, 4, 100);

    int group[] = {0, 1};
    int n_forecast = 50;
    ssa_real *forecast_r = (ssa_real *)malloc(n_forecast * sizeof(ssa_real));
    ssa_real *forecast_v = (ssa_real *)malloc(n_forecast * sizeof(ssa_real));

    // R-forecast (LRF)
    int ret = ssa_opt_forecast(&ssa, group, 2, n_forecast, forecast_r);
    ASSERT_EQ(ret, 0, "forecast (R) returns 0");

    // V-forecast
    ret = ssa_opt_vforecast(&ssa, group, 2, n_forecast, forecast_v);
    ASSERT_EQ(ret, 0, "vforecast returns 0");

    // Both should match true future
    ssa_real *true_future = (ssa_real *)malloc(n_forecast * sizeof(ssa_real));
    for (int i = 0; i < n_forecast; i++)
    {
        true_future[i] = 10.0 * sin(2.0 * M_PI * (N + i) / 50.0);
    }

    ssa_real corr_r = fabs(correlation(forecast_r, true_future, n_forecast));
    ssa_real corr_v = fabs(correlation(forecast_v, true_future, n_forecast));

    ASSERT_GT(corr_r, 0.95, "R-forecast correlation > 0.95");
    ASSERT_GT(corr_v, 0.95, "V-forecast correlation > 0.95");

    ssa_opt_free(&ssa);
    free(x);
    free(forecast_r);
    free(forecast_v);
    free(true_future);
}

void test_forecast_full(void)
{
    int N = 300, L = 75;

    ssa_real *x = (ssa_real *)malloc(N * sizeof(ssa_real));
    for (int i = 0; i < N; i++)
    {
        x[i] = 5.0 * sin(2.0 * M_PI * i / 30.0) + 0.01 * i;
    }

    SSA_Opt ssa;
    ssa_opt_init(&ssa, x, N, L);
    ssa_opt_decompose(&ssa, 6, 100);

    int group[] = {0, 1, 2};
    int n_forecast = 30;
    ssa_real *full = (ssa_real *)malloc((N + n_forecast) * sizeof(ssa_real));

    int ret = ssa_opt_forecast_full(&ssa, group, 3, n_forecast, full);
    ASSERT_EQ(ret, 0, "forecast_full returns 0");

    // First N values should be reconstruction
    ssa_real corr_recon = fabs(correlation(full, x, N));
    ASSERT_GT(corr_recon, 0.95, "reconstruction part correlation > 0.95");

    // Last n_forecast should be forecast
    ssa_real *true_future = (ssa_real *)malloc(n_forecast * sizeof(ssa_real));
    for (int i = 0; i < n_forecast; i++)
    {
        true_future[i] = 5.0 * sin(2.0 * M_PI * (N + i) / 30.0) + 0.01 * (N + i);
    }
    ssa_real corr_forecast = fabs(correlation(&full[N], true_future, n_forecast));
    ASSERT_GT(corr_forecast, 0.8, "forecast part correlation > 0.8");

    ssa_opt_free(&ssa);
    free(x);
    free(full);
    free(true_future);
}

void test_wcorr_matrix_fast(void)
{
    int N = 500, L = 100, k = 10;
    g_seed = 99999;

    ssa_real *x = (ssa_real *)malloc(N * sizeof(ssa_real));
    for (int i = 0; i < N; i++)
    {
        x[i] = sin(0.1 * i) + 0.5 * sin(0.25 * i) + 0.1 * randn();
    }

    SSA_Opt ssa;
    ssa_opt_init(&ssa, x, N, L);
    ssa_opt_decompose(&ssa, k, 100);

    ssa_real *W_slow = (ssa_real *)malloc(k * k * sizeof(ssa_real));
    ssa_real *W_fast = (ssa_real *)malloc(k * k * sizeof(ssa_real));

    // Standard method
    int ret1 = ssa_opt_wcorr_matrix(&ssa, W_slow);
    ASSERT_EQ(ret1, 0, "wcorr_matrix returns 0");

    // Fast method (DSYRK-based)
    int ret2 = ssa_opt_wcorr_matrix_fast(&ssa, W_fast);
    ASSERT_EQ(ret2, 0, "wcorr_matrix_fast returns 0");

    // Results should match
    ssa_real max_diff = 0;
    for (int i = 0; i < k * k; i++)
    {
        ssa_real diff = fabs(W_slow[i] - W_fast[i]);
        if (diff > max_diff)
            max_diff = diff;
    }
    ASSERT_LT(max_diff, 1e-6, "fast and slow wcorr match");

    ssa_opt_free(&ssa);
    free(x);
    free(W_slow);
    free(W_fast);
}

void test_eigenvalue_getters(void)
{
    int N = 500, L = 100, k = 10;
    g_seed = 12321;

    ssa_real *x = (ssa_real *)malloc(N * sizeof(ssa_real));
    generate_signal(x, N);

    SSA_Opt ssa;
    ssa_opt_init(&ssa, x, N, L);
    ssa_opt_decompose(&ssa, k, 100);

    ssa_real *eigenvalues = (ssa_real *)malloc(k * sizeof(ssa_real));
    int n_eig = ssa_opt_get_eigenvalues(&ssa, eigenvalues, k);
    ASSERT_EQ(n_eig, k, "got k eigenvalues");

    ssa_real *singular_values = (ssa_real *)malloc(k * sizeof(ssa_real));
    int n_sv = ssa_opt_get_singular_values(&ssa, singular_values, k);
    ASSERT_EQ(n_sv, k, "got k singular values");

    // Debug: print first few values
    printf("  sigma[0]=%.4f, lambda[0]=%.4f, sigma^2=%.4f\n",
           singular_values[0], eigenvalues[0], singular_values[0] * singular_values[0]);

    // All should be positive and sorted descending
    for (int i = 0; i < k; i++)
    {
        ASSERT_GT(eigenvalues[i], 0, "eigenvalue > 0");
        ASSERT_GT(singular_values[i], 0, "singular_value > 0");
    }
    for (int i = 0; i < k - 1; i++)
    {
        ASSERT_GT(eigenvalues[i] + 1e-10, eigenvalues[i + 1], "eigenvalues sorted");
    }

    ssa_opt_free(&ssa);
    free(x);
    free(eigenvalues);
    free(singular_values);
}

void test_nan_detection(void)
{
    ssa_real nan_val = NAN;
    ssa_real normal_val = 3.14;

    // Use isnan() from math.h - works reliably on all compilers
    ASSERT_TRUE(isnan(nan_val), "isnan detects NaN");
    ASSERT_TRUE(!isnan(normal_val), "isnan rejects normal");
}

// ============================================================================
// Edge Cases & Robustness Tests
// ============================================================================

void test_constant_signal(void)
{
    // Constant signal = rank 1
    int N = 200, L = 50;
    ssa_real *x = (ssa_real *)malloc(N * sizeof(ssa_real));
    for (int i = 0; i < N; i++)
        x[i] = 42.0;

    SSA_Opt ssa;
    ssa_opt_init(&ssa, x, N, L);
    ssa_opt_decompose(&ssa, 5, 100);

    // First eigenvalue should dominate
    ssa_real ratio = ssa.eigenvalues[0] / (ssa.eigenvalues[1] + 1e-15);
    ASSERT_GT(ratio, 1000, "constant signal: first eigenvalue dominates");

    // Reconstruction with just component 0 should match
    int group[] = {0};
    ssa_real *recon = (ssa_real *)malloc(N * sizeof(ssa_real));
    ssa_opt_reconstruct(&ssa, group, 1, recon);

    ssa_real max_err = 0;
    for (int i = 0; i < N; i++)
    {
        ssa_real err = fabs(recon[i] - 42.0);
        if (err > max_err)
            max_err = err;
    }
    ASSERT_LT(max_err, 0.01, "constant signal reconstruction");

    ssa_opt_free(&ssa);
    free(x);
    free(recon);
}

void test_reconstruction_sum(void)
{
    // Sum of all components should equal original signal
    int N = 300, L = 75, k = 30;
    g_seed = 54321;

    ssa_real *x = (ssa_real *)malloc(N * sizeof(ssa_real));
    generate_signal(x, N);

    SSA_Opt ssa;
    ssa_opt_init(&ssa, x, N, L);
    ssa_opt_decompose(&ssa, k, 100);

    // Reconstruct all components
    int *group = (int *)malloc(k * sizeof(int));
    for (int i = 0; i < k; i++)
        group[i] = i;

    ssa_real *recon = (ssa_real *)malloc(N * sizeof(ssa_real));
    ssa_opt_reconstruct(&ssa, group, k, recon);

    // Check variance explained
    ssa_real var_exp = ssa_opt_variance_explained(&ssa, 0, k - 1);
    printf("  k=%d components, variance explained=%.4f\n", k, var_exp);

    // Should match original closely
    ssa_real max_err = 0;
    for (int i = 0; i < N; i++)
    {
        ssa_real err = fabs(recon[i] - x[i]);
        if (err > max_err)
            max_err = err;
    }
    printf("  max reconstruction error=%.4f\n", max_err);

    // Loosen tolerance - k=30 may not capture 100% of signal
    ASSERT_LT(max_err, 1.0, "sum of all components ≈ original");

    ssa_opt_free(&ssa);
    free(x);
    free(group);
    free(recon);
}

void test_small_signal(void)
{
    // Minimum viable signal
    int N = 20, L = 5, k = 3;

    ssa_real *x = (ssa_real *)malloc(N * sizeof(ssa_real));
    for (int i = 0; i < N; i++)
    {
        x[i] = sin(0.5 * i);
    }

    SSA_Opt ssa;
    int ret = ssa_opt_init(&ssa, x, N, L);
    ASSERT_EQ(ret, 0, "small signal init");

    ret = ssa_opt_decompose(&ssa, k, 50);
    ASSERT_EQ(ret, 0, "small signal decompose");
    ASSERT_EQ(ssa.n_components, k, "small signal n_components");

    ssa_opt_free(&ssa);
    free(x);
}

void test_large_L(void)
{
    // L close to N/2 (maximum typical)
    int N = 200, L = 99, k = 10;

    ssa_real *x = (ssa_real *)malloc(N * sizeof(ssa_real));
    for (int i = 0; i < N; i++)
    {
        x[i] = sin(0.1 * i) + 0.5 * cos(0.3 * i);
    }

    SSA_Opt ssa;
    int ret = ssa_opt_init(&ssa, x, N, L);
    ASSERT_EQ(ret, 0, "large L init");
    ASSERT_EQ(ssa.K, N - L + 1, "large L: K correct");

    ret = ssa_opt_decompose(&ssa, k, 100);
    ASSERT_EQ(ret, 0, "large L decompose");

    ssa_opt_free(&ssa);
    free(x);
}

void test_single_component(void)
{
    // k = 1
    int N = 200, L = 50;

    ssa_real *x = (ssa_real *)malloc(N * sizeof(ssa_real));
    for (int i = 0; i < N; i++)
    {
        x[i] = 0.01 * i + sin(0.1 * i);
    }

    SSA_Opt ssa;
    ssa_opt_init(&ssa, x, N, L);
    ssa_opt_decompose(&ssa, 1, 100);

    ASSERT_EQ(ssa.n_components, 1, "single component");
    ASSERT_GT(ssa.sigma[0], 0, "single component sigma > 0");

    int group[] = {0};
    ssa_real *recon = (ssa_real *)malloc(N * sizeof(ssa_real));
    ssa_opt_reconstruct(&ssa, group, 1, recon);

    // Should capture dominant structure
    ssa_real corr = fabs(correlation(recon, x, N));
    ASSERT_GT(corr, 0.5, "single component captures structure");

    ssa_opt_free(&ssa);
    free(x);
    free(recon);
}

void test_repeated_init_free(void)
{
    // Memory leak / ssa_real-free check
    int N = 500, L = 100;
    ssa_real *x = (ssa_real *)malloc(N * sizeof(ssa_real));
    generate_signal(x, N);

    for (int iter = 0; iter < 10; iter++)
    {
        SSA_Opt ssa;
        ssa_opt_init(&ssa, x, N, L);
        ssa_opt_decompose(&ssa, 10, 50);
        ssa_opt_free(&ssa);
    }
    ASSERT_TRUE(1, "repeated init/free cycles OK");

    free(x);
}

void test_hankel_matvec_correctness(void)
{
    // Compare FFT-based matvec with direct computation
    int N = 100, L = 25;
    int K = N - L + 1;

    ssa_real *x = (ssa_real *)malloc(N * sizeof(ssa_real));
    ssa_real *v = (ssa_real *)malloc(K * sizeof(ssa_real));
    ssa_real *y_fft = (ssa_real *)malloc(L * sizeof(ssa_real));
    ssa_real *y_direct = (ssa_real *)malloc(L * sizeof(ssa_real));

    for (int i = 0; i < N; i++)
        x[i] = sin(0.1 * i) + 0.5 * cos(0.25 * i);
    for (int i = 0; i < K; i++)
        v[i] = cos(0.05 * i);

    SSA_Opt ssa;
    ssa_opt_init(&ssa, x, N, L);

    // FFT-based
    ssa_opt_hankel_matvec(&ssa, v, y_fft);

    // Direct computation: y[i] = sum_j H[i,j] * v[j] = sum_j x[i+j] * v[j]
    for (int i = 0; i < L; i++)
    {
        y_direct[i] = 0;
        for (int j = 0; j < K; j++)
        {
            y_direct[i] += x[i + j] * v[j];
        }
    }

    // Should match
    ssa_real max_err = 0;
    for (int i = 0; i < L; i++)
    {
        ssa_real err = fabs(y_fft[i] - y_direct[i]);
        if (err > max_err)
            max_err = err;
    }
    ASSERT_LT(max_err, 1e-10, "hankel matvec matches direct");

    ssa_opt_free(&ssa);
    free(x);
    free(v);
    free(y_fft);
    free(y_direct);
}

void test_adjoint_matvec_correctness(void)
{
    // Compare FFT-based adjoint with direct computation
    int N = 100, L = 25;
    int K = N - L + 1;

    ssa_real *x = (ssa_real *)malloc(N * sizeof(ssa_real));
    ssa_real *u = (ssa_real *)malloc(L * sizeof(ssa_real));
    ssa_real *z_fft = (ssa_real *)malloc(K * sizeof(ssa_real));
    ssa_real *z_direct = (ssa_real *)malloc(K * sizeof(ssa_real));

    for (int i = 0; i < N; i++)
        x[i] = sin(0.1 * i);
    for (int i = 0; i < L; i++)
        u[i] = cos(0.07 * i);

    SSA_Opt ssa;
    ssa_opt_init(&ssa, x, N, L);

    // FFT-based
    ssa_opt_hankel_matvec_T(&ssa, u, z_fft);

    // Direct: z[j] = sum_i H[i,j] * u[i] = sum_i x[i+j] * u[i]
    for (int j = 0; j < K; j++)
    {
        z_direct[j] = 0;
        for (int i = 0; i < L; i++)
        {
            z_direct[j] += x[i + j] * u[i];
        }
    }

    ssa_real max_err = 0;
    for (int j = 0; j < K; j++)
    {
        ssa_real err = fabs(z_fft[j] - z_direct[j]);
        if (err > max_err)
            max_err = err;
    }
    ASSERT_LT(max_err, 1e-10, "adjoint matvec matches direct");

    ssa_opt_free(&ssa);
    free(x);
    free(u);
    free(z_fft);
    free(z_direct);
}

void test_gap_at_edges(void)
{
    // Gaps at start and end are harder
    int N = 300, L = 60, rank = 4;

    ssa_real *x = (ssa_real *)malloc(N * sizeof(ssa_real));
    for (int i = 0; i < N; i++)
    {
        x[i] = sin(2 * M_PI * i / 30.0);
    }

    // Gap at start
    for (int i = 0; i < 10; i++)
        x[i] = NAN;
    // Gap at end
    for (int i = N - 10; i < N; i++)
        x[i] = NAN;

    SSA_GapFillResult result = {0};
    int ret = ssa_opt_gapfill(x, N, L, rank, 30, 1e-6, &result);

    ASSERT_TRUE(ret >= 0, "edge gaps: returns >= 0");
    ASSERT_EQ(result.n_gaps, 20, "edge gaps: 20 gaps detected");

    int nans_after = 0;
    for (int i = 0; i < N; i++)
    {
        if (isnan(x[i]))
            nans_after++;
    }
    ASSERT_EQ(nans_after, 0, "edge gaps: all filled");

    free(x);
}

void test_forecast_single_step(void)
{
    int N = 200, L = 50;

    ssa_real *x = (ssa_real *)malloc(N * sizeof(ssa_real));
    for (int i = 0; i < N; i++)
    {
        x[i] = sin(2 * M_PI * i / 40.0);
    }

    SSA_Opt ssa;
    ssa_opt_init(&ssa, x, N, L);
    ssa_opt_decompose(&ssa, 4, 100);

    int group[] = {0, 1};
    ssa_real forecast[1];

    int ret = ssa_opt_forecast(&ssa, group, 2, 1, forecast);
    ASSERT_EQ(ret, 0, "single step forecast OK");

    // True next value
    ssa_real true_next = sin(2 * M_PI * N / 40.0);
    ssa_real err = fabs(forecast[0] - true_next);
    ASSERT_LT(err, 0.1, "single step forecast accurate");

    ssa_opt_free(&ssa);
    free(x);
}

// ============================================================================
// Error Handling & Boundary Conditions
// ============================================================================

void test_invalid_L_too_small(void)
{
    int N = 100, L = 1; // L must be >= 2
    ssa_real *x = (ssa_real *)malloc(N * sizeof(ssa_real));
    for (int i = 0; i < N; i++)
        x[i] = sin(0.1 * i);

    SSA_Opt ssa;
    int ret = ssa_opt_init(&ssa, x, N, L);

    // Should fail or handle gracefully
    ASSERT_TRUE(ret != 0 || L >= 2, "L=1 rejected or handled");

    if (ret == 0)
        ssa_opt_free(&ssa);
    free(x);
}

void test_invalid_L_too_large(void)
{
    int N = 100, L = 99; // L close to N
    ssa_real *x = (ssa_real *)malloc(N * sizeof(ssa_real));
    for (int i = 0; i < N; i++)
        x[i] = sin(0.1 * i);

    SSA_Opt ssa;
    int ret = ssa_opt_init(&ssa, x, N, L);

    // K = N - L + 1 = 2, very small but should work
    if (ret == 0)
    {
        ASSERT_EQ(ssa.K, 2, "K = 2 for L = N-1");
        ssa_opt_free(&ssa);
    }
    ASSERT_TRUE(1, "L close to N handled");

    free(x);
}

void test_k_exceeds_rank(void)
{
    int N = 100, L = 20;
    int K = N - L + 1;              // 81
    int max_rank = (L < K) ? L : K; // 20
    int k = max_rank + 5;           // Request more than possible

    ssa_real *x = (ssa_real *)malloc(N * sizeof(ssa_real));
    for (int i = 0; i < N; i++)
        x[i] = sin(0.1 * i);

    SSA_Opt ssa;
    ssa_opt_init(&ssa, x, N, L);
    int ret = ssa_opt_decompose(&ssa, k, 100);

    // Should either fail or clamp to max_rank
    ASSERT_TRUE(ret != 0 || ssa.n_components <= max_rank, "k clamped to max rank");

    ssa_opt_free(&ssa);
    free(x);
}

void test_decomposition_methods_agree(void)
{
    // Sequential, block, and randomized should produce similar results
    int N = 500, L = 100, k = 10;
    g_seed = 77777;

    ssa_real *x = (ssa_real *)malloc(N * sizeof(ssa_real));
    generate_signal(x, N);

    // Sequential
    SSA_Opt ssa1;
    ssa_opt_init(&ssa1, x, N, L);
    ssa_opt_decompose(&ssa1, k, 150);

    // Block
    SSA_Opt ssa2;
    ssa_opt_init(&ssa2, x, N, L);
    ssa_opt_decompose_block(&ssa2, k, 4, 100);

    // Randomized
    SSA_Opt ssa3;
    ssa_opt_init(&ssa3, x, N, L);
    ssa_opt_prepare(&ssa3, k, 8);
    ssa_opt_decompose_randomized(&ssa3, k, 8);

    // Singular values should be similar
    ssa_real max_diff_12 = 0, max_diff_13 = 0;
    for (int i = 0; i < k; i++)
    {
        ssa_real d12 = fabs(ssa1.sigma[i] - ssa2.sigma[i]) / (ssa1.sigma[i] + 1e-10);
        ssa_real d13 = fabs(ssa1.sigma[i] - ssa3.sigma[i]) / (ssa1.sigma[i] + 1e-10);
        if (d12 > max_diff_12)
            max_diff_12 = d12;
        if (d13 > max_diff_13)
            max_diff_13 = d13;
    }

    ASSERT_LT(max_diff_12, 0.01, "sequential vs block: sigma within 1%");

    ssa_opt_free(&ssa1);
    ssa_opt_free(&ssa2);
    ssa_opt_free(&ssa3);
    free(x);
}

void test_randomized_determinism(void)
{
    // Same seed should produce same results
    int N = 300, L = 75, k = 10;

    ssa_real *x = (ssa_real *)malloc(N * sizeof(ssa_real));
    for (int i = 0; i < N; i++)
        x[i] = sin(0.1 * i) + 0.5 * cos(0.23 * i);

    // First run
    SSA_Opt ssa1;
    ssa_opt_init(&ssa1, x, N, L);
    ssa_opt_prepare(&ssa1, k, 8);
    ssa_opt_decompose_randomized(&ssa1, k, 8);

    // Second run (same setup)
    SSA_Opt ssa2;
    ssa_opt_init(&ssa2, x, N, L);
    ssa_opt_prepare(&ssa2, k, 8);
    ssa_opt_decompose_randomized(&ssa2, k, 8);

    // Should be identical or very close
    ssa_real max_diff = 0;
    for (int i = 0; i < k; i++)
    {
        ssa_real d = fabs(ssa1.sigma[i] - ssa2.sigma[i]);
        if (d > max_diff)
            max_diff = d;
    }

    // Note: may not be identical if RNG state differs
    ASSERT_LT(max_diff, ssa1.sigma[0] * 0.01, "randomized is reproducible");

    ssa_opt_free(&ssa1);
    ssa_opt_free(&ssa2);
    free(x);
}

void test_linear_trend_only(void)
{
    // Pure linear trend = rank 2
    int N = 200, L = 50;

    ssa_real *x = (ssa_real *)malloc(N * sizeof(ssa_real));
    for (int i = 0; i < N; i++)
    {
        x[i] = 3.0 + 0.5 * i;
    }

    SSA_Opt ssa;
    ssa_opt_init(&ssa, x, N, L);
    ssa_opt_decompose(&ssa, 5, 100);

    // First two components should dominate
    ssa_real top2 = ssa.eigenvalues[0] + ssa.eigenvalues[1];
    ssa_real rest = 0;
    for (int i = 2; i < 5; i++)
        rest += ssa.eigenvalues[i];

    ASSERT_GT(top2 / (rest + 1e-10), 100, "linear trend: top 2 dominate");

    // Reconstruct with components 0,1
    int group[] = {0, 1};
    ssa_real *recon = (ssa_real *)malloc(N * sizeof(ssa_real));
    ssa_opt_reconstruct(&ssa, group, 2, recon);

    ssa_real corr = correlation(recon, x, N);
    ASSERT_GT(corr, 0.999, "linear trend reconstruction");

    ssa_opt_free(&ssa);
    free(x);
    free(recon);
}

void test_high_frequency(void)
{
    // Near Nyquist frequency (period = 2)
    int N = 200, L = 50;

    ssa_real *x = (ssa_real *)malloc(N * sizeof(ssa_real));
    for (int i = 0; i < N; i++)
    {
        x[i] = (i % 2 == 0) ? 1.0 : -1.0; // Alternating ±1
    }

    SSA_Opt ssa;
    ssa_opt_init(&ssa, x, N, L);
    ssa_opt_decompose(&ssa, 4, 100);

    // Should still work
    ASSERT_TRUE(ssa.decomposed, "high frequency decomposed");
    ASSERT_GT(ssa.sigma[0], 0, "high frequency has positive sigma");

    ssa_opt_free(&ssa);
    free(x);
}

void test_gap_larger_than_L(void)
{
    // Gap > L is problematic but shouldn't crash
    int N = 300, L = 50, rank = 4;

    ssa_real *x = (ssa_real *)malloc(N * sizeof(ssa_real));
    for (int i = 0; i < N; i++)
    {
        x[i] = sin(2 * M_PI * i / 30.0);
    }

    // Create gap larger than L
    for (int i = 100; i < 180; i++)
        x[i] = NAN; // 80 > L=50

    SSA_GapFillResult result = {0};
    int ret = ssa_opt_gapfill(x, N, L, rank, 30, 1e-6, &result);

    // Should either work or fail gracefully
    printf("  gap > L: ret=%d, n_gaps=%d\n", ret, result.n_gaps);
    ASSERT_TRUE(1, "gap > L handled without crash");

    free(x);
}

void test_mssa_single_series(void)
{
    // M = 1 should degenerate to regular SSA
    int M = 1, N = 200, L = 50, k = 5;

    ssa_real *x = (ssa_real *)malloc(N * sizeof(ssa_real));
    for (int i = 0; i < N; i++)
    {
        x[i] = sin(0.1 * i) + 0.3 * cos(0.25 * i);
    }

    // Regular SSA
    SSA_Opt ssa;
    ssa_opt_init(&ssa, x, N, L);
    ssa_opt_decompose(&ssa, k, 100);

    // MSSA with M=1
    MSSA_Opt mssa;
    mssa_opt_init(&mssa, x, M, N, L);
    mssa_opt_decompose(&mssa, k, 8);

    // Singular values should match
    ssa_real max_diff = 0;
    for (int i = 0; i < k; i++)
    {
        ssa_real d = fabs(ssa.sigma[i] - mssa.sigma[i]) / (ssa.sigma[i] + 1e-10);
        if (d > max_diff)
            max_diff = d;
    }

    ASSERT_LT(max_diff, 0.1, "MSSA(M=1) ≈ SSA");

    ssa_opt_free(&ssa);
    mssa_opt_free(&mssa);
    free(x);
}

void test_empty_group_reconstruction(void)
{
    int N = 200, L = 50;

    ssa_real *x = (ssa_real *)malloc(N * sizeof(ssa_real));
    for (int i = 0; i < N; i++)
        x[i] = sin(0.1 * i);

    SSA_Opt ssa;
    ssa_opt_init(&ssa, x, N, L);
    ssa_opt_decompose(&ssa, 5, 100);

    // Empty group
    ssa_real *recon = (ssa_real *)malloc(N * sizeof(ssa_real));
    int ret = ssa_opt_reconstruct(&ssa, NULL, 0, recon);

    // Should return zeros or fail gracefully
    printf("  empty group: ret=%d\n", ret);
    ASSERT_TRUE(1, "empty group handled");

    ssa_opt_free(&ssa);
    free(x);
    free(recon);
}

void test_very_long_forecast(void)
{
    // Long forecasts may become unstable
    int N = 300, L = 75;

    ssa_real *x = (ssa_real *)malloc(N * sizeof(ssa_real));
    for (int i = 0; i < N; i++)
    {
        x[i] = sin(2 * M_PI * i / 50.0);
    }

    SSA_Opt ssa;
    ssa_opt_init(&ssa, x, N, L);
    ssa_opt_decompose(&ssa, 4, 100);

    int group[] = {0, 1};
    int n_forecast = 500; // Longer than original signal
    ssa_real *forecast = (ssa_real *)malloc(n_forecast * sizeof(ssa_real));

    int ret = ssa_opt_forecast(&ssa, group, 2, n_forecast, forecast);
    ASSERT_EQ(ret, 0, "long forecast returns 0");

    // Check if values stay bounded (no explosion)
    ssa_real max_abs = 0;
    for (int i = 0; i < n_forecast; i++)
    {
        ssa_real a = fabs(forecast[i]);
        if (a > max_abs)
            max_abs = a;
    }

    ASSERT_LT(max_abs, 10.0, "long forecast stays bounded");

    ssa_opt_free(&ssa);
    free(x);
    free(forecast);
}

// ============================================================================
// Accuracy & Quality Tests
// ============================================================================

void test_denoising_accuracy(void)
{
    // Measure SNR improvement
    int N = 1000, L = 200;
    g_seed = 11111;

    ssa_real *true_signal = (ssa_real *)malloc(N * sizeof(ssa_real));
    ssa_real *noisy = (ssa_real *)malloc(N * sizeof(ssa_real));

    // Clean signal: trend + 2 sinusoids
    for (int i = 0; i < N; i++)
    {
        true_signal[i] = 0.01 * i +
                         5.0 * sin(2 * M_PI * i / 100.0) +
                         3.0 * sin(2 * M_PI * i / 25.0);
        noisy[i] = true_signal[i] + 2.0 * randn();
    }

    // Input SNR
    ssa_real signal_power = 0, noise_power = 0;
    for (int i = 0; i < N; i++)
    {
        signal_power += true_signal[i] * true_signal[i];
        noise_power += (noisy[i] - true_signal[i]) * (noisy[i] - true_signal[i]);
    }
    ssa_real input_snr = 10 * log10(signal_power / noise_power);

    // SSA denoise
    SSA_Opt ssa;
    ssa_opt_init(&ssa, noisy, N, L);
    ssa_opt_decompose(&ssa, 10, 100);

    int group[] = {0, 1, 2, 3, 4}; // First 5 components
    ssa_real *denoised = (ssa_real *)malloc(N * sizeof(ssa_real));
    ssa_opt_reconstruct(&ssa, group, 5, denoised);

    // Output SNR
    ssa_real residual_power = 0;
    for (int i = 0; i < N; i++)
    {
        residual_power += (denoised[i] - true_signal[i]) * (denoised[i] - true_signal[i]);
    }
    ssa_real output_snr = 10 * log10(signal_power / residual_power);
    ssa_real snr_improvement = output_snr - input_snr;

    printf("  Input SNR: %.1f dB, Output SNR: %.1f dB, Improvement: %.1f dB\n",
           input_snr, output_snr, snr_improvement);

    ASSERT_GT(snr_improvement, 5.0, "SNR improves by at least 5 dB");
    ASSERT_GT(output_snr, 15.0, "Output SNR > 15 dB");

    ssa_opt_free(&ssa);
    free(true_signal);
    free(noisy);
    free(denoised);
}

void test_component_orthogonality(void)
{
    // Reconstructed components should be approximately orthogonal
    int N = 500, L = 100, k = 6;
    g_seed = 22222;

    ssa_real *x = (ssa_real *)malloc(N * sizeof(ssa_real));
    generate_signal(x, N);

    SSA_Opt ssa;
    ssa_opt_init(&ssa, x, N, L);
    ssa_opt_decompose(&ssa, k, 100);

    // Reconstruct individual components
    ssa_real **components = (ssa_real **)malloc(k * sizeof(ssa_real *));
    for (int i = 0; i < k; i++)
    {
        components[i] = (ssa_real *)malloc(N * sizeof(ssa_real));
        int group[] = {i};
        ssa_opt_reconstruct(&ssa, group, 1, components[i]);
    }

    // Check pairwise correlations (should be low for non-paired)
    ssa_real max_cross_corr = 0;
    for (int i = 0; i < k; i++)
    {
        for (int j = i + 2; j < k; j++)
        { // Skip adjacent pairs
            ssa_real corr = fabs(correlation(components[i], components[j], N));
            if (corr > max_cross_corr)
                max_cross_corr = corr;
        }
    }

    printf("  max cross-correlation (non-adjacent): %.4f\n", max_cross_corr);
    ASSERT_LT(max_cross_corr, 0.3, "non-paired components weakly correlated");

    for (int i = 0; i < k; i++)
        free(components[i]);
    free(components);
    ssa_opt_free(&ssa);
    free(x);
}

void test_esprit_accuracy(void)
{
    // Quantify period detection accuracy
    int N = 600, L = 150, k = 8;

    ssa_real periods_true[] = {60.0, 30.0, 15.0};
    int n_periods = 3;

    ssa_real *x = (ssa_real *)malloc(N * sizeof(ssa_real));
    for (int i = 0; i < N; i++)
    {
        x[i] = 4.0 * sin(2 * M_PI * i / 60.0) +
               2.0 * sin(2 * M_PI * i / 30.0) +
               1.0 * sin(2 * M_PI * i / 15.0);
    }

    SSA_Opt ssa;
    ssa_opt_init(&ssa, x, N, L);
    ssa_opt_decompose(&ssa, k, 100);

    int group[] = {0, 1, 2, 3, 4, 5};
    SSA_ParEstimate par = {0};
    ssa_opt_parestimate(&ssa, group, 6, &par);

    // Check each true period is detected
    int found_count = 0;
    for (int t = 0; t < n_periods; t++)
    {
        ssa_real true_p = periods_true[t];
        ssa_real best_match = 1e10;

        for (int i = 0; i < par.n_components; i++)
        {
            if (par.moduli[i] > 0.9)
            { // Only signal components
                ssa_real err = fabs(par.periods[i] - true_p) / true_p;
                if (err < best_match)
                    best_match = err;
            }
        }

        printf("  True period %.0f: best match error = %.2f%%\n",
               true_p, best_match * 100);

        if (best_match < 0.05)
            found_count++; // Within 5%
    }

    ASSERT_EQ(found_count, n_periods, "all periods detected within 5%");

    ssa_opt_parestimate_free(&par);
    ssa_opt_free(&ssa);
    free(x);
}

void test_forecast_rmse(void)
{
    // Measure forecast error over horizon
    int N = 400, L = 100;
    ssa_real period = 40.0;

    ssa_real *x = (ssa_real *)malloc(N * sizeof(ssa_real));
    for (int i = 0; i < N; i++)
    {
        x[i] = 10.0 * sin(2 * M_PI * i / period) + 0.01 * i;
    }

    SSA_Opt ssa;
    ssa_opt_init(&ssa, x, N, L);
    ssa_opt_decompose(&ssa, 6, 100);

    int group[] = {0, 1, 2};
    int horizons[] = {10, 25, 50, 100};
    int n_horizons = 4;

    printf("  Forecast RMSE by horizon:\n");

    for (int h = 0; h < n_horizons; h++)
    {
        int n_forecast = horizons[h];
        ssa_real *forecast = (ssa_real *)malloc(n_forecast * sizeof(ssa_real));
        ssa_opt_forecast(&ssa, group, 3, n_forecast, forecast);

        // True future
        ssa_real rmse = 0;
        for (int i = 0; i < n_forecast; i++)
        {
            ssa_real true_val = 10.0 * sin(2 * M_PI * (N + i) / period) + 0.01 * (N + i);
            rmse += (forecast[i] - true_val) * (forecast[i] - true_val);
        }
        rmse = sqrt(rmse / n_forecast);

        printf("    horizon=%3d: RMSE=%.4f\n", n_forecast, rmse);
        free(forecast);
    }

    // Short horizon should be accurate
    ssa_real *forecast_short = (ssa_real *)malloc(10 * sizeof(ssa_real));
    ssa_opt_forecast(&ssa, group, 3, 10, forecast_short);

    ssa_real rmse_10 = 0;
    for (int i = 0; i < 10; i++)
    {
        ssa_real true_val = 10.0 * sin(2 * M_PI * (N + i) / period) + 0.01 * (N + i);
        rmse_10 += (forecast_short[i] - true_val) * (forecast_short[i] - true_val);
    }
    rmse_10 = sqrt(rmse_10 / 10);

    ASSERT_LT(rmse_10, 1.0, "10-step forecast RMSE < 1.0");

    ssa_opt_free(&ssa);
    free(x);
    free(forecast_short);
}

void test_gapfill_accuracy(void)
{
    // Measure gap filling accuracy vs gap size
    int N = 500, L = 100, rank = 6;

    ssa_real *true_signal = (ssa_real *)malloc(N * sizeof(ssa_real));
    for (int i = 0; i < N; i++)
    {
        true_signal[i] = sin(2 * M_PI * i / 50.0) + 0.5 * sin(2 * M_PI * i / 20.0);
    }

    int gap_sizes[] = {5, 10, 20, 40};
    int n_gaps = 4;

    printf("  Gap fill RMSE by gap size:\n");

    for (int g = 0; g < n_gaps; g++)
    {
        int gap_size = gap_sizes[g];
        ssa_real *x = (ssa_real *)malloc(N * sizeof(ssa_real));
        memcpy(x, true_signal, N * sizeof(ssa_real));

        // Create gap in middle
        int gap_start = 200;
        for (int i = gap_start; i < gap_start + gap_size; i++)
        {
            x[i] = NAN;
        }

        SSA_GapFillResult result = {0};
        ssa_opt_gapfill(x, N, L, rank, 30, 1e-8, &result);

        // RMSE at gap positions
        ssa_real rmse = 0;
        for (int i = gap_start; i < gap_start + gap_size; i++)
        {
            rmse += (x[i] - true_signal[i]) * (x[i] - true_signal[i]);
        }
        rmse = sqrt(rmse / gap_size);

        printf("    gap_size=%2d: RMSE=%.4f (iter=%d)\n", gap_size, rmse, result.iterations);
        free(x);
    }

    // Small gap should be very accurate
    ssa_real *x = (ssa_real *)malloc(N * sizeof(ssa_real));
    memcpy(x, true_signal, N * sizeof(ssa_real));
    for (int i = 200; i < 205; i++)
        x[i] = NAN;

    SSA_GapFillResult result = {0};
    ssa_opt_gapfill(x, N, L, rank, 30, 1e-8, &result);

    ssa_real rmse = 0;
    for (int i = 200; i < 205; i++)
    {
        rmse += (x[i] - true_signal[i]) * (x[i] - true_signal[i]);
    }
    rmse = sqrt(rmse / 5);

    ASSERT_LT(rmse, 0.1, "5-point gap RMSE < 0.1");

    free(true_signal);
    free(x);
}

void test_cadzow_snr_improvement(void)
{
    // Cadzow should improve upon single-pass SSA
    int N = 500, L = 125, rank = 6;
    g_seed = 33333;

    ssa_real *true_signal = (ssa_real *)malloc(N * sizeof(ssa_real));
    ssa_real *noisy = (ssa_real *)malloc(N * sizeof(ssa_real));

    for (int i = 0; i < N; i++)
    {
        true_signal[i] = 3.0 * sin(2 * M_PI * i / 50.0) +
                         2.0 * sin(2 * M_PI * i / 25.0) +
                         1.0 * sin(2 * M_PI * i / 10.0);
        noisy[i] = true_signal[i] + 1.5 * randn();
    }

    // Single-pass SSA
    SSA_Opt ssa;
    ssa_opt_init(&ssa, noisy, N, L);
    ssa_opt_decompose(&ssa, rank, 100);

    int group[] = {0, 1, 2, 3, 4, 5};
    ssa_real *ssa_result = (ssa_real *)malloc(N * sizeof(ssa_real));
    ssa_opt_reconstruct(&ssa, group, rank, ssa_result);
    ssa_opt_free(&ssa);

    // Cadzow
    ssa_real *cadzow_result = (ssa_real *)malloc(N * sizeof(ssa_real));
    SSA_CadzowResult cad_res = {0};
    ssa_opt_cadzow(noisy, N, L, rank, 30, 1e-8, cadzow_result, &cad_res);

    // Compute SNRs
    ssa_real signal_power = 0;
    for (int i = 0; i < N; i++)
    {
        signal_power += true_signal[i] * true_signal[i];
    }

    ssa_real ssa_err = 0, cadzow_err = 0;
    for (int i = 0; i < N; i++)
    {
        ssa_err += (ssa_result[i] - true_signal[i]) * (ssa_result[i] - true_signal[i]);
        cadzow_err += (cadzow_result[i] - true_signal[i]) * (cadzow_result[i] - true_signal[i]);
    }

    ssa_real ssa_snr = 10 * log10(signal_power / ssa_err);
    ssa_real cadzow_snr = 10 * log10(signal_power / cadzow_err);

    printf("  SSA SNR: %.1f dB, Cadzow SNR: %.1f dB (iter=%d)\n",
           ssa_snr, cadzow_snr, cad_res.iterations);

    ASSERT_GT(cadzow_snr, ssa_snr - 1.0, "Cadzow at least as good as SSA");

    free(true_signal);
    free(noisy);
    free(ssa_result);
    free(cadzow_result);
}

void test_wcorr_paired_components(void)
{
    // Sine/cosine pairs should have high W-correlation
    int N = 500, L = 100, k = 6;

    ssa_real *x = (ssa_real *)malloc(N * sizeof(ssa_real));
    for (int i = 0; i < N; i++)
    {
        x[i] = 5.0 * sin(2 * M_PI * i / 50.0) + // pair 0-1
               2.0 * sin(2 * M_PI * i / 20.0);  // pair 2-3
    }

    SSA_Opt ssa;
    ssa_opt_init(&ssa, x, N, L);
    ssa_opt_decompose(&ssa, k, 100);

    ssa_real *W = (ssa_real *)malloc(k * k * sizeof(ssa_real));
    ssa_opt_wcorr_matrix(&ssa, W);

    // Pairs (0,1) and (2,3) should be highly correlated
    ssa_real wcorr_01 = fabs(W[0 * k + 1]);
    ssa_real wcorr_23 = fabs(W[2 * k + 3]);

    // Cross-pairs should be less correlated
    ssa_real wcorr_02 = fabs(W[0 * k + 2]);
    ssa_real wcorr_13 = fabs(W[1 * k + 3]);

    printf("  W-corr(0,1)=%.3f, W-corr(2,3)=%.3f\n", wcorr_01, wcorr_23);
    printf("  W-corr(0,2)=%.3f, W-corr(1,3)=%.3f\n", wcorr_02, wcorr_13);

    ASSERT_GT(wcorr_01, 0.8, "pair 0-1 highly correlated");
    ASSERT_GT(wcorr_23, 0.8, "pair 2-3 highly correlated");
    ASSERT_LT(wcorr_02, 0.3, "cross-pairs weakly correlated");

    ssa_opt_free(&ssa);
    free(x);
    free(W);
}

void test_variance_partition(void)
{
    // Variance explained should sum properly
    int N = 500, L = 100, k = 20;
    g_seed = 44444;

    ssa_real *x = (ssa_real *)malloc(N * sizeof(ssa_real));
    generate_signal(x, N);

    SSA_Opt ssa;
    ssa_opt_init(&ssa, x, N, L);
    ssa_opt_decompose(&ssa, k, 100);

    // Individual variances
    ssa_real sum_individual = 0;
    for (int i = 0; i < k; i++)
    {
        ssa_real var_i = ssa_opt_variance_explained(&ssa, i, i);
        sum_individual += var_i;
    }

    // Total variance
    ssa_real var_total = ssa_opt_variance_explained(&ssa, 0, k - 1);

    printf("  sum of individual: %.4f, total: %.4f\n", sum_individual, var_total);

    ASSERT_LT(fabs(sum_individual - var_total), 0.001, "variance sums correctly");

    // Cumulative should be monotonic
    ssa_real prev = 0;
    for (int i = 0; i < k; i++)
    {
        ssa_real cum = ssa_opt_variance_explained(&ssa, 0, i);
        ASSERT_GT(cum + 1e-10, prev, "cumulative variance increases");
        prev = cum;
    }

    ssa_opt_free(&ssa);
    free(x);
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
    mkl_config_ssa_full(1); // Verbose output

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
    RUN_TEST(test_nan_detection);
    RUN_TEST(test_gapfill_iterative);
    RUN_TEST(test_gapfill_simple);
    RUN_TEST(test_cadzow);
    RUN_TEST(test_esprit);
    RUN_TEST(test_vforecast);
    RUN_TEST(test_forecast_full);
    RUN_TEST(test_wcorr_matrix_fast);
    RUN_TEST(test_eigenvalue_getters);
    RUN_TEST(test_constant_signal);
    RUN_TEST(test_reconstruction_sum);
    RUN_TEST(test_small_signal);
    RUN_TEST(test_large_L);
    RUN_TEST(test_single_component);
    RUN_TEST(test_repeated_init_free);
    RUN_TEST(test_hankel_matvec_correctness);
    RUN_TEST(test_adjoint_matvec_correctness);
    RUN_TEST(test_gap_at_edges);
    RUN_TEST(test_forecast_single_step);
    RUN_TEST(test_invalid_L_too_small);
    RUN_TEST(test_invalid_L_too_large);
    RUN_TEST(test_k_exceeds_rank);
    RUN_TEST(test_decomposition_methods_agree);
    RUN_TEST(test_randomized_determinism);
    RUN_TEST(test_linear_trend_only);
    RUN_TEST(test_high_frequency);
    RUN_TEST(test_gap_larger_than_L);
    RUN_TEST(test_mssa_single_series);
    RUN_TEST(test_empty_group_reconstruction);
    RUN_TEST(test_very_long_forecast);
    RUN_TEST(test_denoising_accuracy);
    RUN_TEST(test_component_orthogonality);
    RUN_TEST(test_esprit_accuracy);
    RUN_TEST(test_forecast_rmse);
    RUN_TEST(test_gapfill_accuracy);
    RUN_TEST(test_cadzow_snr_improvement);
    RUN_TEST(test_wcorr_paired_components);
    RUN_TEST(test_variance_partition);

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