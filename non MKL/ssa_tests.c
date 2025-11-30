/*
 * SSA Test Suite
 * 
 * Tests:
 *   1. Basic initialization and decomposition
 *   2. Trend extraction on synthetic data
 *   3. Cycle detection (paired eigenvalues)
 *   4. Noise separation
 *   5. Reconstruction accuracy
 *   6. Variance explained
 *   7. Forecasting
 *   8. Performance benchmarks
 * 
 * Compile:
 *   gcc -O3 -o ssa_tests ssa_tests.c -lm
 * 
 * Run:
 *   ./ssa_tests           # All tests
 *   ./ssa_tests --bench   # Include benchmarks
 */

#define SSA_IMPLEMENTATION
#include "ssa.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

// -----------------------------
// Test Framework
// -----------------------------

#define TEST_PASS "\033[32mPASS\033[0m"
#define TEST_FAIL "\033[31mFAIL\033[0m"

static int g_total = 0;
static int g_passed = 0;
static int g_failed = 0;
static int g_verbose = 0;
static int g_benchmarks = 0;

#define ASSERT_TRUE(cond, msg) do { \
    g_total++; \
    if (cond) { \
        g_passed++; \
        if (g_verbose) printf("  [%s] %s\n", TEST_PASS, msg); \
    } else { \
        g_failed++; \
        printf("  [%s] %s (line %d)\n", TEST_FAIL, msg, __LINE__); \
    } \
} while(0)

#define ASSERT_EQ(a, b, msg) ASSERT_TRUE((a) == (b), msg)

#define ASSERT_NEAR(a, b, tol, msg) do { \
    double _a = (a), _b = (b), _tol = (tol); \
    double _diff = fabs(_a - _b); \
    g_total++; \
    if (_diff <= _tol) { \
        g_passed++; \
        if (g_verbose) printf("  [%s] %s (diff=%.2e)\n", TEST_PASS, msg, _diff); \
    } else { \
        g_failed++; \
        printf("  [%s] %s (expected %.6g, got %.6g, diff=%.2e) line %d\n", \
               TEST_FAIL, msg, _b, _a, _diff, __LINE__); \
    } \
} while(0)

#define ASSERT_GT(a, b, msg) ASSERT_TRUE((a) > (b), msg)
#define ASSERT_LT(a, b, msg) ASSERT_TRUE((a) < (b), msg)

#define RUN_TEST(fn) do { \
    printf("\n[TEST] %s\n", #fn); \
    fn(); \
} while(0)

// Timing
#ifdef __linux__
#include <time.h>
static inline uint64_t get_nanos(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}
#else
static inline uint64_t get_nanos(void) {
    return (uint64_t)clock() * 1000000000ULL / CLOCKS_PER_SEC;
}
#endif

#define BENCHMARK(name, iterations, setup, code) do { \
    if (!g_benchmarks) break; \
    setup; \
    for (int _w = 0; _w < 3; _w++) { code; } \
    uint64_t _start = get_nanos(); \
    for (int _i = 0; _i < (iterations); _i++) { code; } \
    uint64_t _end = get_nanos(); \
    double _ms = (double)(_end - _start) / 1e6; \
    double _per_iter = _ms / (iterations); \
    printf("  [BENCH] %s: %.2f ms total, %.3f ms/iter\n", name, _ms, _per_iter); \
} while(0)

// -----------------------------
// Signal Generation
// -----------------------------

static double randn(void) {
    static int have_spare = 0;
    static double spare;
    
    if (have_spare) {
        have_spare = 0;
        return spare;
    }
    
    double u, v, s;
    do {
        u = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
        v = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
        s = u * u + v * v;
    } while (s >= 1.0 || s == 0.0);
    
    s = sqrt(-2.0 * log(s) / s);
    spare = v * s;
    have_spare = 1;
    return u * s;
}

// Generate: trend + cycle + noise
static void generate_test_signal(double* x, int N, 
                                  double trend_slope,
                                  double cycle_amplitude, double cycle_period,
                                  double noise_std) {
    for (int i = 0; i < N; i++) {
        double t = (double)i / N;
        double trend = trend_slope * t;
        double cycle = cycle_amplitude * sin(2.0 * M_PI * i / cycle_period);
        double noise = noise_std * randn();
        x[i] = trend + cycle + noise;
    }
}

// Generate pure trend
static void generate_trend(double* x, int N, double slope, double curve) {
    for (int i = 0; i < N; i++) {
        double t = (double)i / N;
        x[i] = slope * t + curve * t * t;
    }
}

// Generate pure cycle
static void generate_cycle(double* x, int N, double amplitude, double period) {
    for (int i = 0; i < N; i++) {
        x[i] = amplitude * sin(2.0 * M_PI * i / period);
    }
}

// Compute correlation
static double correlation(const double* a, const double* b, int n) {
    double mean_a = 0, mean_b = 0;
    for (int i = 0; i < n; i++) {
        mean_a += a[i];
        mean_b += b[i];
    }
    mean_a /= n;
    mean_b /= n;
    
    double cov = 0, var_a = 0, var_b = 0;
    for (int i = 0; i < n; i++) {
        double da = a[i] - mean_a;
        double db = b[i] - mean_b;
        cov += da * db;
        var_a += da * da;
        var_b += db * db;
    }
    
    return cov / sqrt(var_a * var_b + 1e-15);
}

// Compute RMSE
static double rmse(const double* a, const double* b, int n) {
    double sum = 0;
    for (int i = 0; i < n; i++) {
        double d = a[i] - b[i];
        sum += d * d;
    }
    return sqrt(sum / n);
}

// -----------------------------
// Tests
// -----------------------------

void test_initialization(void) {
    int N = 1000;
    int L = 250;
    
    double* x = malloc(N * sizeof(double));
    for (int i = 0; i < N; i++) x[i] = sin(0.1 * i);
    
    SSA ssa;
    int ret = ssa_init(&ssa, x, N, L);
    
    ASSERT_EQ(ret, 0, "ssa_init returns 0");
    ASSERT_TRUE(ssa.initialized, "initialized flag set");
    ASSERT_EQ(ssa.N, N, "N stored correctly");
    ASSERT_EQ(ssa.L, L, "L stored correctly");
    ASSERT_EQ(ssa.K, N - L + 1, "K computed correctly");
    ASSERT_TRUE(ssa.fft_x != NULL, "fft_x allocated");
    ASSERT_TRUE(ssa.fft_x_rev != NULL, "fft_x_rev allocated");
    
    ssa_free(&ssa);
    free(x);
}

void test_decomposition(void) {
    int N = 500;
    int L = 100;
    int k = 10;
    
    double* x = malloc(N * sizeof(double));
    srand(12345);
    generate_test_signal(x, N, 1.0, 0.5, 50, 0.1);
    
    SSA ssa;
    ssa_init(&ssa, x, N, L);
    int ret = ssa_decompose(&ssa, k, 100);
    
    ASSERT_EQ(ret, 0, "ssa_decompose returns 0");
    ASSERT_TRUE(ssa.decomposed, "decomposed flag set");
    ASSERT_EQ(ssa.n_components, k, "n_components = k");
    ASSERT_TRUE(ssa.sigma != NULL, "sigma allocated");
    ASSERT_TRUE(ssa.U != NULL, "U allocated");
    ASSERT_TRUE(ssa.V != NULL, "V allocated");
    
    // Singular values should be decreasing
    for (int i = 0; i < k - 1; i++) {
        ASSERT_GT(ssa.sigma[i], ssa.sigma[i+1] - 1e-10, "sigma decreasing");
    }
    
    // All sigma should be positive
    for (int i = 0; i < k; i++) {
        ASSERT_GT(ssa.sigma[i], 0, "sigma positive");
    }
    
    ssa_free(&ssa);
    free(x);
}

void test_trend_extraction(void) {
    int N = 1000;
    int L = 200;
    
    // Generate trend + small noise
    double* x = malloc(N * sizeof(double));
    double* true_trend = malloc(N * sizeof(double));
    
    srand(22222);
    generate_trend(true_trend, N, 2.0, 0.5);
    for (int i = 0; i < N; i++) {
        x[i] = true_trend[i] + 0.05 * randn();
    }
    
    SSA ssa;
    ssa_init(&ssa, x, N, L);
    ssa_decompose(&ssa, 10, 200);  // More components and iterations
    
    double* extracted_trend = malloc(N * sizeof(double));
    ssa_get_trend(&ssa, extracted_trend);
    
    // Correlation with true trend (use absolute value for sign ambiguity)
    double corr = fabs(correlation(extracted_trend, true_trend, N));
    ASSERT_GT(corr, 0.95, "trend |correlation| > 0.95");
    
    ssa_free(&ssa);
    free(x);
    free(true_trend);
    free(extracted_trend);
}

void test_cycle_detection(void) {
    int N = 1000;
    int L = 250;
    
    // Pure sinusoid should give paired eigenvalues
    double* x = malloc(N * sizeof(double));
    generate_cycle(x, N, 1.0, 50);
    
    SSA ssa;
    ssa_init(&ssa, x, N, L);
    ssa_decompose(&ssa, 10, 100);
    
    // First two eigenvalues should be nearly equal (the sin/cos pair)
    double ratio = ssa.eigenvalues[0] / (ssa.eigenvalues[1] + 1e-15);
    ASSERT_LT(fabs(ratio - 1.0), 0.1, "first two eigenvalues paired");
    
    // Find pairs
    int pairs[20];
    int n_pairs = ssa_find_pairs(&ssa, pairs, 10, 0.1);
    ASSERT_GT(n_pairs, 0, "at least one pair found");
    ASSERT_EQ(pairs[0], 0, "first pair starts at 0");
    ASSERT_EQ(pairs[1], 1, "first pair is (0, 1)");
    
    ssa_free(&ssa);
    free(x);
}

void test_reconstruction_accuracy(void) {
    int N = 500;
    int L = 100;
    int k = 30;  // More components for better reconstruction
    
    double* x = malloc(N * sizeof(double));
    srand(33333);
    for (int i = 0; i < N; i++) {
        x[i] = sin(0.1 * i) + 0.3 * cos(0.25 * i) + 0.1 * randn();
    }
    
    SSA ssa;
    ssa_init(&ssa, x, N, L);
    ssa_decompose(&ssa, k, 150);
    
    // Reconstruct from all components
    int* all_group = malloc(k * sizeof(int));
    for (int i = 0; i < k; i++) all_group[i] = i;
    
    double* reconstructed = malloc(N * sizeof(double));
    ssa_reconstruct(&ssa, all_group, k, reconstructed);
    
    // Should nearly match original (since we used all components)
    double err = rmse(reconstructed, x, N);
    ASSERT_LT(err, 0.15, "full reconstruction RMSE < 0.15");
    
    double corr = correlation(reconstructed, x, N);
    ASSERT_GT(corr, 0.95, "full reconstruction correlation > 0.95");
    
    ssa_free(&ssa);
    free(x);
    free(all_group);
    free(reconstructed);
}

void test_variance_explained(void) {
    int N = 500;
    int L = 100;
    int k = 20;
    
    double* x = malloc(N * sizeof(double));
    srand(44444);
    generate_test_signal(x, N, 1.0, 0.5, 50, 0.2);
    
    SSA ssa;
    ssa_init(&ssa, x, N, L);
    ssa_decompose(&ssa, k, 150);
    
    // Total variance explained by all components should be significant
    double total = ssa_variance_explained(&ssa, 0, k - 1);
    ASSERT_GT(total, 0.8, "total variance > 80%");
    
    // First component should explain significant variance (trend)
    double first = ssa_variance_explained(&ssa, 0, 0);
    ASSERT_GT(first, 0.3, "first component > 30% variance");
    
    // Variance should be monotonically decreasing (after sorting)
    int violations = 0;
    for (int i = 0; i < k - 1; i++) {
        double v_i = ssa_variance_explained(&ssa, i, i);
        double v_j = ssa_variance_explained(&ssa, i + 1, i + 1);
        if (v_i + 1e-10 < v_j) violations++;
    }
    ASSERT_LT(violations, 3, "variance mostly decreasing");
    
    ssa_free(&ssa);
    free(x);
}

void test_noise_separation(void) {
    int N = 1000;
    int L = 200;
    
    // Strong signal + noise
    double* x = malloc(N * sizeof(double));
    double* signal = malloc(N * sizeof(double));
    double* noise_true = malloc(N * sizeof(double));
    
    srand(55555);
    for (int i = 0; i < N; i++) {
        signal[i] = sin(0.05 * i) + 0.5 * cos(0.12 * i);
        noise_true[i] = 0.2 * randn();
        x[i] = signal[i] + noise_true[i];
    }
    
    SSA ssa;
    ssa_init(&ssa, x, N, L);
    ssa_decompose(&ssa, 20, 150);
    
    // Extract signal (first few components)
    int signal_group[] = {0, 1, 2, 3};
    double* extracted_signal = malloc(N * sizeof(double));
    ssa_reconstruct(&ssa, signal_group, 4, extracted_signal);
    
    // Signal correlation should be high (use absolute for sign ambiguity)
    double sig_corr = fabs(correlation(extracted_signal, signal, N));
    ASSERT_GT(sig_corr, 0.90, "signal |correlation| > 0.90");
    
    // Extract noise (remaining components)
    double* extracted_noise = malloc(N * sizeof(double));
    ssa_get_noise(&ssa, 10, extracted_noise);
    
    // Noise should have low correlation with signal
    double noise_sig_corr = fabs(correlation(extracted_noise, signal, N));
    ASSERT_LT(noise_sig_corr, 0.4, "noise-signal |correlation| < 0.4");
    
    ssa_free(&ssa);
    free(x);
    free(signal);
    free(noise_true);
    free(extracted_signal);
    free(extracted_noise);
}

void test_forecasting(void) {
    int N = 500;
    int L = 100;
    int horizon = 50;
    
    // Pure sinusoid (should forecast perfectly)
    double* x = malloc((N + horizon) * sizeof(double));
    for (int i = 0; i < N + horizon; i++) {
        x[i] = sin(0.1 * i);
    }
    
    SSA ssa;
    ssa_init(&ssa, x, N, L);  // Only use first N points
    ssa_decompose(&ssa, 4, 100);
    
    int group[] = {0, 1};  // Use first pair (the sinusoid)
    double* forecast = malloc(horizon * sizeof(double));
    int ret = ssa_forecast(&ssa, group, 2, horizon, forecast);
    
    ASSERT_EQ(ret, 0, "forecast returns 0");
    
    // Compare forecast to true continuation (use absolute correlation)
    double corr = fabs(correlation(forecast, &x[N], horizon));
    ASSERT_GT(corr, 0.90, "forecast |correlation| > 0.90");
    
    ssa_free(&ssa);
    free(x);
    free(forecast);
}

void test_wcorrelation(void) {
    int N = 500;
    int L = 100;
    int k = 6;
    
    // Two independent sinusoids
    double* x = malloc(N * sizeof(double));
    for (int i = 0; i < N; i++) {
        x[i] = sin(0.1 * i) + 0.5 * sin(0.3 * i);
    }
    
    SSA ssa;
    ssa_init(&ssa, x, N, L);
    ssa_decompose(&ssa, k, 100);
    
    double* wcorr = malloc(k * k * sizeof(double));
    int ret = ssa_wcorrelation(&ssa, wcorr, k);
    
    ASSERT_EQ(ret, 0, "wcorrelation returns 0");
    
    // Diagonal should be 1.0 for non-zero variance components
    int nonzero_diag = 0;
    for (int i = 0; i < k; i++) {
        if (ssa.sigma[i] > 1e-6) {  // Only check non-zero components
            ASSERT_NEAR(wcorr[i * k + i], 1.0, 0.01, "diagonal = 1 for nonzero component");
            nonzero_diag++;
        }
    }
    ASSERT_GT(nonzero_diag, 0, "at least one nonzero component");
    
    // Components 0,1 should be correlated (first sinusoid pair)
    if (ssa.sigma[0] > 1e-6 && ssa.sigma[1] > 1e-6) {
        ASSERT_GT(fabs(wcorr[0 * k + 1]), 0.5, "pair 0-1 correlated");
    }
    
    ssa_free(&ssa);
    free(x);
    free(wcorr);
}

void test_edge_cases(void) {
    // Very short series
    {
        double x[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
        SSA ssa;
        int ret = ssa_init(&ssa, x, 10, 4);
        ASSERT_EQ(ret, 0, "short series init OK");
        
        ret = ssa_decompose(&ssa, 3, 100);
        ASSERT_EQ(ret, 0, "short series decompose OK");
        
        double* trend = malloc(10 * sizeof(double));
        ssa_get_trend(&ssa, trend);
        
        // For very short series, just check that decomposition works
        // and produces reasonable values
        ASSERT_TRUE(ssa.sigma[0] > 0, "short series has nonzero sigma");
        
        ssa_free(&ssa);
        free(trend);
    }
    
    // L = N/2 (maximum useful window)
    {
        int N = 100;
        double* x = malloc(N * sizeof(double));
        for (int i = 0; i < N; i++) x[i] = sin(0.2 * i);
        
        SSA ssa;
        int ret = ssa_init(&ssa, x, N, N/2);
        ASSERT_EQ(ret, 0, "L=N/2 init OK");
        
        ret = ssa_decompose(&ssa, 5, 100);
        ASSERT_EQ(ret, 0, "L=N/2 decompose OK");
        
        ssa_free(&ssa);
        free(x);
    }
    
    // Invalid inputs
    {
        double x[] = {1, 2, 3};
        SSA ssa;
        
        int ret = ssa_init(&ssa, x, 3, 2);  // N=3 too short
        ASSERT_EQ(ret, -1, "N=3 rejected");
        
        ret = ssa_init(&ssa, x, 10, 0);  // L=0 invalid
        ASSERT_EQ(ret, -1, "L=0 rejected");
        
        ret = ssa_init(NULL, x, 10, 5);  // NULL ssa
        ASSERT_EQ(ret, -1, "NULL ssa rejected");
    }
}

void test_benchmark_decomposition(void) {
    if (!g_benchmarks) return;
    
    printf("\n  [BENCH] SSA Decomposition Performance\n");
    
    int sizes[] = {1000, 5000, 10000, 50000, 100000};
    int n_sizes = sizeof(sizes) / sizeof(sizes[0]);
    
    for (int s = 0; s < n_sizes; s++) {
        int N = sizes[s];
        int L = N / 4;
        int k = 20;
        
        double* x = malloc(N * sizeof(double));
        srand(99999);
        for (int i = 0; i < N; i++) {
            x[i] = sin(0.01 * i) + 0.1 * randn();
        }
        
        SSA ssa;
        
        // Time init
        uint64_t t0 = get_nanos();
        ssa_init(&ssa, x, N, L);
        uint64_t t1 = get_nanos();
        double init_ms = (t1 - t0) / 1e6;
        
        // Time decompose
        uint64_t t2 = get_nanos();
        ssa_decompose(&ssa, k, 100);
        uint64_t t3 = get_nanos();
        double decomp_ms = (t3 - t2) / 1e6;
        
        // Time reconstruct
        int group[] = {0, 1, 2, 3, 4};
        double* output = malloc(N * sizeof(double));
        uint64_t t4 = get_nanos();
        ssa_reconstruct(&ssa, group, 5, output);
        uint64_t t5 = get_nanos();
        double recon_ms = (t5 - t4) / 1e6;
        
        printf("         N=%6d, L=%5d: init=%.1f ms, decomp=%.1f ms, recon=%.1f ms\n",
               N, L, init_ms, decomp_ms, recon_ms);
        
        ssa_free(&ssa);
        free(x);
        free(output);
    }
}

void test_benchmark_vs_naive(void) {
    if (!g_benchmarks) return;
    
    // Compare FFT-accelerated vs theoretical naive complexity
    // Naive: O(L * K) per matvec, O(k * iter * L * K) total
    // FFT:   O(N log N) per matvec, O(k * iter * N log N) total
    
    int N = 10000;
    int L = 2500;
    int K = N - L + 1;
    int k = 20;
    int iter = 100;
    
    // Theoretical naive ops: k * iter * 2 * L * K ≈ 37.5 billion
    double naive_ops = (double)k * iter * 2.0 * L * K;
    
    // Theoretical FFT ops: k * iter * 2 * N * log2(N) ≈ 53 million
    double fft_ops = (double)k * iter * 2.0 * N * log2(N);
    
    printf("\n  [BENCH] Complexity Comparison (N=%d, L=%d, k=%d)\n", N, L, k);
    printf("         Naive ops:  %.2e\n", naive_ops);
    printf("         FFT ops:    %.2e\n", fft_ops);
    printf("         Speedup:    %.0fx theoretical\n", naive_ops / fft_ops);
}

// -----------------------------
// Main
// -----------------------------

int main(int argc, char** argv) {
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--verbose") == 0 || strcmp(argv[i], "-v") == 0) {
            g_verbose = 1;
        } else if (strcmp(argv[i], "--bench") == 0 || strcmp(argv[i], "-b") == 0) {
            g_benchmarks = 1;
        }
    }
    
    printf("==========================================\n");
    printf("   SSA Test Suite (FFT-Accelerated)\n");
    printf("==========================================\n");
    
    RUN_TEST(test_initialization);
    RUN_TEST(test_decomposition);
    RUN_TEST(test_trend_extraction);
    RUN_TEST(test_cycle_detection);
    RUN_TEST(test_reconstruction_accuracy);
    RUN_TEST(test_variance_explained);
    RUN_TEST(test_noise_separation);
    RUN_TEST(test_forecasting);
    RUN_TEST(test_wcorrelation);
    RUN_TEST(test_edge_cases);
    
    if (g_benchmarks) {
        printf("\n=== Benchmarks ===\n");
        RUN_TEST(test_benchmark_decomposition);
        RUN_TEST(test_benchmark_vs_naive);
    }
    
    printf("\n==========================================\n");
    printf("   RESULTS: %d/%d passed", g_passed, g_total);
    if (g_failed > 0) {
        printf(" (%d FAILED)", g_failed);
    }
    printf("\n==========================================\n");
    
    return (g_failed > 0) ? 1 : 0;
}
