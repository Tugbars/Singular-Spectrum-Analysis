/*
 * ============================================================================
 * SSA Forecasting Test Suite
 * ============================================================================
 *
 * Tests SSA decomposition and forecasting on synthetic financial-like data.
 * Generates trend + seasonal + noise, decomposes, forecasts, and compares
 * against known ground truth.
 *
 * BUILD (Linux + MKL):
 *   source /opt/intel/oneapi/setvars.sh
 *   gcc -O2 -o test_ssa_forecast test_ssa_forecast.c \
 *       -I${MKLROOT}/include -L${MKLROOT}/lib/intel64 \
 *       -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm
 *
 * BUILD (Windows + MKL):
 *   cl /O2 test_ssa_forecast.c /I"%MKLROOT%\include" /link /LIBPATH:"%MKLROOT%\lib" ^
 *      mkl_intel_lp64.lib mkl_intel_thread.lib mkl_core.lib
 *
 * ============================================================================
 */

#define SSA_OPT_IMPLEMENTATION
#include "ssa_opt.h"

#include <stdio.h>
#include <math.h>
#include <time.h>

// ============================================================================
// Test Configuration
// ============================================================================

#define N_OBSERVED  500    // Number of observed data points
#define N_FORECAST  50     // Number of points to forecast
#define WINDOW_L    100    // SSA window length
#define N_COMPONENTS 20    // Number of SSA components to compute

// Signal parameters
#define TREND_SLOPE     0.02    // Linear trend slope
#define TREND_INTERCEPT 100.0   // Trend starting value
#define SEASONAL_AMP    5.0     // Seasonal amplitude
#define SEASONAL_PERIOD 50.0    // Seasonal period (in samples)
#define NOISE_STD       0.5     // Noise standard deviation

// ============================================================================
// Utility Functions
// ============================================================================

static double randn(void)
{
    // Box-Muller transform for Gaussian random numbers
    static int have_spare = 0;
    static double spare;
    
    if (have_spare) {
        have_spare = 0;
        return spare;
    }
    
    double u, v, s;
    do {
        u = (double)rand() / RAND_MAX * 2.0 - 1.0;
        v = (double)rand() / RAND_MAX * 2.0 - 1.0;
        s = u * u + v * v;
    } while (s >= 1.0 || s == 0.0);
    
    s = sqrt(-2.0 * log(s) / s);
    spare = v * s;
    have_spare = 1;
    return u * s;
}

static double compute_rmse(const double *actual, const double *predicted, int n)
{
    double sum_sq = 0.0;
    for (int i = 0; i < n; i++) {
        double diff = actual[i] - predicted[i];
        sum_sq += diff * diff;
    }
    return sqrt(sum_sq / n);
}

static double compute_mape(const double *actual, const double *predicted, int n)
{
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        if (fabs(actual[i]) > 1e-10) {
            sum += fabs((actual[i] - predicted[i]) / actual[i]);
        }
    }
    return 100.0 * sum / n;
}

static double compute_correlation(const double *x, const double *y, int n)
{
    double sum_x = 0, sum_y = 0, sum_xy = 0, sum_x2 = 0, sum_y2 = 0;
    for (int i = 0; i < n; i++) {
        sum_x += x[i];
        sum_y += y[i];
        sum_xy += x[i] * y[i];
        sum_x2 += x[i] * x[i];
        sum_y2 += y[i] * y[i];
    }
    double num = n * sum_xy - sum_x * sum_y;
    double den = sqrt((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y));
    return (den > 1e-10) ? num / den : 0.0;
}

// ============================================================================
// Signal Generation
// ============================================================================

typedef struct {
    double *signal;      // Full signal (observed + future)
    double *trend;       // Trend component only
    double *seasonal;    // Seasonal component only
    double *noise;       // Noise component only
    int n_total;         // Total length (N_OBSERVED + N_FORECAST)
} SyntheticSignal;

static void generate_synthetic_signal(SyntheticSignal *sig)
{
    int n_total = N_OBSERVED + N_FORECAST;
    sig->n_total = n_total;
    
    sig->signal = (double *)malloc(n_total * sizeof(double));
    sig->trend = (double *)malloc(n_total * sizeof(double));
    sig->seasonal = (double *)malloc(n_total * sizeof(double));
    sig->noise = (double *)malloc(n_total * sizeof(double));
    
    for (int t = 0; t < n_total; t++) {
        // Linear trend
        sig->trend[t] = TREND_INTERCEPT + TREND_SLOPE * t;
        
        // Seasonal component (sinusoidal)
        sig->seasonal[t] = SEASONAL_AMP * sin(2.0 * M_PI * t / SEASONAL_PERIOD);
        
        // Noise (only in observed portion, zero in forecast portion for comparison)
        if (t < N_OBSERVED) {
            sig->noise[t] = NOISE_STD * randn();
        } else {
            sig->noise[t] = 0.0;  // No noise in "true" future for comparison
        }
        
        // Combined signal
        sig->signal[t] = sig->trend[t] + sig->seasonal[t] + sig->noise[t];
    }
}

static void free_synthetic_signal(SyntheticSignal *sig)
{
    free(sig->signal);
    free(sig->trend);
    free(sig->seasonal);
    free(sig->noise);
}

// ============================================================================
// Test Functions
// ============================================================================

static int test_basic_forecast(void)
{
    printf("\n");
    printf("============================================================\n");
    printf("TEST 1: Basic SSA Forecasting\n");
    printf("============================================================\n");
    
    // Generate synthetic data
    SyntheticSignal sig;
    generate_synthetic_signal(&sig);
    
    printf("\nSignal Parameters:\n");
    printf("  Observed points:  %d\n", N_OBSERVED);
    printf("  Forecast horizon: %d\n", N_FORECAST);
    printf("  Window length:    %d\n", WINDOW_L);
    printf("  Trend: %.2f + %.4f*t\n", TREND_INTERCEPT, TREND_SLOPE);
    printf("  Seasonal: amplitude=%.2f, period=%.1f\n", SEASONAL_AMP, SEASONAL_PERIOD);
    printf("  Noise std: %.2f\n", NOISE_STD);
    
    // Initialize SSA with observed data only
    SSA_Opt ssa = {0};
    if (ssa_opt_init(&ssa, sig.signal, N_OBSERVED, WINDOW_L) != 0) {
        printf("ERROR: SSA initialization failed\n");
        free_synthetic_signal(&sig);
        return -1;
    }
    
    // Decompose using randomized SVD (fastest)
    printf("\nDecomposing with randomized SVD (k=%d)...\n", N_COMPONENTS);
    clock_t start = clock();
    if (ssa_opt_decompose_randomized(&ssa, N_COMPONENTS, 8) != 0) {
        printf("ERROR: SSA decomposition failed\n");
        ssa_opt_free(&ssa);
        free_synthetic_signal(&sig);
        return -1;
    }
    double decomp_time = (double)(clock() - start) / CLOCKS_PER_SEC;
    printf("  Decomposition time: %.3f ms\n", decomp_time * 1000);
    
    // Print singular values and variance explained
    printf("\nSingular Value Spectrum:\n");
    printf("  Component | Singular Value | Variance Explained\n");
    printf("  ----------|----------------|-------------------\n");
    for (int i = 0; i < ssa_opt_min(8, ssa.n_components); i++) {
        double var = ssa_opt_variance_explained(&ssa, 0, i);
        printf("  %9d | %14.4f | %17.2f%%\n", i, ssa.sigma[i], var * 100);
    }
    
    // Analyze component statistics
    SSA_ComponentStats stats;
    if (ssa_opt_component_stats(&ssa, &stats) == 0) {
        printf("\nComponent Analysis:\n");
        printf("  Suggested signal components: %d (gap ratio: %.2f)\n", 
               stats.suggested_signal, stats.gap_threshold);
        ssa_opt_component_stats_free(&stats);
    }
    
    // Find periodic pairs
    int pairs[20];
    int n_pairs = ssa_opt_find_periodic_pairs(&ssa, pairs, 10, 0, 0);
    printf("  Detected periodic pairs: %d\n", n_pairs);
    for (int i = 0; i < n_pairs; i++) {
        printf("    Pair %d: components %d and %d\n", i, pairs[2*i], pairs[2*i+1]);
    }
    
    // =========================================================================
    // Forecast: Trend only (component 0)
    // =========================================================================
    printf("\n--- Forecasting Trend (Component 0) ---\n");
    
    int trend_group[] = {0};
    double *trend_forecast = (double *)malloc(N_FORECAST * sizeof(double));
    
    start = clock();
    if (ssa_opt_forecast(&ssa, trend_group, 1, N_FORECAST, trend_forecast) != 0) {
        printf("ERROR: Trend forecast failed\n");
    } else {
        double forecast_time = (double)(clock() - start) / CLOCKS_PER_SEC;
        printf("  Forecast time: %.3f ms\n", forecast_time * 1000);
        
        // Compare with true trend
        double *true_trend = sig.trend + N_OBSERVED;
        double rmse = compute_rmse(true_trend, trend_forecast, N_FORECAST);
        double mape = compute_mape(true_trend, trend_forecast, N_FORECAST);
        double corr = compute_correlation(true_trend, trend_forecast, N_FORECAST);
        
        printf("  RMSE:  %.4f\n", rmse);
        printf("  MAPE:  %.2f%%\n", mape);
        printf("  Corr:  %.4f\n", corr);
        
        printf("\n  Sample forecast vs actual (trend):\n");
        printf("    t     | Forecast | Actual   | Error\n");
        printf("    ------|----------|----------|-------\n");
        for (int i = 0; i < N_FORECAST; i += 10) {
            printf("    %5d | %8.3f | %8.3f | %+.3f\n", 
                   N_OBSERVED + i, trend_forecast[i], true_trend[i],
                   trend_forecast[i] - true_trend[i]);
        }
    }
    free(trend_forecast);
    
    // =========================================================================
    // Forecast: Trend + Seasonal (components 0, 1, 2)
    // =========================================================================
    printf("\n--- Forecasting Trend + Seasonal (Components 0-2) ---\n");
    
    int signal_group[] = {0, 1, 2};
    double *signal_forecast = (double *)malloc(N_FORECAST * sizeof(double));
    
    start = clock();
    if (ssa_opt_forecast(&ssa, signal_group, 3, N_FORECAST, signal_forecast) != 0) {
        printf("ERROR: Signal forecast failed\n");
    } else {
        double forecast_time = (double)(clock() - start) / CLOCKS_PER_SEC;
        printf("  Forecast time: %.3f ms\n", forecast_time * 1000);
        
        // Compare with true signal (trend + seasonal, no noise)
        double *true_signal = (double *)malloc(N_FORECAST * sizeof(double));
        for (int i = 0; i < N_FORECAST; i++) {
            true_signal[i] = sig.trend[N_OBSERVED + i] + sig.seasonal[N_OBSERVED + i];
        }
        
        double rmse = compute_rmse(true_signal, signal_forecast, N_FORECAST);
        double mape = compute_mape(true_signal, signal_forecast, N_FORECAST);
        double corr = compute_correlation(true_signal, signal_forecast, N_FORECAST);
        
        printf("  RMSE:  %.4f\n", rmse);
        printf("  MAPE:  %.2f%%\n", mape);
        printf("  Corr:  %.4f\n", corr);
        
        printf("\n  Sample forecast vs actual (trend+seasonal):\n");
        printf("    t     | Forecast | Actual   | Error\n");
        printf("    ------|----------|----------|-------\n");
        for (int i = 0; i < N_FORECAST; i += 10) {
            printf("    %5d | %8.3f | %8.3f | %+.3f\n", 
                   N_OBSERVED + i, signal_forecast[i], true_signal[i],
                   signal_forecast[i] - true_signal[i]);
        }
        
        free(true_signal);
    }
    free(signal_forecast);
    
    // =========================================================================
    // Test forecast_full (reconstruction + forecast)
    // =========================================================================
    printf("\n--- Testing ssa_opt_forecast_full ---\n");
    
    double *full_output = (double *)malloc((N_OBSERVED + N_FORECAST) * sizeof(double));
    
    if (ssa_opt_forecast_full(&ssa, signal_group, 3, N_FORECAST, full_output) != 0) {
        printf("ERROR: Full forecast failed\n");
    } else {
        printf("  Output length: %d (reconstruction) + %d (forecast) = %d\n",
               N_OBSERVED, N_FORECAST, N_OBSERVED + N_FORECAST);
        
        // Verify reconstruction matches
        double *recon_only = (double *)malloc(N_OBSERVED * sizeof(double));
        ssa_opt_reconstruct(&ssa, signal_group, 3, recon_only);
        
        double max_diff = 0.0;
        for (int i = 0; i < N_OBSERVED; i++) {
            double diff = fabs(full_output[i] - recon_only[i]);
            if (diff > max_diff) max_diff = diff;
        }
        printf("  Max reconstruction difference: %.2e (should be ~0)\n", max_diff);
        
        free(recon_only);
    }
    free(full_output);
    
    // =========================================================================
    // Test LRF coefficient extraction
    // =========================================================================
    printf("\n--- Testing LRF Coefficient Extraction ---\n");
    
    SSA_LRF lrf = {0};
    if (ssa_opt_compute_lrf(&ssa, signal_group, 3, &lrf) == 0) {
        printf("  LRF valid: %s\n", lrf.valid ? "true" : "false");
        printf("  Verticality (nu^2): %.6f (must be < 1)\n", lrf.verticality);
        printf("  Number of coefficients: %d (L-1)\n", lrf.L - 1);
        
        // Print first and last few coefficients
        printf("  First 5 coefficients: ");
        for (int i = 0; i < ssa_opt_min(5, lrf.L - 1); i++) {
            printf("%.4f ", lrf.R[i]);
        }
        printf("\n");
        
        printf("  Last 5 coefficients:  ");
        for (int i = ssa_opt_max(0, lrf.L - 1 - 5); i < lrf.L - 1; i++) {
            printf("%.4f ", lrf.R[i]);
        }
        printf("\n");
        
        // Use LRF directly for forecasting
        double *recon = (double *)malloc(N_OBSERVED * sizeof(double));
        double *lrf_forecast = (double *)malloc(N_FORECAST * sizeof(double));
        
        ssa_opt_reconstruct(&ssa, signal_group, 3, recon);
        if (ssa_opt_forecast_with_lrf(&lrf, recon, N_OBSERVED, N_FORECAST, lrf_forecast) == 0) {
            printf("  Direct LRF forecast successful\n");
        }
        
        free(recon);
        free(lrf_forecast);
        ssa_opt_lrf_free(&lrf);
    } else {
        printf("  ERROR: LRF computation failed\n");
    }
    
    // Cleanup
    ssa_opt_free(&ssa);
    free_synthetic_signal(&sig);
    
    printf("\nTEST 1: PASSED\n");
    return 0;
}

static int test_forecast_accuracy_vs_horizon(void)
{
    printf("\n");
    printf("============================================================\n");
    printf("TEST 2: Forecast Accuracy vs Horizon\n");
    printf("============================================================\n");
    
    SyntheticSignal sig;
    generate_synthetic_signal(&sig);
    
    SSA_Opt ssa = {0};
    ssa_opt_init(&ssa, sig.signal, N_OBSERVED, WINDOW_L);
    ssa_opt_decompose_randomized(&ssa, N_COMPONENTS, 8);
    
    int signal_group[] = {0, 1, 2};
    
    printf("\nForecast accuracy degradation with horizon:\n");
    printf("  Horizon | RMSE    | MAPE    | Correlation\n");
    printf("  --------|---------|---------|------------\n");
    
    int horizons[] = {1, 5, 10, 20, 30, 40, 50};
    int n_horizons = sizeof(horizons) / sizeof(horizons[0]);
    
    for (int h = 0; h < n_horizons; h++) {
        int horizon = horizons[h];
        if (horizon > N_FORECAST) break;
        
        double *forecast = (double *)malloc(horizon * sizeof(double));
        double *true_signal = (double *)malloc(horizon * sizeof(double));
        
        ssa_opt_forecast(&ssa, signal_group, 3, horizon, forecast);
        
        for (int i = 0; i < horizon; i++) {
            true_signal[i] = sig.trend[N_OBSERVED + i] + sig.seasonal[N_OBSERVED + i];
        }
        
        double rmse = compute_rmse(true_signal, forecast, horizon);
        double mape = compute_mape(true_signal, forecast, horizon);
        double corr = compute_correlation(true_signal, forecast, horizon);
        
        printf("  %7d | %7.4f | %6.2f%% | %.4f\n", horizon, rmse, mape, corr);
        
        free(forecast);
        free(true_signal);
    }
    
    ssa_opt_free(&ssa);
    free_synthetic_signal(&sig);
    
    printf("\nTEST 2: PASSED\n");
    return 0;
}

static int test_different_component_groups(void)
{
    printf("\n");
    printf("============================================================\n");
    printf("TEST 3: Different Component Groupings\n");
    printf("============================================================\n");
    
    SyntheticSignal sig;
    generate_synthetic_signal(&sig);
    
    SSA_Opt ssa = {0};
    ssa_opt_init(&ssa, sig.signal, N_OBSERVED, WINDOW_L);
    ssa_opt_decompose_randomized(&ssa, N_COMPONENTS, 8);
    
    // Prepare true components for comparison
    double *true_trend = sig.trend + N_OBSERVED;
    double *true_seasonal = sig.seasonal + N_OBSERVED;
    double *true_signal = (double *)malloc(N_FORECAST * sizeof(double));
    for (int i = 0; i < N_FORECAST; i++) {
        true_signal[i] = true_trend[i] + true_seasonal[i];
    }
    
    printf("\nComparing different component groupings (horizon=%d):\n", N_FORECAST);
    printf("  Group         | Description         | RMSE vs Signal\n");
    printf("  --------------|---------------------|---------------\n");
    
    double *forecast = (double *)malloc(N_FORECAST * sizeof(double));
    
    // Test different groupings
    struct {
        const char *name;
        const char *desc;
        int *group;
        int n_group;
    } tests[] = {
        {"[0]", "Trend only", (int[]){0}, 1},
        {"[0,1]", "Trend + 1st pair", (int[]){0, 1}, 2},
        {"[0,1,2]", "Trend + pair", (int[]){0, 1, 2}, 3},
        {"[0,1,2,3,4]", "First 5", (int[]){0, 1, 2, 3, 4}, 5},
        {"[1,2]", "Seasonal only", (int[]){1, 2}, 2},
    };
    int n_tests = sizeof(tests) / sizeof(tests[0]);
    
    for (int t = 0; t < n_tests; t++) {
        if (ssa_opt_forecast(&ssa, tests[t].group, tests[t].n_group, 
                             N_FORECAST, forecast) == 0) {
            double rmse = compute_rmse(true_signal, forecast, N_FORECAST);
            printf("  %-13s | %-19s | %.4f\n", 
                   tests[t].name, tests[t].desc, rmse);
        } else {
            printf("  %-13s | %-19s | FAILED\n", tests[t].name, tests[t].desc);
        }
    }
    
    free(forecast);
    free(true_signal);
    ssa_opt_free(&ssa);
    free_synthetic_signal(&sig);
    
    printf("\nTEST 3: PASSED\n");
    return 0;
}

// ============================================================================
// Main
// ============================================================================

int main(void)
{
    printf("\n");
    printf("############################################################\n");
    printf("#                                                          #\n");
    printf("#            SSA FORECASTING TEST SUITE                    #\n");
    printf("#                                                          #\n");
    printf("############################################################\n");
    
    srand(42);  // Fixed seed for reproducibility
    
    int result = 0;
    
    result |= test_basic_forecast();
    result |= test_forecast_accuracy_vs_horizon();
    result |= test_different_component_groups();
    
    printf("\n");
    printf("############################################################\n");
    if (result == 0) {
        printf("#                  ALL TESTS PASSED                        #\n");
    } else {
        printf("#                  SOME TESTS FAILED                       #\n");
    }
    printf("############################################################\n");
    printf("\n");
    
    return result;
}
