/*
 * V-forecast Test
 * Compare recurrent forecast (R-forecast) vs vector forecast (V-forecast)
 */

#define SSA_OPT_IMPLEMENTATION
#define SSA_OPT_IMPLEMENTATION
#include "ssa_opt_r2c.h"

#define SSA_OPT_ANALYSIS_IMPLEMENTATION
#include "ssa_opt_analysis.h"

#define SSA_OPT_FORECAST_IMPLEMENTATION
#include "ssa_opt_forecast.h"

#define SSA_OPT_ADVANCED_IMPLEMENTATION
#include "ssa_opt_advanced.h"
#include <stdio.h>
#include <math.h>

#ifdef _WIN32
#include <windows.h>
static ssa_real get_time_ms(void) {
    LARGE_INTEGER freq, counter;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&counter);
    return (ssa_real)counter.QuadPart * 1000.0 / (ssa_real)freq.QuadPart;
}
#else
#include <time.h>
static ssa_real get_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}
#endif

int main(void) {
    printf("=== V-Forecast Test ===\n\n");
    
    // Test parameters
    int N = 1000;
    int L = 250;
    int k = 30;
    int n_forecast = 50;
    
    // Generate test signal: trend + seasonal + noise
    ssa_real *x = (ssa_real *)mkl_malloc(N * sizeof(ssa_real), 64);
    for (int i = 0; i < N; i++) {
        ssa_real t = (ssa_real)i / N;
        x[i] = 10.0 * t +                          // Linear trend
               5.0 * sin(2.0 * M_PI * 12 * t) +    // Annual cycle
               2.0 * sin(2.0 * M_PI * 52 * t) +    // Weekly cycle
               0.5 * ((ssa_real)rand() / RAND_MAX - 0.5);  // Noise
    }
    
    // Initialize and decompose
    SSA_Opt ssa = {0};
    ssa_opt_init(&ssa, x, N, L);
    ssa_opt_prepare(&ssa, k, 8);
    ssa_opt_decompose_randomized(&ssa, k, 8);
    
    printf("Signal: N=%d, L=%d, k=%d\n", N, L, k);
    printf("Forecast horizon: %d steps\n\n", n_forecast);
    
    // Component group for forecasting (trend + main periodic)
    int group[] = {0, 1, 2, 3, 4, 5};
    int n_group = 6;
    
    // Allocate forecast arrays
    ssa_real *r_forecast = (ssa_real *)mkl_malloc(n_forecast * sizeof(ssa_real), 64);
    ssa_real *v_forecast = (ssa_real *)mkl_malloc(n_forecast * sizeof(ssa_real), 64);
    
    // ======================
    // Test 1: Compare outputs
    // ======================
    printf("Test 1: Comparing R-forecast vs V-forecast outputs\n");
    
    int ret1 = ssa_opt_forecast(&ssa, group, n_group, n_forecast, r_forecast);
    int ret2 = ssa_opt_vforecast(&ssa, group, n_group, n_forecast, v_forecast);
    
    if (ret1 != 0 || ret2 != 0) {
        printf("FAILED: forecast returned error\n");
        return 1;
    }
    
    // Compare
    ssa_real max_diff = 0.0;
    ssa_real sum_diff = 0.0;
    for (int i = 0; i < n_forecast; i++) {
        ssa_real diff = fabs(r_forecast[i] - v_forecast[i]);
        if (diff > max_diff) max_diff = diff;
        sum_diff += diff;
    }
    ssa_real mean_diff = sum_diff / n_forecast;
    
    printf("  Max difference:  %.2e\n", max_diff);
    printf("  Mean difference: %.2e\n", mean_diff);
    
    // For single-step mathematically equivalent, but accumulated errors differ
    if (max_diff < 1e-6) {
        printf("  Result: IDENTICAL (as expected for short horizons)\n\n");
    } else if (max_diff < 0.1) {
        printf("  Result: SIMILAR (small numerical differences)\n\n");
    } else {
        printf("  Result: DIFFERENT (expected for long horizons)\n\n");
    }
    
    // Print first/last few values
    printf("  First 5 forecasts:\n");
    printf("    R-forecast: ");
    for (int i = 0; i < 5; i++) printf("%.4f ", r_forecast[i]);
    printf("\n    V-forecast: ");
    for (int i = 0; i < 5; i++) printf("%.4f ", v_forecast[i]);
    printf("\n");
    
    printf("  Last 5 forecasts:\n");
    printf("    R-forecast: ");
    for (int i = n_forecast - 5; i < n_forecast; i++) printf("%.4f ", r_forecast[i]);
    printf("\n    V-forecast: ");
    for (int i = n_forecast - 5; i < n_forecast; i++) printf("%.4f ", v_forecast[i]);
    printf("\n\n");
    
    // ======================
    // Test 2: Performance
    // ======================
    printf("Test 2: Performance comparison\n");
    
    int n_runs = 1000;
    ssa_real t0, t1;
    
    // R-forecast timing
    t0 = get_time_ms();
    for (int i = 0; i < n_runs; i++) {
        ssa_opt_forecast(&ssa, group, n_group, n_forecast, r_forecast);
    }
    t1 = get_time_ms();
    ssa_real r_time = (t1 - t0) / n_runs;
    
    // V-forecast timing
    t0 = get_time_ms();
    for (int i = 0; i < n_runs; i++) {
        ssa_opt_vforecast(&ssa, group, n_group, n_forecast, v_forecast);
    }
    t1 = get_time_ms();
    ssa_real v_time = (t1 - t0) / n_runs;
    
    printf("  R-forecast: %.3f ms/call\n", r_time);
    printf("  V-forecast: %.3f ms/call\n", v_time);
    printf("  Ratio: %.2fx\n\n", v_time / r_time);
    
    // ======================
    // Test 3: V-forecast fast
    // ======================
    printf("Test 3: V-forecast fast (from base signal)\n");
    
    ssa_real *reconstructed = (ssa_real *)mkl_malloc(N * sizeof(ssa_real), 64);
    ssa_opt_reconstruct(&ssa, group, n_group, reconstructed);
    
    t0 = get_time_ms();
    for (int i = 0; i < n_runs; i++) {
        ssa_opt_vforecast_fast(&ssa, group, n_group, reconstructed, N, n_forecast, v_forecast);
    }
    t1 = get_time_ms();
    ssa_real v_fast_time = (t1 - t0) / n_runs;
    
    printf("  V-forecast fast: %.3f ms/call\n", v_fast_time);
    printf("  Speedup vs V-forecast: %.2fx\n\n", v_time / v_fast_time);
    
    // ======================
    // Test 4: Long horizon stability
    // ======================
    printf("Test 4: Long horizon stability (500 steps)\n");
    
    int long_horizon = 500;
    ssa_real *r_long = (ssa_real *)mkl_malloc(long_horizon * sizeof(ssa_real), 64);
    ssa_real *v_long = (ssa_real *)mkl_malloc(long_horizon * sizeof(ssa_real), 64);
    
    ssa_opt_forecast(&ssa, group, n_group, long_horizon, r_long);
    ssa_opt_vforecast(&ssa, group, n_group, long_horizon, v_long);
    
    // Check for NaN/Inf (instability)
    int r_stable = 1, v_stable = 1;
    for (int i = 0; i < long_horizon; i++) {
        if (!isfinite(r_long[i])) r_stable = 0;
        if (!isfinite(v_long[i])) v_stable = 0;
    }
    
    printf("  R-forecast stable: %s\n", r_stable ? "YES" : "NO");
    printf("  V-forecast stable: %s\n", v_stable ? "YES" : "NO");
    
    // Check variance explosion
    ssa_real r_var_start = 0, r_var_end = 0;
    ssa_real v_var_start = 0, v_var_end = 0;
    for (int i = 0; i < 50; i++) {
        r_var_start += r_long[i] * r_long[i];
        v_var_start += v_long[i] * v_long[i];
    }
    for (int i = long_horizon - 50; i < long_horizon; i++) {
        r_var_end += r_long[i] * r_long[i];
        v_var_end += v_long[i] * v_long[i];
    }
    r_var_start /= 50; r_var_end /= 50;
    v_var_start /= 50; v_var_end /= 50;
    
    printf("  R-forecast variance ratio (end/start): %.2f\n", r_var_end / r_var_start);
    printf("  V-forecast variance ratio (end/start): %.2f\n", v_var_end / v_var_start);
    
    // Cleanup
    mkl_free(x);
    mkl_free(r_forecast);
    mkl_free(v_forecast);
    mkl_free(reconstructed);
    mkl_free(r_long);
    mkl_free(v_long);
    ssa_opt_free(&ssa);
    
    printf("\n=== V-Forecast Test Complete ===\n");
    return 0;
}
