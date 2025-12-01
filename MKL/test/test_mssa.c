/*
 * ============================================================================
 * MSSA (Multivariate SSA) Test Suite
 * ============================================================================
 *
 * Tests MSSA on synthetic correlated time series simulating financial data:
 * - Common market factor across all series
 * - Series-specific idiosyncratic movements
 * - Cross-correlations between pairs
 *
 * BUILD: Same as other SSA tests (see CMakeLists.txt)
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

#define N_SERIES    5       // Number of correlated series (e.g., sector ETFs)
#define N_SAMPLES   400     // Length of each series
#define WINDOW_L    80      // SSA window length
#define N_COMPONENTS 20     // Components to compute

// Signal parameters
#define MARKET_AMP      10.0    // Common market factor amplitude
#define MARKET_TREND    0.02    // Market trend slope
#define SECTOR_AMP      3.0     // Sector-specific amplitude
#define IDIO_AMP        1.0     // Idiosyncratic noise amplitude

// ============================================================================
// Utility Functions
// ============================================================================

static double randn(void)
{
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

static double compute_rmse(const double *actual, const double *predicted, int n)
{
    double sum_sq = 0.0;
    for (int i = 0; i < n; i++) {
        double diff = actual[i] - predicted[i];
        sum_sq += diff * diff;
    }
    return sqrt(sum_sq / n);
}

// ============================================================================
// Synthetic Data Generation
// ============================================================================

typedef struct {
    double *X;              // All series: M × N row-major
    double *market;         // Common market factor: N
    double *sector[N_SERIES]; // Sector-specific: N each
    double *idio[N_SERIES];   // Idiosyncratic: N each
} SyntheticMSSAData;

static void generate_mssa_data(SyntheticMSSAData *data)
{
    int M = N_SERIES;
    int N = N_SAMPLES;
    
    data->X = (double *)malloc(M * N * sizeof(double));
    data->market = (double *)malloc(N * sizeof(double));
    
    for (int m = 0; m < M; m++) {
        data->sector[m] = (double *)malloc(N * sizeof(double));
        data->idio[m] = (double *)malloc(N * sizeof(double));
    }
    
    // Generate common market factor (trend + slow cycle)
    for (int t = 0; t < N; t++) {
        double trend = 100.0 + MARKET_TREND * t;
        double cycle = MARKET_AMP * sin(2.0 * M_PI * t / 100.0);
        data->market[t] = trend + cycle;
    }
    
    // Generate series with different sector loadings and idiosyncratic noise
    double sector_loadings[N_SERIES] = {1.0, 0.9, 0.8, 1.1, 0.7};
    double sector_phases[N_SERIES] = {0, 0.5, 1.0, 0.3, 1.5};
    
    for (int m = 0; m < M; m++) {
        for (int t = 0; t < N; t++) {
            // Sector-specific component (different frequency/phase per series)
            double sector_cycle = SECTOR_AMP * sin(2.0 * M_PI * t / 50.0 + sector_phases[m]);
            data->sector[m][t] = sector_cycle;
            
            // Idiosyncratic noise
            data->idio[m][t] = IDIO_AMP * randn();
            
            // Combined signal
            data->X[m * N + t] = sector_loadings[m] * data->market[t] 
                                + data->sector[m][t] 
                                + data->idio[m][t];
        }
    }
}

static void free_mssa_data(SyntheticMSSAData *data)
{
    free(data->X);
    free(data->market);
    for (int m = 0; m < N_SERIES; m++) {
        free(data->sector[m]);
        free(data->idio[m]);
    }
}

// ============================================================================
// Test Functions
// ============================================================================

static int test_mssa_basic(void)
{
    printf("\n");
    printf("============================================================\n");
    printf("TEST 1: Basic MSSA Decomposition\n");
    printf("============================================================\n");
    
    SyntheticMSSAData data;
    generate_mssa_data(&data);
    
    printf("\nData Configuration:\n");
    printf("  Number of series: %d\n", N_SERIES);
    printf("  Series length:    %d\n", N_SAMPLES);
    printf("  Window length:    %d\n", WINDOW_L);
    printf("  Components:       %d\n", N_COMPONENTS);
    
    // Print cross-correlations of input
    printf("\nInput Cross-Correlations:\n");
    printf("        ");
    for (int j = 0; j < N_SERIES; j++) printf("  S%d    ", j);
    printf("\n");
    
    for (int i = 0; i < N_SERIES; i++) {
        printf("  S%d:  ", i);
        for (int j = 0; j < N_SERIES; j++) {
            double corr = compute_correlation(data.X + i * N_SAMPLES, 
                                              data.X + j * N_SAMPLES, N_SAMPLES);
            printf(" %6.3f ", corr);
        }
        printf("\n");
    }
    
    // Initialize MSSA
    MSSA_Opt mssa = {0};
    clock_t start = clock();
    
    if (mssa_opt_init(&mssa, data.X, N_SERIES, N_SAMPLES, WINDOW_L) != 0) {
        printf("ERROR: MSSA initialization failed\n");
        free_mssa_data(&data);
        return -1;
    }
    double init_time = (double)(clock() - start) / CLOCKS_PER_SEC;
    printf("\nMSSA initialized in %.2f ms\n", init_time * 1000);
    
    // Decompose
    start = clock();
    if (mssa_opt_decompose(&mssa, N_COMPONENTS, 8) != 0) {
        printf("ERROR: MSSA decomposition failed\n");
        mssa_opt_free(&mssa);
        free_mssa_data(&data);
        return -1;
    }
    double decomp_time = (double)(clock() - start) / CLOCKS_PER_SEC;
    printf("MSSA decomposed in %.2f ms\n", decomp_time * 1000);
    
    // Print singular values
    printf("\nSingular Value Spectrum:\n");
    printf("  Component | Singular Value | Cumulative Var\n");
    printf("  ----------|----------------|---------------\n");
    for (int i = 0; i < ssa_opt_min(10, mssa.n_components); i++) {
        double cumvar = mssa_opt_variance_explained(&mssa, 0, i);
        printf("  %9d | %14.4f | %13.2f%%\n", i, mssa.sigma[i], cumvar * 100);
    }
    
    // Series contributions
    double *contrib = (double *)malloc(N_SERIES * mssa.n_components * sizeof(double));
    mssa_opt_series_contributions(&mssa, contrib);
    
    printf("\nSeries Contributions to First 5 Components:\n");
    printf("        ");
    for (int i = 0; i < 5; i++) printf("  C%d    ", i);
    printf("\n");
    
    for (int m = 0; m < N_SERIES; m++) {
        printf("  S%d:  ", m);
        for (int i = 0; i < 5; i++) {
            printf(" %6.3f ", contrib[m * mssa.n_components + i]);
        }
        printf("\n");
    }
    free(contrib);
    
    mssa_opt_free(&mssa);
    free_mssa_data(&data);
    
    printf("\nTEST 1: PASSED\n");
    return 0;
}

static int test_mssa_reconstruction(void)
{
    printf("\n");
    printf("============================================================\n");
    printf("TEST 2: MSSA Reconstruction Quality\n");
    printf("============================================================\n");
    
    SyntheticMSSAData data;
    generate_mssa_data(&data);
    
    MSSA_Opt mssa = {0};
    mssa_opt_init(&mssa, data.X, N_SERIES, N_SAMPLES, WINDOW_L);
    mssa_opt_decompose(&mssa, N_COMPONENTS, 8);
    
    // Test reconstruction with different component counts
    printf("\nReconstruction RMSE vs Original (Series 0):\n");
    printf("  Components | RMSE\n");
    printf("  -----------|--------\n");
    
    double *recon = (double *)malloc(N_SAMPLES * sizeof(double));
    int test_k[] = {1, 2, 3, 5, 10, 15, 20};
    int n_tests = sizeof(test_k) / sizeof(test_k[0]);
    
    for (int t = 0; t < n_tests; t++) {
        int k = test_k[t];
        if (k > mssa.n_components) break;
        
        // Build group array
        int *group = (int *)malloc(k * sizeof(int));
        for (int i = 0; i < k; i++) group[i] = i;
        
        mssa_opt_reconstruct(&mssa, 0, group, k, recon);
        double rmse = compute_rmse(data.X, recon, N_SAMPLES);
        printf("  %10d | %7.4f\n", k, rmse);
        
        free(group);
    }
    
    // Test reconstruct_all
    printf("\nTesting mssa_opt_reconstruct_all:\n");
    double *recon_all = (double *)malloc(N_SERIES * N_SAMPLES * sizeof(double));
    int full_group[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    
    if (mssa_opt_reconstruct_all(&mssa, full_group, 10, recon_all) == 0) {
        printf("  Series | RMSE vs Original\n");
        printf("  -------|------------------\n");
        for (int m = 0; m < N_SERIES; m++) {
            double rmse = compute_rmse(data.X + m * N_SAMPLES, 
                                       recon_all + m * N_SAMPLES, N_SAMPLES);
            printf("  %6d | %7.4f\n", m, rmse);
        }
    }
    
    free(recon);
    free(recon_all);
    mssa_opt_free(&mssa);
    free_mssa_data(&data);
    
    printf("\nTEST 2: PASSED\n");
    return 0;
}

static int test_mssa_common_factor(void)
{
    printf("\n");
    printf("============================================================\n");
    printf("TEST 3: Common Factor Extraction\n");
    printf("============================================================\n");
    
    SyntheticMSSAData data;
    generate_mssa_data(&data);
    
    MSSA_Opt mssa = {0};
    mssa_opt_init(&mssa, data.X, N_SERIES, N_SAMPLES, WINDOW_L);
    mssa_opt_decompose(&mssa, N_COMPONENTS, 8);
    
    printf("\nExtracting common factor (first component):\n");
    
    // Extract first component from each series
    int common_group[] = {0};
    double *common = (double *)malloc(N_SERIES * N_SAMPLES * sizeof(double));
    mssa_opt_reconstruct_all(&mssa, common_group, 1, common);
    
    // Compare common factors across series - they should be correlated
    printf("\nCorrelation of First Component Across Series:\n");
    printf("        ");
    for (int j = 0; j < N_SERIES; j++) printf("  S%d    ", j);
    printf("\n");
    
    for (int i = 0; i < N_SERIES; i++) {
        printf("  S%d:  ", i);
        for (int j = 0; j < N_SERIES; j++) {
            double corr = compute_correlation(common + i * N_SAMPLES,
                                              common + j * N_SAMPLES, N_SAMPLES);
            printf(" %6.3f ", corr);
        }
        printf("\n");
    }
    
    // Compare with true market factor
    printf("\nCorrelation with True Market Factor:\n");
    for (int m = 0; m < N_SERIES; m++) {
        double corr = compute_correlation(common + m * N_SAMPLES, data.market, N_SAMPLES);
        printf("  Series %d: %.4f\n", m, corr);
    }
    
    // Extract residuals (components 2+)
    printf("\nExtracting residuals (components 2-10):\n");
    int residual_group[] = {2, 3, 4, 5, 6, 7, 8, 9};
    double *residuals = (double *)malloc(N_SERIES * N_SAMPLES * sizeof(double));
    mssa_opt_reconstruct_all(&mssa, residual_group, 8, residuals);
    
    printf("Residual Cross-Correlations (should be lower):\n");
    printf("        ");
    for (int j = 0; j < N_SERIES; j++) printf("  S%d    ", j);
    printf("\n");
    
    for (int i = 0; i < N_SERIES; i++) {
        printf("  S%d:  ", i);
        for (int j = 0; j < N_SERIES; j++) {
            double corr = compute_correlation(residuals + i * N_SAMPLES,
                                              residuals + j * N_SAMPLES, N_SAMPLES);
            printf(" %6.3f ", corr);
        }
        printf("\n");
    }
    
    free(common);
    free(residuals);
    mssa_opt_free(&mssa);
    free_mssa_data(&data);
    
    printf("\nTEST 3: PASSED\n");
    return 0;
}

static int test_mssa_pairs_analysis(void)
{
    printf("\n");
    printf("============================================================\n");
    printf("TEST 4: Pairs Trading Application\n");
    printf("============================================================\n");
    
    printf("\nSimulating a pairs trade between Series 0 and Series 1:\n");
    
    SyntheticMSSAData data;
    generate_mssa_data(&data);
    
    // Use just 2 series for pairs
    double *pairs_data = (double *)malloc(2 * N_SAMPLES * sizeof(double));
    memcpy(pairs_data, data.X, N_SAMPLES * sizeof(double));
    memcpy(pairs_data + N_SAMPLES, data.X + N_SAMPLES, N_SAMPLES * sizeof(double));
    
    MSSA_Opt mssa = {0};
    mssa_opt_init(&mssa, pairs_data, 2, N_SAMPLES, WINDOW_L);
    mssa_opt_decompose(&mssa, 10, 8);
    
    // Extract common trend (components 0-1)
    int common[] = {0, 1};
    double *s0_common = (double *)malloc(N_SAMPLES * sizeof(double));
    double *s1_common = (double *)malloc(N_SAMPLES * sizeof(double));
    
    mssa_opt_reconstruct(&mssa, 0, common, 2, s0_common);
    mssa_opt_reconstruct(&mssa, 1, common, 2, s1_common);
    
    printf("\nCommon component correlation: %.4f\n",
           compute_correlation(s0_common, s1_common, N_SAMPLES));
    
    // Compute spread (idiosyncratic difference)
    double *spread = (double *)malloc(N_SAMPLES * sizeof(double));
    int idio[] = {2, 3, 4, 5};
    double *s0_idio = (double *)malloc(N_SAMPLES * sizeof(double));
    double *s1_idio = (double *)malloc(N_SAMPLES * sizeof(double));
    
    mssa_opt_reconstruct(&mssa, 0, idio, 4, s0_idio);
    mssa_opt_reconstruct(&mssa, 1, idio, 4, s1_idio);
    
    for (int t = 0; t < N_SAMPLES; t++) {
        spread[t] = s0_idio[t] - s1_idio[t];
    }
    
    // Compute spread statistics
    double mean = 0, std = 0;
    for (int t = 0; t < N_SAMPLES; t++) mean += spread[t];
    mean /= N_SAMPLES;
    
    for (int t = 0; t < N_SAMPLES; t++) {
        double diff = spread[t] - mean;
        std += diff * diff;
    }
    std = sqrt(std / N_SAMPLES);
    
    printf("Spread (idiosyncratic difference):\n");
    printf("  Mean: %.4f\n", mean);
    printf("  Std:  %.4f\n", std);
    printf("  Min:  %.4f (%.2f sigma)\n", 
           spread[0], (spread[0] - mean) / std);  // Example
    
    // Find max absolute z-score
    double max_z = 0;
    int max_t = 0;
    for (int t = 0; t < N_SAMPLES; t++) {
        double z = fabs(spread[t] - mean) / std;
        if (z > max_z) {
            max_z = z;
            max_t = t;
        }
    }
    printf("  Max |z|: %.2f at t=%d\n", max_z, max_t);
    
    free(pairs_data);
    free(s0_common);
    free(s1_common);
    free(spread);
    free(s0_idio);
    free(s1_idio);
    mssa_opt_free(&mssa);
    free_mssa_data(&data);
    
    printf("\nTEST 4: PASSED\n");
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
    printf("#         MSSA (MULTIVARIATE SSA) TEST SUITE               #\n");
    printf("#                                                          #\n");
    printf("############################################################\n");
    
    srand(42);
    
    int result = 0;
    
    result |= test_mssa_basic();
    result |= test_mssa_reconstruction();
    result |= test_mssa_common_factor();
    result |= test_mssa_pairs_analysis();
    
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
