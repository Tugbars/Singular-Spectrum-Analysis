/*
 * test_gapfill.c - Test gap filling functionality
 * 
 * Build (Windows):
 *   cl /O2 test_gapfill.c /I"%MKLROOT%\include" mkl_rt.lib /Fe:test_gapfill.exe
 * 
 * Build (Linux):
 *   gcc -O2 test_gapfill.c -I${MKLROOT}/include -lmkl_rt -lm -o test_gapfill
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define SSA_OPT_IMPLEMENTATION
#include "ssa_opt.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Generate test signal: trend + two sinusoids
void generate_signal(ssa_real *x, int N) {
    for (int i = 0; i < N; i++) {
        ssa_real t = (ssa_real)i / N;
        x[i] = 0.02 * i +                          // trend
               sin(2 * M_PI * i / 50.0) +          // period 50
               0.5 * sin(2 * M_PI * i / 20.0);     // period 20
    }
}

// Create gaps (set to NaN)
void create_gaps(ssa_real *x, int start, int end) {
    for (int i = start; i < end; i++) {
        x[i] = NAN;
    }
}

// Check if value is NaN
int is_nan(ssa_real x) {
    return x != x;
}

// Count NaNs
int count_nans(const ssa_real *x, int N) {
    int count = 0;
    for (int i = 0; i < N; i++) {
        if (is_nan(x[i])) count++;
    }
    return count;
}

// Compute RMSE at gap positions
ssa_real compute_gap_rmse(const ssa_real *true_signal, const ssa_real *filled, 
                        const int *gap_mask, int N) {
    ssa_real sum_sq = 0;
    int count = 0;
    for (int i = 0; i < N; i++) {
        if (gap_mask[i]) {
            ssa_real diff = filled[i] - true_signal[i];
            sum_sq += diff * diff;
            count++;
        }
    }
    return (count > 0) ? sqrt(sum_sq / count) : 0.0;
}

int main() {
    printf("=== Gap Filling Test ===\n\n");
    
    // Parameters
    int N = 500;
    int L = 100;
    int rank = 6;
    int max_iter = 30;
    ssa_real tol = 1e-8;
    
    // Allocate
    ssa_real *true_signal = (ssa_real *)malloc(N * sizeof(ssa_real));
    ssa_real *x = (ssa_real *)malloc(N * sizeof(ssa_real));
    int *gap_mask = (int *)calloc(N, sizeof(int));
    
    if (!true_signal || !x || !gap_mask) {
        printf("ERROR: Memory allocation failed\n");
        return 1;
    }
    
    // Generate true signal
    generate_signal(true_signal, N);
    
    // Copy and create gaps
    memcpy(x, true_signal, N * sizeof(ssa_real));
    
    // Gap regions: 50-64, 150-179, 300-319, 400-409
    int gap_regions[][2] = {{50, 65}, {150, 180}, {300, 320}, {400, 410}};
    int n_regions = 4;
    
    for (int r = 0; r < n_regions; r++) {
        int start = gap_regions[r][0];
        int end = gap_regions[r][1];
        create_gaps(x, start, end);
        for (int i = start; i < end; i++) {
            gap_mask[i] = 1;
        }
    }
    
    int n_gaps = count_nans(x, N);
    printf("Signal: N=%d, L=%d, rank=%d\n", N, L, rank);
    printf("Gaps: %d values (%.1f%%)\n", n_gaps, 100.0 * n_gaps / N);
    printf("Gap regions: ");
    for (int r = 0; r < n_regions; r++) {
        printf("[%d-%d] ", gap_regions[r][0], gap_regions[r][1] - 1);
    }
    printf("\n\n");
    
    // Debug: Check a few values before filling
    printf("Before filling:\n");
    printf("  x[0] = %.4f (should be valid)\n", x[0]);
    printf("  x[50] = %.4f (should be NaN)\n", x[50]);
    printf("  x[100] = %.4f (should be valid)\n", x[100]);
    printf("  NaN count: %d\n\n", count_nans(x, N));
    
    // === Test iterative gap filling ===
    printf("Testing ssa_opt_gapfill (iterative)...\n");
    
    SSA_GapFillResult result = {0};
    int ret = ssa_opt_gapfill(x, N, L, rank, max_iter, tol, &result);
    
    printf("  Return code: %d\n", ret);
    printf("  Iterations: %d\n", result.iterations);
    printf("  Final diff: %.2e\n", result.final_diff);
    printf("  Converged: %d\n", result.converged);
    printf("  n_gaps reported: %d\n", result.n_gaps);
    
    // Check result
    int nans_after = count_nans(x, N);
    printf("  NaN count after: %d\n", nans_after);
    
    if (nans_after == 0) {
        ssa_real rmse = compute_gap_rmse(true_signal, x, gap_mask, N);
        printf("  RMSE at gap positions: %.6f\n", rmse);
        printf("  SUCCESS: All gaps filled\n");
    } else {
        printf("  FAILED: %d NaN values remain\n", nans_after);
        
        // Debug: which positions still have NaN?
        printf("  First few NaN positions: ");
        int printed = 0;
        for (int i = 0; i < N && printed < 10; i++) {
            if (is_nan(x[i])) {
                printf("%d ", i);
                printed++;
            }
        }
        printf("\n");
    }
    
    printf("\n");
    
    // === Test simple gap filling ===
    printf("Testing ssa_opt_gapfill_simple...\n");
    
    // Reset signal with gaps
    memcpy(x, true_signal, N * sizeof(ssa_real));
    for (int r = 0; r < n_regions; r++) {
        create_gaps(x, gap_regions[r][0], gap_regions[r][1]);
    }
    
    SSA_GapFillResult result2 = {0};
    ret = ssa_opt_gapfill_simple(x, N, L, rank, &result2);
    
    printf("  Return code: %d\n", ret);
    printf("  n_gaps reported: %d\n", result2.n_gaps);
    
    nans_after = count_nans(x, N);
    printf("  NaN count after: %d\n", nans_after);
    
    if (nans_after == 0) {
        ssa_real rmse = compute_gap_rmse(true_signal, x, gap_mask, N);
        printf("  RMSE at gap positions: %.6f\n", rmse);
        printf("  SUCCESS: All gaps filled\n");
    } else {
        printf("  FAILED: %d NaN values remain\n", nans_after);
    }
    
    // Cleanup
    free(true_signal);
    free(x);
    free(gap_mask);
    
    printf("\n=== Test Complete ===\n");
    return 0;
}
