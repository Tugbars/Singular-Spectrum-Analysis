/*
 * Cadzow Iterations Test
 * Test finite-rank signal approximation via iterative projection
 */

#define SSA_OPT_IMPLEMENTATION
#include "ssa_opt.h"
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

ssa_real compute_rmse(const ssa_real *a, const ssa_real *b, int n) {
    ssa_real sum = 0.0;
    for (int i = 0; i < n; i++) {
        ssa_real d = a[i] - b[i];
        sum += d * d;
    }
    return sqrt(sum / n);
}

ssa_real compute_snr(const ssa_real *signal, const ssa_real *noisy, int n) {
    ssa_real sig_power = 0.0, noise_power = 0.0;
    for (int i = 0; i < n; i++) {
        sig_power += signal[i] * signal[i];
        ssa_real noise = noisy[i] - signal[i];
        noise_power += noise * noise;
    }
    return 10.0 * log10(sig_power / noise_power);
}

int main(void) {
    printf("=== Cadzow Iterations Test ===\n\n");
    
    // Test parameters
    int N = 500;
    int L = 125;
    int rank = 4;  // True signal has rank 4 (2 sinusoids = 2 pairs)
    ssa_real noise_std = 0.5;
    
    // Allocate arrays
    ssa_real *x_true = (ssa_real *)mkl_malloc(N * sizeof(ssa_real), 64);
    ssa_real *x_noisy = (ssa_real *)mkl_malloc(N * sizeof(ssa_real), 64);
    ssa_real *x_ssa = (ssa_real *)mkl_malloc(N * sizeof(ssa_real), 64);
    ssa_real *x_cadzow = (ssa_real *)mkl_malloc(N * sizeof(ssa_real), 64);
    
    // Generate true signal: sum of two sinusoids (rank 4)
    for (int i = 0; i < N; i++) {
        ssa_real t = (ssa_real)i / N;
        x_true[i] = 3.0 * sin(2.0 * M_PI * 5 * t) +   // Frequency 5
                    2.0 * sin(2.0 * M_PI * 12 * t);   // Frequency 12
    }
    
    // Add noise
    srand(42);
    for (int i = 0; i < N; i++) {
        ssa_real u1 = (ssa_real)rand() / RAND_MAX;
        ssa_real u2 = (ssa_real)rand() / RAND_MAX;
        ssa_real noise = noise_std * sqrt(-2.0 * log(u1 + 1e-10)) * cos(2.0 * M_PI * u2);
        x_noisy[i] = x_true[i] + noise;
    }
    
    printf("Signal: N=%d, L=%d, true_rank=%d\n", N, L, rank);
    printf("Noise std: %.2f\n", noise_std);
    printf("Input SNR: %.2f dB\n\n", compute_snr(x_true, x_noisy, N));
    
    // =============================
    // Test 1: Single-pass SSA vs Cadzow
    // =============================
    printf("Test 1: Single-pass SSA vs Cadzow denoising\n");
    
    // Single-pass SSA
    SSA_Opt ssa = {0};
    ssa_opt_init(&ssa, x_noisy, N, L);
    ssa_opt_prepare(&ssa, rank, 8);
    ssa_opt_decompose_randomized(&ssa, rank, 8);
    int group[] = {0, 1, 2, 3};
    ssa_opt_reconstruct(&ssa, group, rank, x_ssa);
    ssa_opt_free(&ssa);
    
    // Cadzow iterations
    SSA_CadzowResult cadzow_result;
    ssa_real t0 = get_time_ms();
    int iters = ssa_opt_cadzow(x_noisy, N, L, rank, 20, 1e-9, x_cadzow, &cadzow_result);
    ssa_real t1 = get_time_ms();
    
    printf("  Single-pass SSA:\n");
    printf("    RMSE vs true: %.6f\n", compute_rmse(x_true, x_ssa, N));
    printf("    Output SNR:   %.2f dB\n", compute_snr(x_true, x_ssa, N));
    
    printf("  Cadzow (%d iterations, converged=%s):\n", 
           cadzow_result.iterations,
           cadzow_result.converged > 0.5 ? "yes" : "no");
    printf("    RMSE vs true: %.6f\n", compute_rmse(x_true, x_cadzow, N));
    printf("    Output SNR:   %.2f dB\n", compute_snr(x_true, x_cadzow, N));
    printf("    Final diff:   %.2e\n", cadzow_result.final_diff);
    printf("    Time:         %.2f ms\n\n", t1 - t0);
    
    // =============================
    // Test 2: Weighted Cadzow (regularization)
    // =============================
    printf("Test 2: Weighted Cadzow (alpha blending)\n");
    
    ssa_real alphas[] = {1.0, 0.9, 0.7, 0.5};
    int n_alphas = 4;
    
    for (int a = 0; a < n_alphas; a++) {
        ssa_real alpha = alphas[a];
        SSA_CadzowResult res;
        ssa_opt_cadzow_weighted(x_noisy, N, L, rank, 20, 1e-9, alpha, x_cadzow, &res);
        printf("  alpha=%.1f: RMSE=%.6f, SNR=%.2f dB\n", 
               alpha, compute_rmse(x_true, x_cadzow, N), compute_snr(x_true, x_cadzow, N));
    }
    printf("\n");
    
    // =============================
    // Test 3: Convergence behavior
    // =============================
    printf("Test 3: Convergence with different max_iter\n");
    
    int max_iters[] = {1, 2, 5, 10, 20, 50};
    int n_tests = 6;
    
    for (int t = 0; t < n_tests; t++) {
        SSA_CadzowResult res;
        ssa_opt_cadzow(x_noisy, N, L, rank, max_iters[t], 1e-12, x_cadzow, &res);
        printf("  max_iter=%2d: iters=%2d, RMSE=%.6f, diff=%.2e, conv=%s\n",
               max_iters[t], res.iterations, compute_rmse(x_true, x_cadzow, N),
               res.final_diff, res.converged > 0.5 ? "yes" : "no");
    }
    printf("\n");
    
    // =============================
    // Test 4: Effect of rank
    // =============================
    printf("Test 4: Effect of rank parameter\n");
    
    int ranks[] = {2, 4, 6, 8, 10};
    int n_ranks = 5;
    
    for (int r = 0; r < n_ranks; r++) {
        SSA_CadzowResult res;
        ssa_opt_cadzow(x_noisy, N, L, ranks[r], 20, 1e-9, x_cadzow, &res);
        printf("  rank=%2d: iters=%2d, RMSE=%.6f, SNR=%.2f dB\n",
               ranks[r], res.iterations, compute_rmse(x_true, x_cadzow, N),
               compute_snr(x_true, x_cadzow, N));
    }
    printf("\n");
    
    // =============================
    // Test 5: Performance timing
    // =============================
    printf("Test 5: Performance comparison\n");
    
    int n_runs = 10;
    
    // Single-pass SSA timing
    t0 = get_time_ms();
    for (int i = 0; i < n_runs; i++) {
        SSA_Opt ssa2 = {0};
        ssa_opt_init(&ssa2, x_noisy, N, L);
        ssa_opt_prepare(&ssa2, rank, 8);
        ssa_opt_decompose_randomized(&ssa2, rank, 8);
        ssa_opt_reconstruct(&ssa2, group, rank, x_ssa);
        ssa_opt_free(&ssa2);
    }
    t1 = get_time_ms();
    ssa_real ssa_time = (t1 - t0) / n_runs;
    
    // Cadzow timing (typically 5-10 iterations)
    t0 = get_time_ms();
    for (int i = 0; i < n_runs; i++) {
        SSA_CadzowResult res;
        ssa_opt_cadzow(x_noisy, N, L, rank, 10, 1e-9, x_cadzow, &res);
    }
    t1 = get_time_ms();
    ssa_real cadzow_time = (t1 - t0) / n_runs;
    
    printf("  Single-pass SSA: %.2f ms\n", ssa_time);
    printf("  Cadzow (10 iter): %.2f ms\n", cadzow_time);
    printf("  Slowdown: %.1fx\n\n", cadzow_time / ssa_time);
    
    // Cleanup
    mkl_free(x_true);
    mkl_free(x_noisy);
    mkl_free(x_ssa);
    mkl_free(x_cadzow);
    
    printf("=== Cadzow Test Complete ===\n");
    return 0;
}
