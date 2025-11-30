/*
 * SSA MKL Example with Proper 14900KF Configuration
 * 
 * Compile with CMake or:
 *   source /opt/intel/oneapi/setvars.sh
 *   gcc -O3 -march=native -fopenmp -DSSA_USE_MKL -DSSA_USE_MKL_CONFIG \
 *       -I${MKLROOT}/include -o ssa_example ssa_example.c \
 *       -L${MKLROOT}/lib/intel64 -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core \
 *       -liomp5 -lpthread -lm
 * 
 * Run:
 *   ./ssa_example
 */

// Only define if not already set by CMake
#ifndef SSA_USE_MKL
#define SSA_USE_MKL
#endif

#define SSA_USE_MKL_CONFIG
#define SSA_OPT_IMPLEMENTATION

#include "mkl_config.h"
#include "ssa_opt.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

// Timing
#ifdef _WIN32
#include <windows.h>
static inline double get_time_ms(void) {
    LARGE_INTEGER freq, counter;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart * 1000.0 / freq.QuadPart;
}
#elif defined(__linux__)
#include <time.h>
static inline double get_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}
#else
static inline double get_time_ms(void) {
    return (double)clock() * 1000.0 / CLOCKS_PER_SEC;
}
#endif

// Generate test signal: trend + cycles + noise
void generate_signal(double* x, int N) {
    unsigned int seed = 42;
    for (int i = 0; i < N; i++) {
        double t = (double)i / N;
        
        // Trend
        double trend = 100.0 + 50.0 * t + 10.0 * t * t;
        
        // Cycles
        double cycle1 = 20.0 * sin(2.0 * M_PI * i / 100.0);   // Period 100
        double cycle2 = 10.0 * sin(2.0 * M_PI * i / 25.0);    // Period 25
        
        // Noise
        seed = seed * 1103515245 + 12345;
        double u1 = (double)((seed >> 16) & 0x7fff) / 32768.0;
        seed = seed * 1103515245 + 12345;
        double u2 = (double)((seed >> 16) & 0x7fff) / 32768.0;
        double noise = 5.0 * sqrt(-2.0 * log(u1 + 1e-10)) * cos(2.0 * M_PI * u2);
        
        x[i] = trend + cycle1 + cycle2 + noise;
    }
}

void print_separator(void) {
    printf("================================================================\n");
}

int main(int argc, char** argv) {
    int verbose = 1;
    int run_benchmark = 0;
    
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-q") == 0 || strcmp(argv[i], "--quiet") == 0) {
            verbose = 0;
        }
        if (strcmp(argv[i], "-b") == 0 || strcmp(argv[i], "--bench") == 0) {
            run_benchmark = 1;
        }
    }
    
    print_separator();
    printf("SSA with Intel MKL - 14900KF Configuration Example\n");
    print_separator();
    printf("\n");
    
    // =========================================================================
    // Step 1: Configure MKL for 14900KF
    // =========================================================================
    
    printf("Step 1: Configuring MKL for Intel 14900KF...\n\n");
    
#ifdef SSA_USE_MKL
    // Option A: Use our configuration helper (recommended)
    mkl_config_14900kf(verbose);
    
    // Option B: Manual configuration
    // mkl_set_num_threads(16);        // P-cores only
    // mkl_set_dynamic(0);             // Disable dynamic threads
    
    printf("\n");
#else
    printf("MKL not enabled. Using built-in FFT.\n\n");
#endif
    
    // =========================================================================
    // Step 2: Generate Test Data
    // =========================================================================
    
    printf("Step 2: Generating test signal...\n");
    
    int N = 10000;      // Signal length
    int L = N / 4;      // Window length
    int k = 20;         // Number of components
    
    double* x = (double*)malloc(N * sizeof(double));
    generate_signal(x, N);
    
    printf("  Signal length: %d\n", N);
    printf("  Window length: %d\n", L);
    printf("  Components: %d\n\n", k);
    
    // =========================================================================
    // Step 3: Run SSA
    // =========================================================================
    
    printf("Step 3: Running SSA decomposition...\n");
    
    SSA_Opt ssa;
    
    // Init
    double t0 = get_time_ms();
    int ret = ssa_opt_init(&ssa, x, N, L);
    double t1 = get_time_ms();
    
    if (ret != 0) {
        fprintf(stderr, "ERROR: ssa_opt_init failed\n");
        free(x);
        return 1;
    }
    printf("  Init time: %.2f ms\n", t1 - t0);
    
    // Decompose
    double t2 = get_time_ms();
    ret = ssa_opt_decompose(&ssa, k, 150);
    double t3 = get_time_ms();
    
    if (ret != 0) {
        fprintf(stderr, "ERROR: ssa_opt_decompose failed\n");
        ssa_opt_free(&ssa);
        free(x);
        return 1;
    }
    printf("  Decompose time: %.2f ms\n", t3 - t2);
    
    // =========================================================================
    // Step 4: Analyze Results
    // =========================================================================
    
    printf("\nStep 4: Analyzing results...\n\n");
    
    printf("  Singular values and variance explained:\n");
    printf("  %-5s  %-12s  %-10s  %-12s\n", "Comp", "Sigma", "Variance%", "Cumulative%");
    printf("  %-5s  %-12s  %-10s  %-12s\n", "----", "-----", "---------", "-----------");
    
    double cumulative = 0.0;
    for (int i = 0; i < k && i < 10; i++) {
        double var = ssa_opt_variance_explained(&ssa, i, i) * 100.0;
        cumulative += var;
        printf("  %-5d  %-12.3f  %-10.2f  %-12.2f\n", 
               i, ssa.sigma[i], var, cumulative);
    }
    if (k > 10) {
        printf("  ... (%d more components)\n", k - 10);
    }
    
    // =========================================================================
    // Step 5: Reconstruct Components
    // =========================================================================
    
    printf("\nStep 5: Reconstructing components...\n");
    
    double* trend = (double*)malloc(N * sizeof(double));
    double* cycles = (double*)malloc(N * sizeof(double));
    double* noise = (double*)malloc(N * sizeof(double));
    
    // Trend (component 0)
    double t4 = get_time_ms();
    ssa_opt_get_trend(&ssa, trend);
    double t5 = get_time_ms();
    printf("  Trend reconstruction: %.2f ms\n", t5 - t4);
    
    // Cycles (components 1-4)
    int cycle_group[] = {1, 2, 3, 4};
    double t6 = get_time_ms();
    ssa_opt_reconstruct(&ssa, cycle_group, 4, cycles);
    double t7 = get_time_ms();
    printf("  Cycle reconstruction: %.2f ms\n", t7 - t6);
    
    // Noise (remaining components)
    double t8 = get_time_ms();
    ssa_opt_get_noise(&ssa, 10, noise);
    double t9 = get_time_ms();
    printf("  Noise reconstruction: %.2f ms\n", t9 - t8);
    
    // =========================================================================
    // Step 6: Summary Statistics
    // =========================================================================
    
    printf("\nStep 6: Summary statistics...\n\n");
    
    // Compute statistics for each component
    double trend_mean = 0, trend_var = 0;
    double cycle_mean = 0, cycle_var = 0;
    double noise_mean = 0, noise_var = 0;
    
    for (int i = 0; i < N; i++) {
        trend_mean += trend[i];
        cycle_mean += cycles[i];
        noise_mean += noise[i];
    }
    trend_mean /= N;
    cycle_mean /= N;
    noise_mean /= N;
    
    for (int i = 0; i < N; i++) {
        trend_var += (trend[i] - trend_mean) * (trend[i] - trend_mean);
        cycle_var += (cycles[i] - cycle_mean) * (cycles[i] - cycle_mean);
        noise_var += (noise[i] - noise_mean) * (noise[i] - noise_mean);
    }
    trend_var /= N;
    cycle_var /= N;
    noise_var /= N;
    
    printf("  %-12s  %-12s  %-12s\n", "Component", "Mean", "Std Dev");
    printf("  %-12s  %-12s  %-12s\n", "---------", "----", "-------");
    printf("  %-12s  %-12.2f  %-12.2f\n", "Trend", trend_mean, sqrt(trend_var));
    printf("  %-12s  %-12.2f  %-12.2f\n", "Cycles", cycle_mean, sqrt(cycle_var));
    printf("  %-12s  %-12.2f  %-12.2f\n", "Noise", noise_mean, sqrt(noise_var));
    
    // =========================================================================
    // Step 7: Benchmark (optional)
    // =========================================================================
    
    if (run_benchmark) {
        printf("\n");
        print_separator();
        printf("Benchmark: Different Problem Sizes\n");
        print_separator();
        printf("\n");
        
        int sizes[] = {1000, 5000, 10000, 50000, 100000};
        int n_sizes = sizeof(sizes) / sizeof(sizes[0]);
        
        printf("%-10s  %-10s  %-12s  %-12s  %-12s\n",
               "N", "L", "Init (ms)", "Decomp (ms)", "Recon (ms)");
        printf("%-10s  %-10s  %-12s  %-12s  %-12s\n",
               "---", "---", "---------", "-----------", "----------");
        
        for (int s = 0; s < n_sizes; s++) {
            int Nb = sizes[s];
            int Lb = Nb / 4;
            int kb = 20;
            
            double* xb = (double*)malloc(Nb * sizeof(double));
            generate_signal(xb, Nb);
            
            SSA_Opt ssa_bench;
            
            double tb0 = get_time_ms();
            ssa_opt_init(&ssa_bench, xb, Nb, Lb);
            double tb1 = get_time_ms();
            
            double tb2 = get_time_ms();
            ssa_opt_decompose(&ssa_bench, kb, 100);
            double tb3 = get_time_ms();
            
            int grp[] = {0, 1, 2, 3, 4};
            double* out = (double*)malloc(Nb * sizeof(double));
            double tb4 = get_time_ms();
            ssa_opt_reconstruct(&ssa_bench, grp, 5, out);
            double tb5 = get_time_ms();
            
            printf("%-10d  %-10d  %-12.1f  %-12.1f  %-12.1f\n",
                   Nb, Lb, tb1-tb0, tb3-tb2, tb5-tb4);
            
            ssa_opt_free(&ssa_bench);
            free(xb);
            free(out);
        }
    }
    
    // =========================================================================
    // Cleanup
    // =========================================================================
    
    printf("\n");
    print_separator();
    printf("Done!\n");
    print_separator();
    
    ssa_opt_free(&ssa);
    free(x);
    free(trend);
    free(cycles);
    free(noise);
    
    return 0;
}