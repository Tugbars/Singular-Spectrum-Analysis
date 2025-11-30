/*
 * MKL Configuration for Intel Hybrid CPUs (14th Gen Raptor Lake, etc.)
 * 
 * Intel 14900KF specifics:
 *   - 8 P-cores (Performance) with HT = 16 threads
 *   - 16 E-cores (Efficient) = 16 threads  
 *   - Total: 32 threads, but NOT equal performance
 *   - P-cores: ~2x faster than E-cores for AVX-512 workloads
 * 
 * For compute-intensive FFT workloads, you typically want:
 *   - Use P-cores only (threads 0-15 on most systems)
 *   - Or let MKL auto-detect but limit thread count
 * 
 * Usage:
 *   #include "mkl_config.h"
 *   
 *   int main() {
 *       mkl_config_init();  // Call once at startup
 *       // ... your code ...
 *   }
 * 
 * Environment variables (set before running):
 *   export MKL_NUM_THREADS=16          # Use 16 threads (P-cores only)
 *   export MKL_DYNAMIC=FALSE           # Don't dynamically adjust
 *   export OMP_NUM_THREADS=16          # OpenMP threads
 *   export OMP_PLACES=cores            # Bind to cores
 *   export OMP_PROC_BIND=close         # Keep threads close
 *   export KMP_AFFINITY=granularity=fine,compact,1,0
 *   export KMP_HW_SUBSET=8c,2t         # 8 cores, 2 threads each (P-cores)
 */

#ifndef MKL_CONFIG_H
#define MKL_CONFIG_H

#ifdef SSA_USE_MKL

#include <mkl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

// ============================================================================
// CPU Detection
// ============================================================================

typedef struct {
    int num_p_cores;        // Performance cores
    int num_e_cores;        // Efficiency cores
    int threads_per_p_core; // Usually 2 (hyperthreading)
    int threads_per_e_core; // Usually 1
    int total_threads;
    int is_hybrid;          // 1 if P+E cores detected
    int has_avx512;
    int has_avx2;
    char cpu_name[64];
} CPUInfo;

static inline void cpuid(int info[4], int leaf, int subleaf) {
    __asm__ __volatile__ (
        "cpuid"
        : "=a"(info[0]), "=b"(info[1]), "=c"(info[2]), "=d"(info[3])
        : "a"(leaf), "c"(subleaf)
    );
}

static CPUInfo detect_cpu(void) {
    CPUInfo info = {0};
    int regs[4];
    
    // Get CPU brand string
    cpuid(regs, 0x80000002, 0);
    memcpy(info.cpu_name, regs, 16);
    cpuid(regs, 0x80000003, 0);
    memcpy(info.cpu_name + 16, regs, 16);
    cpuid(regs, 0x80000004, 0);
    memcpy(info.cpu_name + 32, regs, 16);
    
    // Check for AVX-512
    cpuid(regs, 7, 0);
    info.has_avx512 = (regs[1] >> 16) & 1;  // AVX-512F
    info.has_avx2 = (regs[1] >> 5) & 1;     // AVX2
    
    // Detect hybrid architecture (12th gen+)
    // This is a simplified detection - real detection uses CPUID leaf 0x1A
    cpuid(regs, 0, 0);
    int max_leaf = regs[0];
    
    if (max_leaf >= 0x1A) {
        cpuid(regs, 0x1A, 0);
        int core_type = (regs[0] >> 24) & 0xFF;
        // core_type: 0x20 = Atom/E-core, 0x40 = Core/P-core
        info.is_hybrid = 1;
        
        // For 14900KF: 8P + 16E
        // This is hardcoded - proper detection would enumerate all cores
        if (strstr(info.cpu_name, "14900") || strstr(info.cpu_name, "14700")) {
            info.num_p_cores = 8;
            info.num_e_cores = 16;
            info.threads_per_p_core = 2;
            info.threads_per_e_core = 1;
        } else if (strstr(info.cpu_name, "13900") || strstr(info.cpu_name, "13700")) {
            info.num_p_cores = 8;
            info.num_e_cores = 16;
            info.threads_per_p_core = 2;
            info.threads_per_e_core = 1;
        } else if (strstr(info.cpu_name, "12900") || strstr(info.cpu_name, "12700")) {
            info.num_p_cores = 8;
            info.num_e_cores = 8;
            info.threads_per_p_core = 2;
            info.threads_per_e_core = 1;
        } else {
            // Generic hybrid assumption
            info.num_p_cores = 6;
            info.num_e_cores = 8;
            info.threads_per_p_core = 2;
            info.threads_per_e_core = 1;
        }
    } else {
        // Non-hybrid (older Intel or AMD)
        info.is_hybrid = 0;
        cpuid(regs, 1, 0);
        int logical_cores = (regs[1] >> 16) & 0xFF;
        info.num_p_cores = logical_cores / 2;  // Assume HT
        info.threads_per_p_core = 2;
    }
    
    info.total_threads = info.num_p_cores * info.threads_per_p_core + 
                         info.num_e_cores * info.threads_per_e_core;
    
    return info;
}

// ============================================================================
// MKL Configuration
// ============================================================================

typedef enum {
    MKL_CONFIG_AUTO,        // Let MKL decide (may use all cores)
    MKL_CONFIG_P_CORES,     // Use P-cores only (recommended for compute)
    MKL_CONFIG_ALL_CORES,   // Use all cores
    MKL_CONFIG_SEQUENTIAL,  // Single-threaded (for small problems)
    MKL_CONFIG_CUSTOM       // Use MKL_NUM_THREADS environment variable
} MKLConfigMode;

typedef struct {
    MKLConfigMode mode;
    int num_threads;
    int verbose;
    CPUInfo cpu;
} MKLConfig;

static MKLConfig g_mkl_config = {0};

/*
 * Initialize MKL with optimal settings for the detected CPU.
 * 
 * @param mode       Threading mode (see MKLConfigMode)
 * @param verbose    Print configuration info
 * @return           0 on success, -1 on error
 */
static int mkl_config_init_ex(MKLConfigMode mode, int verbose) {
    g_mkl_config.cpu = detect_cpu();
    g_mkl_config.mode = mode;
    g_mkl_config.verbose = verbose;
    
    CPUInfo* cpu = &g_mkl_config.cpu;
    
    if (verbose) {
        printf("=== MKL Configuration ===\n");
        printf("CPU: %s\n", cpu->cpu_name);
        printf("Hybrid: %s\n", cpu->is_hybrid ? "Yes" : "No");
        if (cpu->is_hybrid) {
            printf("P-cores: %d (x%d threads = %d)\n", 
                   cpu->num_p_cores, cpu->threads_per_p_core,
                   cpu->num_p_cores * cpu->threads_per_p_core);
            printf("E-cores: %d (x%d threads = %d)\n",
                   cpu->num_e_cores, cpu->threads_per_e_core,
                   cpu->num_e_cores * cpu->threads_per_e_core);
        }
        printf("Total threads: %d\n", cpu->total_threads);
        printf("AVX-512: %s\n", cpu->has_avx512 ? "Yes" : "No");
        printf("AVX2: %s\n", cpu->has_avx2 ? "Yes" : "No");
    }
    
    // Determine thread count
    int num_threads;
    switch (mode) {
        case MKL_CONFIG_SEQUENTIAL:
            num_threads = 1;
            break;
            
        case MKL_CONFIG_P_CORES:
            // Use P-cores only (best for compute-intensive work)
            num_threads = cpu->num_p_cores * cpu->threads_per_p_core;
            if (num_threads == 0) num_threads = cpu->total_threads / 2;
            break;
            
        case MKL_CONFIG_ALL_CORES:
            num_threads = cpu->total_threads;
            break;
            
        case MKL_CONFIG_CUSTOM:
            // Use environment variable
            {
                char* env = getenv("MKL_NUM_THREADS");
                if (env) {
                    num_threads = atoi(env);
                } else {
                    num_threads = cpu->num_p_cores * cpu->threads_per_p_core;
                }
            }
            break;
            
        case MKL_CONFIG_AUTO:
        default:
            // For hybrid CPUs, default to P-cores only
            if (cpu->is_hybrid) {
                num_threads = cpu->num_p_cores * cpu->threads_per_p_core;
            } else {
                num_threads = cpu->total_threads;
            }
            break;
    }
    
    g_mkl_config.num_threads = num_threads;
    
    // Configure MKL
    mkl_set_num_threads(num_threads);
    
    // Disable dynamic thread adjustment (important for consistent performance)
    mkl_set_dynamic(0);
    
#ifdef _OPENMP
    omp_set_num_threads(num_threads);
#endif
    
    if (verbose) {
        printf("MKL threads: %d\n", num_threads);
        printf("MKL dynamic: disabled\n");
        
        // Print MKL version
        MKLVersion ver;
        mkl_get_version(&ver);
        printf("MKL version: %d.%d.%d (%s)\n", 
               ver.MajorVersion, ver.MinorVersion, ver.UpdateVersion,
               ver.ProductStatus);
        printf("=========================\n\n");
    }
    
    return 0;
}

/*
 * Initialize MKL with default settings (P-cores only for hybrid CPUs).
 */
static int mkl_config_init(void) {
    return mkl_config_init_ex(MKL_CONFIG_AUTO, 0);
}

/*
 * Initialize MKL with verbose output.
 */
static int mkl_config_init_verbose(void) {
    return mkl_config_init_ex(MKL_CONFIG_AUTO, 1);
}

/*
 * Get optimal thread count for a given problem size.
 * 
 * For small problems, using fewer threads may be faster due to overhead.
 * 
 * @param n          Problem size (e.g., FFT length)
 * @return           Recommended thread count
 */
static int mkl_config_optimal_threads(int n) {
    CPUInfo* cpu = &g_mkl_config.cpu;
    int max_threads = cpu->num_p_cores * cpu->threads_per_p_core;
    if (max_threads == 0) max_threads = 8;
    
    // Empirical thresholds for FFT
    if (n < 1024) return 1;           // Very small: single thread
    if (n < 4096) return 2;           // Small: 2 threads
    if (n < 16384) return 4;          // Medium: 4 threads
    if (n < 65536) return 8;          // Large: 8 threads
    return max_threads;               // Very large: all P-core threads
}

/*
 * Temporarily set thread count for a specific operation.
 * Useful for tuning per-operation parallelism.
 */
static void mkl_config_set_threads(int n) {
    mkl_set_num_threads(n);
#ifdef _OPENMP
    omp_set_num_threads(n);
#endif
}

/*
 * Restore default thread count.
 */
static void mkl_config_restore_threads(void) {
    mkl_set_num_threads(g_mkl_config.num_threads);
#ifdef _OPENMP
    omp_set_num_threads(g_mkl_config.num_threads);
#endif
}

// ============================================================================
// Thread Affinity for Hybrid CPUs
// ============================================================================

/*
 * Set thread affinity to P-cores only.
 * 
 * On most systems with 14900KF, P-core threads are 0-15.
 * This function sets KMP_AFFINITY to bind to those cores.
 * 
 * IMPORTANT: Call BEFORE mkl_config_init() or at program start.
 */
static void mkl_config_set_p_core_affinity(void) {
    // Intel OpenMP affinity
    // granularity=fine: bind to logical processors
    // compact: pack threads close together
    // 1,0: start from processor 0, stride 1
    setenv("KMP_AFFINITY", "granularity=fine,compact,1,0", 1);
    
    // Limit to P-cores (8 cores, 2 threads each = 16 threads)
    setenv("KMP_HW_SUBSET", "8c,2t", 1);
    
    // Alternative: explicit processor list
    // setenv("KMP_AFFINITY", "explicit,proclist=[0-15]", 1);
}

/*
 * Set thread affinity for maximum throughput (all cores).
 * Use when running multiple independent tasks.
 */
static void mkl_config_set_all_core_affinity(void) {
    setenv("KMP_AFFINITY", "granularity=fine,scatter", 1);
}

/*
 * Disable thread affinity (let OS schedule).
 */
static void mkl_config_disable_affinity(void) {
    setenv("KMP_AFFINITY", "disabled", 1);
}

// ============================================================================
// Instruction Set Configuration
// ============================================================================

/*
 * Force specific instruction set.
 * 
 * @param level  One of: "AVX512", "AVX2", "AVX", "SSE4_2"
 * 
 * Note: 14900KF supports AVX-512 on P-cores but NOT on E-cores.
 * When using all cores, MKL may fall back to AVX2 for compatibility.
 * When using P-cores only, AVX-512 can be used for best performance.
 */
static void mkl_config_set_instructions(const char* level) {
    if (strcmp(level, "AVX512") == 0) {
        setenv("MKL_ENABLE_INSTRUCTIONS", "AVX512", 1);
    } else if (strcmp(level, "AVX2") == 0) {
        setenv("MKL_ENABLE_INSTRUCTIONS", "AVX2", 1);
    } else if (strcmp(level, "AVX") == 0) {
        setenv("MKL_ENABLE_INSTRUCTIONS", "AVX", 1);
    } else if (strcmp(level, "SSE4_2") == 0) {
        setenv("MKL_ENABLE_INSTRUCTIONS", "SSE4_2", 1);
    }
}

// ============================================================================
// Verbose/Debug Mode
// ============================================================================

/*
 * Enable MKL verbose mode for debugging.
 * Prints information about each MKL call.
 */
static void mkl_config_enable_verbose(void) {
    setenv("MKL_VERBOSE", "1", 1);
}

/*
 * Enable memory alignment checking.
 */
static void mkl_config_check_alignment(void) {
    setenv("MKL_CBWR", "AUTO,STRICT", 1);
}

// ============================================================================
// Recommended Setup for 14900KF
// ============================================================================

/*
 * Optimal configuration for Intel 14900KF.
 * 
 * Call at program start, BEFORE any MKL operations.
 * 
 * @param verbose  Print configuration info
 */
static void mkl_config_14900kf(int verbose) {
    // Step 1: Set affinity to P-cores (must be before MKL init)
    mkl_config_set_p_core_affinity();
    
    // Step 2: Force AVX-512 (P-cores support it)
    mkl_config_set_instructions("AVX512");
    
    // Step 3: Initialize MKL with P-cores only
    mkl_config_init_ex(MKL_CONFIG_P_CORES, verbose);
}

#endif // SSA_USE_MKL

#endif // MKL_CONFIG_H
