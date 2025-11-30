/*
 * MKL Configuration for Intel Hybrid CPUs (14th Gen Raptor Lake, etc.)
 *
 * Cross-platform: Windows (MSVC, MinGW) and Linux (GCC, Clang)
 *
 * Intel 14900KF specifics:
 *   - 8 P-cores (Performance) with HT = 16 threads
 *   - 16 E-cores (Efficient) = 16 threads
 *   - Total: 32 threads, but NOT equal performance
 *   - P-cores: ~2x faster than E-cores for compute workloads
 *   - AVX-512 is DISABLED on consumer chips (use AVX2)
 *
 * Usage:
 *   #include "mkl_config.h"
 *
 *   int main() {
 *       mkl_config_init();  // Call once at startup
 *       // ... your code ...
 *   }
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
// Platform-specific includes and macros
// ============================================================================

#ifdef _WIN32
#include <intrin.h>
#define MKL_SETENV(name, value) _putenv_s(name, value)
#else
#define MKL_SETENV(name, value) setenv(name, value, 1)
#endif

// ============================================================================
// CPU Detection
// ============================================================================

typedef struct
{
    int num_p_cores;        // Performance cores
    int num_e_cores;        // Efficiency cores
    int threads_per_p_core; // Usually 2 (hyperthreading)
    int threads_per_e_core; // Usually 1
    int total_threads;
    int is_hybrid; // 1 if P+E cores detected
    int has_avx512;
    int has_avx2;
    char cpu_name[64];
} CPUInfo;

static inline void mkl_cpuid(int info[4], int leaf, int subleaf)
{
#if defined(_MSC_VER) || defined(_WIN32)
    __cpuidex(info, leaf, subleaf);
#else
    __asm__ __volatile__(
        "cpuid"
        : "=a"(info[0]), "=b"(info[1]), "=c"(info[2]), "=d"(info[3])
        : "a"(leaf), "c"(subleaf));
#endif
}

static CPUInfo detect_cpu(void)
{
    CPUInfo info = {0};
    int regs[4];

    // Get CPU brand string
    mkl_cpuid(regs, 0x80000002, 0);
    memcpy(info.cpu_name, regs, 16);
    mkl_cpuid(regs, 0x80000003, 0);
    memcpy(info.cpu_name + 16, regs, 16);
    mkl_cpuid(regs, 0x80000004, 0);
    memcpy(info.cpu_name + 32, regs, 16);

    // Check for AVX-512
    mkl_cpuid(regs, 7, 0);
    info.has_avx512 = (regs[1] >> 16) & 1; // AVX-512F
    info.has_avx2 = (regs[1] >> 5) & 1;    // AVX2

    // Detect hybrid architecture (12th gen+)
    mkl_cpuid(regs, 0, 0);
    int max_leaf = regs[0];

    if (max_leaf >= 0x1A)
    {
        mkl_cpuid(regs, 0x1A, 0);
        // core_type: 0x20 = Atom/E-core, 0x40 = Core/P-core
        info.is_hybrid = 1;

        // For 14900KF: 8P + 16E
        if (strstr(info.cpu_name, "14900") || strstr(info.cpu_name, "14700"))
        {
            info.num_p_cores = 8;
            info.num_e_cores = 16;
            info.threads_per_p_core = 2;
            info.threads_per_e_core = 1;
        }
        else if (strstr(info.cpu_name, "13900") || strstr(info.cpu_name, "13700"))
        {
            info.num_p_cores = 8;
            info.num_e_cores = 16;
            info.threads_per_p_core = 2;
            info.threads_per_e_core = 1;
        }
        else if (strstr(info.cpu_name, "12900") || strstr(info.cpu_name, "12700"))
        {
            info.num_p_cores = 8;
            info.num_e_cores = 8;
            info.threads_per_p_core = 2;
            info.threads_per_e_core = 1;
        }
        else
        {
            // Generic hybrid assumption
            info.num_p_cores = 6;
            info.num_e_cores = 8;
            info.threads_per_p_core = 2;
            info.threads_per_e_core = 1;
        }
    }
    else
    {
        // Non-hybrid (older Intel or AMD)
        info.is_hybrid = 0;
        mkl_cpuid(regs, 1, 0);
        int logical_cores = (regs[1] >> 16) & 0xFF;
        info.num_p_cores = logical_cores / 2; // Assume HT
        info.threads_per_p_core = 2;
    }

    info.total_threads = info.num_p_cores * info.threads_per_p_core +
                         info.num_e_cores * info.threads_per_e_core;

    return info;
}

// ============================================================================
// MKL Configuration
// ============================================================================

typedef enum
{
    MKL_CONFIG_AUTO,       // Let MKL decide (may use all cores)
    MKL_CONFIG_P_CORES,    // Use P-cores only (recommended for compute)
    MKL_CONFIG_ALL_CORES,  // Use all cores
    MKL_CONFIG_SEQUENTIAL, // Single-threaded (for small problems)
    MKL_CONFIG_CUSTOM      // Use MKL_NUM_THREADS environment variable
} MKLConfigMode;

typedef struct
{
    MKLConfigMode mode;
    int num_threads;
    int verbose;
    CPUInfo cpu;
} MKLConfig;

static MKLConfig g_mkl_config = {0};

/*
 * Initialize MKL with optimal settings for the detected CPU.
 */
static int mkl_config_init_ex(MKLConfigMode mode, int verbose)
{
    g_mkl_config.cpu = detect_cpu();
    g_mkl_config.mode = mode;
    g_mkl_config.verbose = verbose;

    CPUInfo *cpu = &g_mkl_config.cpu;

    if (verbose)
    {
        printf("=== MKL Configuration ===\n");
        printf("CPU: %s\n", cpu->cpu_name);
        printf("Hybrid: %s\n", cpu->is_hybrid ? "Yes" : "No");
        if (cpu->is_hybrid)
        {
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
    switch (mode)
    {
    case MKL_CONFIG_SEQUENTIAL:
        num_threads = 1;
        break;

    case MKL_CONFIG_P_CORES:
        num_threads = cpu->num_p_cores * cpu->threads_per_p_core;
        if (num_threads == 0)
            num_threads = cpu->total_threads / 2;
        break;

    case MKL_CONFIG_ALL_CORES:
        num_threads = cpu->total_threads;
        break;

    case MKL_CONFIG_CUSTOM:
    {
        char *env = getenv("MKL_NUM_THREADS");
        if (env)
        {
            num_threads = atoi(env);
        }
        else
        {
            num_threads = cpu->num_p_cores * cpu->threads_per_p_core;
        }
    }
    break;

    case MKL_CONFIG_AUTO:
    default:
        if (cpu->is_hybrid)
        {
            num_threads = cpu->num_p_cores * cpu->threads_per_p_core;
        }
        else
        {
            num_threads = cpu->total_threads;
        }
        break;
    }

    g_mkl_config.num_threads = num_threads;

    // Configure MKL
    mkl_set_num_threads(num_threads);
    mkl_set_dynamic(0);

#ifdef _OPENMP
    omp_set_num_threads(num_threads);
#endif

    if (verbose)
    {
        printf("MKL threads: %d\n", num_threads);
        printf("MKL dynamic: disabled\n");

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
static int mkl_config_init(void)
{
    return mkl_config_init_ex(MKL_CONFIG_AUTO, 0);
}

/*
 * Initialize MKL with verbose output.
 */
static int mkl_config_init_verbose(void)
{
    return mkl_config_init_ex(MKL_CONFIG_AUTO, 1);
}

/*
 * Get optimal thread count for a given problem size.
 */
static int mkl_config_optimal_threads(int n)
{
    CPUInfo *cpu = &g_mkl_config.cpu;
    int max_threads = cpu->num_p_cores * cpu->threads_per_p_core;
    if (max_threads == 0)
        max_threads = 8;

    if (n < 1024)
        return 1;
    if (n < 4096)
        return 2;
    if (n < 16384)
        return 4;
    if (n < 65536)
        return 8;
    return max_threads;
}

/*
 * Temporarily set thread count for a specific operation.
 */
static void mkl_config_set_threads(int n)
{
    mkl_set_num_threads(n);
#ifdef _OPENMP
    omp_set_num_threads(n);
#endif
}

/*
 * Restore default thread count.
 */
static void mkl_config_restore_threads(void)
{
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
 * IMPORTANT: Call BEFORE mkl_config_init() or at program start.
 */
static void mkl_config_set_p_core_affinity(void)
{
    MKL_SETENV("KMP_AFFINITY", "granularity=fine,compact,1,0");
    MKL_SETENV("KMP_HW_SUBSET", "8c,2t");
}

/*
 * Set thread affinity for maximum throughput (all cores).
 */
static void mkl_config_set_all_core_affinity(void)
{
    MKL_SETENV("KMP_AFFINITY", "granularity=fine,scatter");
}

/*
 * Disable thread affinity (let OS schedule).
 */
static void mkl_config_disable_affinity(void)
{
    MKL_SETENV("KMP_AFFINITY", "disabled");
}

// ============================================================================
// Instruction Set Configuration
// ============================================================================

/*
 * Force specific instruction set.
 * Note: 14900KF has AVX-512 DISABLED by Intel. Use AVX2.
 */
static void mkl_config_set_instructions(const char *level)
{
    if (strcmp(level, "AVX512") == 0)
    {
        MKL_SETENV("MKL_ENABLE_INSTRUCTIONS", "AVX512");
    }
    else if (strcmp(level, "AVX2") == 0)
    {
        MKL_SETENV("MKL_ENABLE_INSTRUCTIONS", "AVX2");
    }
    else if (strcmp(level, "AVX") == 0)
    {
        MKL_SETENV("MKL_ENABLE_INSTRUCTIONS", "AVX");
    }
    else if (strcmp(level, "SSE4_2") == 0)
    {
        MKL_SETENV("MKL_ENABLE_INSTRUCTIONS", "SSE4_2");
    }
}

// ============================================================================
// Verbose/Debug Mode
// ============================================================================

static void mkl_config_enable_verbose(void)
{
    MKL_SETENV("MKL_VERBOSE", "1");
}

static void mkl_config_check_alignment(void)
{
    MKL_SETENV("MKL_CBWR", "AUTO,STRICT");
}

// ============================================================================
// Recommended Setup for 14900KF
// ============================================================================

/*
 * Optimal configuration for Intel 14900KF.
 * Call at program start, BEFORE any MKL operations.
 *
 * Note: Uses AVX2 because AVX-512 is disabled on consumer Raptor Lake.
 */
static void mkl_config_14900kf(int verbose)
{
    mkl_config_set_p_core_affinity();
    mkl_config_set_instructions("AVX2"); // AVX-512 disabled on 14900KF!
    mkl_config_init_ex(MKL_CONFIG_P_CORES, verbose);
}

#endif // SSA_USE_MKL

#endif // MKL_CONFIG_H