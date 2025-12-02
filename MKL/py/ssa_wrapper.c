/*
 * ============================================================================
 * SSA Shared Library Wrapper with MKL Configuration
 * ============================================================================
 *
 * Thin wrapper to compile ssa_opt.h as a shared library for Python ctypes.
 * Includes MKL configuration functions for optimal performance.
 *
 * BUILD (Linux):
 *   source /opt/intel/oneapi/setvars.sh
 *   gcc -shared -fPIC -O3 -o libssa.so ssa_wrapper.c \
 *       -DSSA_OPT_IMPLEMENTATION -DSSA_USE_MKL \
 *       -I${MKLROOT}/include -L${MKLROOT}/lib/intel64 \
 *       -lmkl_rt -lm
 *
 * BUILD (Windows with CMake):
 *   cmake --build . --config Release
 *
 * ============================================================================
 */

#define _USE_MATH_DEFINES
#define SSA_OPT_IMPLEMENTATION
#include "ssa_opt.h"
#include "mkl_config.h"

/* All SSA functions are already exported via the header. 
 * Below we add MKL configuration exports. */

#ifdef _WIN32
#define DLL_EXPORT __declspec(dllexport)
#else
#define DLL_EXPORT __attribute__((visibility("default")))
#endif

/* ============================================================================
 * MKL Configuration Exports
 * ============================================================================ */

/**
 * Initialize MKL with full SSA-optimized settings.
 * Call once at program/module startup.
 *
 * @param verbose  1 = print configuration details, 0 = silent
 * @return 0 on success
 */
DLL_EXPORT int ssa_mkl_init(int verbose)
{
    mkl_config_ssa_full(verbose);
    return 0;
}

/**
 * Initialize MKL specifically for Intel 14900KF (or similar hybrid CPU).
 * Sets P-core affinity and AVX2 instructions.
 *
 * @param verbose  1 = print configuration details, 0 = silent
 * @return 0 on success
 */
DLL_EXPORT int ssa_mkl_init_14900kf(int verbose)
{
    mkl_config_14900kf(verbose);
    return 0;
}

/**
 * Initialize MKL with generic settings (safe for any CPU).
 *
 * @param verbose  1 = print configuration details, 0 = silent
 * @return 0 on success
 */
DLL_EXPORT int ssa_mkl_init_generic(int verbose)
{
    mkl_config_generic(verbose);
    return 0;
}

/**
 * Set MKL thread count manually.
 *
 * @param n  Number of threads (0 = auto)
 */
DLL_EXPORT void ssa_mkl_set_threads(int n)
{
    if (n > 0) {
        mkl_set_num_threads(n);
    }
}

/**
 * Get current MKL thread count.
 *
 * @return Number of threads MKL is using
 */
DLL_EXPORT int ssa_mkl_get_threads(void)
{
    return mkl_get_max_threads();
}

/**
 * Get CPU info detected by mkl_config.
 *
 * @param p_cores      Output: number of P-cores (can be NULL)
 * @param e_cores      Output: number of E-cores (can be NULL)
 * @param is_hybrid    Output: 1 if hybrid CPU (can be NULL)
 * @param has_avx512   Output: 1 if AVX-512 available (can be NULL)
 */
DLL_EXPORT void ssa_mkl_get_cpu_info(int *p_cores, int *e_cores, 
                                      int *is_hybrid, int *has_avx512)
{
    CPUInfo cpu = detect_cpu();
    if (p_cores) *p_cores = cpu.num_p_cores;
    if (e_cores) *e_cores = cpu.num_e_cores;
    if (is_hybrid) *is_hybrid = cpu.is_hybrid;
    if (has_avx512) *has_avx512 = cpu.has_avx512;
}

/**
 * Set thread affinity to P-cores only.
 * Must be called BEFORE ssa_mkl_init() or any MKL operations.
 */
DLL_EXPORT void ssa_mkl_set_p_core_affinity(void)
{
    mkl_config_set_p_core_affinity();
}

/**
 * Disable thread affinity (let OS schedule).
 */
DLL_EXPORT void ssa_mkl_disable_affinity(void)
{
    mkl_config_disable_affinity();
}