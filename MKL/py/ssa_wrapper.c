/*
 * ============================================================================
 * SSA Shared Library Wrapper
 * ============================================================================
 *
 * Thin wrapper to compile ssa_opt.h as a shared library for Python ctypes.
 *
 * BUILD (Linux):
 *   source /opt/intel/oneapi/setvars.sh
 *   gcc -shared -fPIC -O3 -o libssa.so ssa_wrapper.c \
 *       -DSSA_OPT_IMPLEMENTATION -DSSA_USE_MKL \
 *       -I${MKLROOT}/include -L${MKLROOT}/lib/intel64 \
 *       -lmkl_rt -lm
 *
 * BUILD (Windows):
 *   cl /LD /O2 /DSSA_OPT_IMPLEMENTATION /DSSA_USE_MKL ssa_wrapper.c ^
 *      /I"%MKLROOT%\include" /link /LIBPATH:"%MKLROOT%\lib" mkl_rt.lib
 *
 * ============================================================================
 */

#define SSA_OPT_IMPLEMENTATION
#include "ssa_opt.h"

/* All functions are already exported via the header. This file just triggers
 * the implementation inclusion and provides a compilation unit for the
 * shared library. */
