#!/bin/bash
# =============================================================================
# MKL Environment Setup for Intel 14900KF
# =============================================================================
#
# Usage:
#   source setup_mkl_14900kf.sh
#   
# Then compile and run your program:
#   gcc -O3 -DSSA_USE_MKL -I${MKLROOT}/include -o ssa_test ssa_mkl_test.c \
#       -L${MKLROOT}/lib/intel64 -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core \
#       -liomp5 -lpthread -lm
#   ./ssa_test
#
# =============================================================================

# Check if MKL is installed
if [ -f "/opt/intel/oneapi/setvars.sh" ]; then
    echo "Loading Intel oneAPI environment..."
    source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
elif [ -n "$MKLROOT" ]; then
    echo "MKLROOT already set: $MKLROOT"
else
    echo "ERROR: Intel MKL not found!"
    echo ""
    echo "Install MKL:"
    echo "  sudo apt update"
    echo "  sudo apt install intel-oneapi-mkl-devel"
    echo ""
    echo "Or download from:"
    echo "  https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl-download.html"
    return 1 2>/dev/null || exit 1
fi

echo ""
echo "=== Intel 14900KF MKL Configuration ==="
echo ""

# -----------------------------------------------------------------------------
# CPU Topology for 14900KF:
#   P-cores: 8 cores with HT = threads 0-15 (fast, AVX-512)
#   E-cores: 16 cores = threads 16-31 (slower, no AVX-512)
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Threading Configuration
# -----------------------------------------------------------------------------

# Use P-cores only (16 threads from 8 P-cores with hyperthreading)
# This gives best single-threaded and parallel performance for compute
export MKL_NUM_THREADS=16
export OMP_NUM_THREADS=16

# Disable dynamic thread adjustment (more predictable performance)
export MKL_DYNAMIC=FALSE
export OMP_DYNAMIC=FALSE

echo "Threads: MKL_NUM_THREADS=$MKL_NUM_THREADS (P-cores only)"

# -----------------------------------------------------------------------------
# Thread Affinity - Pin to P-cores
# -----------------------------------------------------------------------------

# Method 1: Intel OpenMP specific (recommended)
# granularity=fine: bind to logical processors
# compact: pack threads close together  
# 1,0: start from processor 0, stride 1
export KMP_AFFINITY="granularity=fine,compact,1,0"

# Limit to 8 P-cores with 2 threads each
export KMP_HW_SUBSET="8c,2t"

# Method 2: Alternative - explicit processor list
# export KMP_AFFINITY="explicit,proclist=[0-15]"

# Method 3: Standard OpenMP (less control)
# export OMP_PLACES=cores
# export OMP_PROC_BIND=close

echo "Affinity: KMP_AFFINITY=$KMP_AFFINITY"
echo "Subset: KMP_HW_SUBSET=$KMP_HW_SUBSET"

# -----------------------------------------------------------------------------
# Instruction Set - Force AVX-512 (P-cores support it)
# -----------------------------------------------------------------------------

# P-cores support AVX-512, E-cores don't
# Since we're only using P-cores, we can force AVX-512
export MKL_ENABLE_INSTRUCTIONS=AVX512

echo "Instructions: MKL_ENABLE_INSTRUCTIONS=$MKL_ENABLE_INSTRUCTIONS"

# -----------------------------------------------------------------------------
# Memory Configuration
# -----------------------------------------------------------------------------

# Use huge pages if available (better TLB performance)
export MKL_FAST_MEMORY_LIMIT=0

# Alignment for optimal SIMD performance
export MKL_CBWR=AUTO,STRICT

# -----------------------------------------------------------------------------
# Threading Layer
# -----------------------------------------------------------------------------

# Options: INTEL (default), GNU, TBB, SEQUENTIAL
# INTEL is best for Intel CPUs
export MKL_THREADING_LAYER=INTEL

echo "Threading layer: MKL_THREADING_LAYER=$MKL_THREADING_LAYER"

# -----------------------------------------------------------------------------
# Interface Layer (for 64-bit indexing)
# -----------------------------------------------------------------------------

# LP64: 32-bit integers (default, sufficient for most cases)
# ILP64: 64-bit integers (for arrays > 2^31 elements)
export MKL_INTERFACE_LAYER=LP64

# -----------------------------------------------------------------------------
# Debug/Verbose (uncomment for debugging)
# -----------------------------------------------------------------------------

# export MKL_VERBOSE=1  # Print MKL function calls

# -----------------------------------------------------------------------------
# NUMA Configuration (if multi-socket)
# -----------------------------------------------------------------------------

# For single-socket 14900KF, this isn't needed
# But useful for server systems
# export KMP_TOPOLOGY_METHOD=hwloc
# export KMP_AFFINITY="granularity=fine,compact,1,0"

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------

echo ""
echo "=== Configuration Summary ==="
echo "MKLROOT: $MKLROOT"
echo "Threads: $MKL_NUM_THREADS (P-cores: 0-15)"
echo "Affinity: P-cores only via KMP_HW_SUBSET"
echo "Instructions: AVX-512"
echo "Threading: Intel OpenMP"
echo ""
echo "Ready! Compile with:"
echo "  gcc -O3 -march=native -DSSA_USE_MKL -I\${MKLROOT}/include \\"
echo "      -o ssa_test ssa_mkl_test.c \\"
echo "      -L\${MKLROOT}/lib/intel64 \\"
echo "      -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core \\"
echo "      -liomp5 -lpthread -lm"
echo ""

# -----------------------------------------------------------------------------
# Verify Configuration
# -----------------------------------------------------------------------------

# Quick test to verify MKL is working
if command -v python3 &> /dev/null; then
    echo "Verifying MKL..."
    python3 -c "
import ctypes
try:
    mkl = ctypes.CDLL('libmkl_rt.so')
    print('  MKL runtime: OK')
except:
    print('  MKL runtime: Not found (might still work with static linking)')
" 2>/dev/null
fi

echo "=== Setup Complete ==="
