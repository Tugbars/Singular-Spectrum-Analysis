cmake_minimum_required(VERSION 3.16)
project(ssa_mkl C)

set(CMAKE_C_STANDARD 11)

option(USE_MKL "Use Intel MKL for optimized FFT and BLAS" ON)
option(USE_FLOAT "Use single precision (float) instead of double" OFF)

# ============================================================================
# Compiler Detection
# ============================================================================
set(COMPILER_IS_ICX FALSE)
set(COMPILER_IS_MSVC FALSE)
set(COMPILER_IS_GCC FALSE)
set(COMPILER_IS_CLANG FALSE)

if(CMAKE_C_COMPILER_ID MATCHES "IntelLLVM")
    set(COMPILER_IS_ICX TRUE)
    message(STATUS "Detected Intel ICX compiler: ${CMAKE_C_COMPILER_VERSION}")
elseif(MSVC)
    set(COMPILER_IS_MSVC TRUE)
    message(STATUS "Detected MSVC compiler: ${MSVC_VERSION}")
elseif(CMAKE_C_COMPILER_ID MATCHES "GNU")
    set(COMPILER_IS_GCC TRUE)
    message(STATUS "Detected GCC compiler: ${CMAKE_C_COMPILER_VERSION}")
elseif(CMAKE_C_COMPILER_ID MATCHES "Clang")
    set(COMPILER_IS_CLANG TRUE)
    message(STATUS "Detected Clang compiler: ${CMAKE_C_COMPILER_VERSION}")
endif()

# ============================================================================
# Test Executables
# ============================================================================
add_executable(ssa_opt_test test/ssa_opt_test.c)
add_executable(ssa_block_test test/ssa_block_test.c)
add_executable(ssa_forecast_test test/test_ssa_forecast.c)
add_executable(mssa_test test/test_mssa.c)
add_executable(wcorr_parallel_bench test/wcorr_parallel_bench.c)
add_executable(ssa_vforecast_test test/ssa_vforecast_test.c)
add_executable(ssa_cadzow_test test/ssa_cadzow_test.c)

set(TEST_TARGETS
    ssa_opt_test
    ssa_block_test
    ssa_forecast_test
    mssa_test
    wcorr_parallel_bench
    ssa_vforecast_test
    ssa_cadzow_test
)

# Include directories
foreach(target ${TEST_TARGETS})
    target_include_directories(${target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
endforeach()

# ============================================================================
# Float precision option
# ============================================================================
if(USE_FLOAT)
    foreach(target ${TEST_TARGETS})
        target_compile_definitions(${target} PRIVATE SSA_USE_FLOAT)
    endforeach()
    message(STATUS "Precision: FLOAT (32-bit)")
else()
    message(STATUS "Precision: DOUBLE (64-bit)")
endif()

# ============================================================================
# MKL Configuration
# ============================================================================
if(USE_MKL)
    # Force LP64 (32-bit integers) instead of ILP64
    set(MKL_INTERFACE lp64)
    find_package(MKL CONFIG REQUIRED)
    
    foreach(target ${TEST_TARGETS})
        target_compile_definitions(${target} PRIVATE SSA_USE_MKL _USE_MATH_DEFINES)
        target_link_libraries(${target} PRIVATE MKL::MKL)
    endforeach()
    
    message(STATUS "MKL found: ${MKL_ROOT}")
    message(STATUS "MKL interface: ${MKL_INTERFACE}")
else()
    if(NOT COMPILER_IS_MSVC)
        foreach(target ${TEST_TARGETS})
            target_link_libraries(${target} PRIVATE m)
        endforeach()
    endif()
    message(STATUS "Building without MKL (built-in FFT)")
endif()

# ============================================================================
# OpenMP Support
# ============================================================================
find_package(OpenMP)
if(OpenMP_C_FOUND)
    foreach(target ${TEST_TARGETS})
        target_link_libraries(${target} PRIVATE OpenMP::OpenMP_C)
    endforeach()
    message(STATUS "OpenMP found: ${OpenMP_C_VERSION}")
endif()

# ============================================================================
# Compiler-Specific Optimization Flags
# ============================================================================
if(COMPILER_IS_ICX)
    # Intel ICX: Maximum performance for Intel CPUs
    # -xHost: Optimize for current CPU (use -xcore-avx2 for portable AVX2)
    # -qopt-zmm-usage=high: Aggressive AVX-512 usage (if available)
    # -ipo: Interprocedural optimization (whole-program)
    # -ffast-math: Aggressive FP optimizations
    # -qopenmp: Intel OpenMP runtime (lower overhead than libgomp)
    set(ICX_RELEASE_FLAGS
        -O3
        -xHost
        -ipo
        -ffast-math
        -fno-strict-aliasing
        -qopt-report=3
        -qopt-report-phase=vec
    )
    
    # For portable builds (works on any AVX2 CPU):
    # set(ICX_RELEASE_FLAGS -O3 -xcore-avx2 -ipo -ffast-math)
    
    foreach(target ${TEST_TARGETS})
        target_compile_options(${target} PRIVATE 
            $<$<CONFIG:Release>:${ICX_RELEASE_FLAGS}>
            $<$<CONFIG:Debug>:-g -O0>
        )
        # IPO requires special linker flags
        set_target_properties(${target} PROPERTIES
            INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE
        )
    endforeach()
    
    message(STATUS "ICX optimization flags: ${ICX_RELEASE_FLAGS}")

elseif(COMPILER_IS_MSVC)
    # MSVC: Standard optimization
    foreach(target ${TEST_TARGETS})
        target_compile_options(${target} PRIVATE 
            $<$<CONFIG:Release>:/O2 /fp:fast /arch:AVX2 /GL>
            /W3
        )
        # Whole program optimization
        set_target_properties(${target} PROPERTIES
            LINK_FLAGS_RELEASE "/LTCG"
        )
    endforeach()

elseif(COMPILER_IS_GCC)
    # GCC: Good optimization for Intel/AMD
    set(GCC_RELEASE_FLAGS
        -O3
        -march=native
        -ffast-math
        -funroll-loops
        -ftree-vectorize
        -flto
    )
    foreach(target ${TEST_TARGETS})
        target_compile_options(${target} PRIVATE
            $<$<CONFIG:Release>:${GCC_RELEASE_FLAGS}>
            $<$<CONFIG:Debug>:-g -O0>
        )
        set_target_properties(${target} PROPERTIES
            INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE
        )
    endforeach()

elseif(COMPILER_IS_CLANG)
    # Clang: Similar to GCC
    set(CLANG_RELEASE_FLAGS
        -O3
        -march=native
        -ffast-math
        -funroll-loops
        -flto
    )
    foreach(target ${TEST_TARGETS})
        target_compile_options(${target} PRIVATE
            $<$<CONFIG:Release>:${CLANG_RELEASE_FLAGS}>
            $<$<CONFIG:Debug>:-g -O0>
        )
        set_target_properties(${target} PROPERTIES
            INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE
        )
    endforeach()
endif()

# ============================================================================
# Python shared library
# ============================================================================
if(WIN32)
    add_library(ssa SHARED py/ssa_wrapper.c py/ssa.def)
else()
    add_library(ssa SHARED py/ssa_wrapper.c)
endif()

target_include_directories(ssa PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(ssa PRIVATE SSA_OPT_IMPLEMENTATION SSA_USE_MKL _USE_MATH_DEFINES)

if(USE_FLOAT)
    target_compile_definitions(ssa PRIVATE SSA_USE_FLOAT)
endif()

if(USE_MKL)
    target_link_libraries(ssa PRIVATE MKL::MKL)
else()
    if(NOT COMPILER_IS_MSVC)
        target_link_libraries(ssa PRIVATE m)
    endif()
endif()

if(OpenMP_C_FOUND)
    target_link_libraries(ssa PRIVATE OpenMP::OpenMP_C)
endif()

# Apply same compiler flags to shared library
if(COMPILER_IS_ICX)
    target_compile_options(ssa PRIVATE 
        $<$<CONFIG:Release>:${ICX_RELEASE_FLAGS}>
    )
    set_target_properties(ssa PROPERTIES
        INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE
    )
elseif(COMPILER_IS_MSVC)
    target_compile_options(ssa PRIVATE 
        $<$<CONFIG:Release>:/O2 /fp:fast /arch:AVX2 /GL>
        /W3
    )
elseif(COMPILER_IS_GCC)
    target_compile_options(ssa PRIVATE
        $<$<CONFIG:Release>:${GCC_RELEASE_FLAGS}>
    )
    set_target_properties(ssa PROPERTIES
        INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE
    )
elseif(COMPILER_IS_CLANG)
    target_compile_options(ssa PRIVATE
        $<$<CONFIG:Release>:${CLANG_RELEASE_FLAGS}>
    )
    set_target_properties(ssa PROPERTIES
        INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE
    )
endif()

# ============================================================================
# Build Instructions
# ============================================================================
message(STATUS "")
message(STATUS "=== Build Instructions ===")
if(COMPILER_IS_ICX)
    message(STATUS "ICX detected. For best results:")
    message(STATUS "  cmake -DCMAKE_C_COMPILER=icx -DCMAKE_BUILD_TYPE=Release ..")
    message(STATUS "  cmake --build . --config Release")
elseif(WIN32)
    message(STATUS "Windows detected. To use ICX:")
    message(STATUS "  1. Open Intel oneAPI command prompt")
    message(STATUS "  2. cmake -G Ninja -DCMAKE_C_COMPILER=icx -DCMAKE_BUILD_TYPE=Release ..")
    message(STATUS "  3. cmake --build .")
endif()
message(STATUS "")