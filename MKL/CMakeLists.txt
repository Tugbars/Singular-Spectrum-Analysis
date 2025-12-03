cmake_minimum_required(VERSION 3.16)
project(ssa_mkl C)

set(CMAKE_C_STANDARD 11)

option(USE_MKL "Use Intel MKL for optimized FFT and BLAS" ON)

# Tests are in test/ subdirectory
add_executable(ssa_opt_test test/ssa_opt_test.c)
add_executable(ssa_block_test test/ssa_block_test.c)
add_executable(ssa_forecast_test test/test_ssa_forecast.c)
add_executable(mssa_test test/test_mssa.c)
add_executable(wcorr_parallel_bench test/wcorr_parallel_bench.c)
add_executable(ssa_vforecast_test test/ssa_vforecast_test.c)
add_executable(ssa_cadzow_test test/ssa_cadzow_test.c)


# All test executables
set(TEST_TARGETS
    ssa_opt_test
    ssa_block_test
    ssa_forecast_test
    mssa_test
    wcorr_parallel_bench
    ssa_vforecast_test
    ssa_cadzow_test
)

# Include MKL directory so tests can find ssa_opt.h and mkl_config.h
foreach(target ${TEST_TARGETS})
    target_include_directories(${target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
endforeach()

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
    if(NOT MSVC)
        foreach(target ${TEST_TARGETS})
            target_link_libraries(${target} PRIVATE m)
        endforeach()
    endif()
    message(STATUS "Building without MKL (built-in FFT)")
endif()

# OpenMP support
find_package(OpenMP)
if(OpenMP_C_FOUND)
    foreach(target ${TEST_TARGETS})
        target_link_libraries(${target} PRIVATE OpenMP::OpenMP_C)
    endforeach()
    message(STATUS "OpenMP found: ${OpenMP_C_VERSION}")
endif()

if(MSVC)
    foreach(target ${TEST_TARGETS})
        target_compile_options(${target} PRIVATE $<$<CONFIG:Release>:/O2 /fp:fast /arch:AVX2> /W3)
    endforeach()
endif()
# ============================================================================
# Python shared library
# ============================================================================
if(WIN32)
    # Windows: use .def file for exports
    add_library(ssa SHARED py/ssa_wrapper.c py/ssa.def)
else()
    add_library(ssa SHARED py/ssa_wrapper.c)
endif()

target_include_directories(ssa PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(ssa PRIVATE SSA_OPT_IMPLEMENTATION SSA_USE_MKL _USE_MATH_DEFINES)

if(USE_MKL)
    target_link_libraries(ssa PRIVATE MKL::MKL)
else()
    if(NOT MSVC)
        target_link_libraries(ssa PRIVATE m)
    endif()
endif()

if(MSVC)
    target_compile_options(ssa PRIVATE $<$<CONFIG:Release>:/O2 /fp:fast /arch:AVX2> /W3)
endif()