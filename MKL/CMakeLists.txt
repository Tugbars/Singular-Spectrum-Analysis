cmake_minimum_required(VERSION 3.16)
project(ssa_mkl C)

set(CMAKE_C_STANDARD 11)

option(USE_MKL "Use Intel MKL for optimized FFT and BLAS" ON)

add_executable(ssa_opt_test ssa_opt_test.c)
add_executable(ssa_example ssa_example.c)
add_executable(ssa_mkl_debug ssa_mkl_debug.c)
add_executable(ssa_block_test ssa_block_test.c)
add_executable(ssa_forecast_test test_ssa_forecast.c)
add_executable(mssa_test test_mssa.c)

if(USE_MKL)
    # Force LP64 (32-bit integers) instead of ILP64
    set(MKL_INTERFACE lp64)
    find_package(MKL CONFIG REQUIRED)
    
    target_compile_definitions(ssa_opt_test PRIVATE SSA_USE_MKL _USE_MATH_DEFINES)
    target_link_libraries(ssa_opt_test PRIVATE MKL::MKL)
    
    target_compile_definitions(ssa_example PRIVATE SSA_USE_MKL _USE_MATH_DEFINES)
    target_link_libraries(ssa_example PRIVATE MKL::MKL)
    
    target_compile_definitions(ssa_mkl_debug PRIVATE SSA_USE_MKL _USE_MATH_DEFINES)
    target_link_libraries(ssa_mkl_debug PRIVATE MKL::MKL)
    
    target_compile_definitions(ssa_block_test PRIVATE SSA_USE_MKL _USE_MATH_DEFINES)
    target_link_libraries(ssa_block_test PRIVATE MKL::MKL)
    
    target_compile_definitions(ssa_forecast_test PRIVATE SSA_USE_MKL _USE_MATH_DEFINES)
    target_link_libraries(ssa_forecast_test PRIVATE MKL::MKL)
    
    target_compile_definitions(mssa_test PRIVATE SSA_USE_MKL _USE_MATH_DEFINES)
    target_link_libraries(mssa_test PRIVATE MKL::MKL)
    
    message(STATUS "MKL found: ${MKL_ROOT}")
    message(STATUS "MKL interface: ${MKL_INTERFACE}")
else()
    if(NOT MSVC)
        target_link_libraries(ssa_opt_test PRIVATE m)
        target_link_libraries(ssa_example PRIVATE m)
        target_link_libraries(ssa_mkl_debug PRIVATE m)
        target_link_libraries(ssa_block_test PRIVATE m)
        target_link_libraries(ssa_forecast_test PRIVATE m)
        target_link_libraries(mssa_test PRIVATE m)
    endif()
    message(STATUS "Building without MKL (built-in FFT)")
endif()

if(MSVC)
    target_compile_options(ssa_opt_test PRIVATE $<$<CONFIG:Release>:/O2 /fp:fast> /W3)
    target_compile_options(ssa_example PRIVATE $<$<CONFIG:Release>:/O2 /fp:fast> /W3)
    target_compile_options(ssa_mkl_debug PRIVATE $<$<CONFIG:Release>:/O2 /fp:fast> /W3)
    target_compile_options(ssa_block_test PRIVATE $<$<CONFIG:Release>:/O2 /fp:fast> /W3)
    target_compile_options(ssa_forecast_test PRIVATE $<$<CONFIG:Release>:/O2 /fp:fast> /W3)
    target_compile_options(mssa_test PRIVATE $<$<CONFIG:Release>:/O2 /fp:fast> /W3)
endif()