cmake_minimum_required(VERSION 3.16)
project(ssa_mkl C)

set(CMAKE_C_STANDARD 11)

option(USE_MKL "Use Intel MKL for optimized FFT and BLAS" ON)

add_executable(ssa_opt_test ssa_opt_test.c)
add_executable(ssa_example ssa_example.c)

if(USE_MKL)
    find_package(MKL CONFIG REQUIRED)
    
    target_compile_definitions(ssa_opt_test PRIVATE SSA_USE_MKL)
    target_link_libraries(ssa_opt_test PRIVATE MKL::MKL)
    
    target_compile_definitions(ssa_example PRIVATE SSA_USE_MKL)
    target_link_libraries(ssa_example PRIVATE MKL::MKL)
    
    message(STATUS "MKL found: ${MKL_ROOT}")
else()
    if(NOT MSVC)
        target_link_libraries(ssa_opt_test PRIVATE m)
        target_link_libraries(ssa_example PRIVATE m)
    endif()
    message(STATUS "Building without MKL (built-in FFT)")
endif()

# Only add optimization flags in Release mode
if(MSVC)
    target_compile_options(ssa_opt_test PRIVATE $<$<CONFIG:Release>:/O2 /fp:fast> /W3)
    target_compile_options(ssa_example PRIVATE $<$<CONFIG:Release>:/O2 /fp:fast> /W3)
else()
    target_compile_options(ssa_opt_test PRIVATE $<$<CONFIG:Release>:-O3 -march=native -ffast-math> -Wall)
    target_compile_options(ssa_example PRIVATE $<$<CONFIG:Release>:-O3 -march=native -ffast-math> -Wall)
endif()