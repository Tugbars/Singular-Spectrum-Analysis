cmake_minimum_required(VERSION 3.16)
project(singular_spectrum_analysis C)

option(USE_MKL "Use Intel MKL for optimized FFT and BLAS" ON)

# Add MKL subdirectory
add_subdirectory(MKL)