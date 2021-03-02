cmake_minimum_required(VERSION 3.9.0 FATAL_ERROR)
cmake_policy(VERSION 3.9.0)

# Demo CMake project to test using the package config files

# load compilers
project(TestCMakeConsumer LANGUAGES C Fortran)

# find QUICK
find_package(QUICK CONFIG REQUIRED)

# build API test
add_executable(test-api quick_api_test.F90)
target_link_libraries(test-api QUICK::libquick)

add_executable(test-api.mpi quick_api_test.F90)
target_compile_definitions(test-api.mpi PRIVATE MPIV)
target_link_libraries(test-api.mpi QUICK::libquick_mpi)
