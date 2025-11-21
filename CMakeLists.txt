cmake_minimum_required(VERSION 3.10)
project(prir_project)

set(CMAKE_CXX_STANDARD 17)

find_package(MPI REQUIRED)
find_package(OpenMP)

add_executable(log_analyzer_mpi log_analyzer_mpi.cpp)
target_link_libraries(log_analyzer_mpi PRIVATE MPI::MPI_CXX)

add_executable(log_analyzer_openmp log_analyzer_openmp.cpp)
if(OpenMP_CXX_FOUND)
    target_link_libraries(log_analyzer_openmp PRIVATE OpenMP::OpenMP_CXX)
endif()
