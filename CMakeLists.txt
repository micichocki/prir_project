cmake_minimum_required(VERSION 3.10)
project(prir_project)

set(CMAKE_CXX_STANDARD 17)

find_package(MPI REQUIRED)
find_package(OpenMP)

add_executable(log_analyzer_hybrid 
    log_analyzer_hybrid.cpp 
    Serializer.cpp 
)

target_link_libraries(log_analyzer_hybrid 
    PRIVATE MPI::MPI_CXX 
    PRIVATE OpenMP::OpenMP_CXX
)
