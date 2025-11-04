cmake_minimum_required(VERSION 4.0)
project(prir_project)

set(CMAKE_CXX_STANDARD 20)

add_executable(prir_project main.cpp
        log_analyzer_openmp.cpp)
