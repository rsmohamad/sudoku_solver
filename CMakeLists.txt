cmake_minimum_required(VERSION 3.5)
project(sudoku_solver)

find_package(OpenCV REQUIRED)

set(CMAKE_CXX_STANDARD 11)
set(SOURCE_FILES main.cpp)

add_executable(sudoku_solver ${SOURCE_FILES})
target_link_libraries( sudoku_solver ${OpenCV_LIBS} )