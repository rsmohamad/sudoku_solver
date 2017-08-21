cmake_minimum_required(VERSION 3.5)
project(sudoku_solver)

find_package(OpenCV REQUIRED)
find_package(cmake_modules REQUIRED)
find_package( PkgConfig REQUIRED)

pkg_search_module( TESSERACT REQUIRED tesseract )
pkg_search_module( LEPTONICA REQUIRED lept )
include_directories( ${TESSERACT_INCLUDE_DIRS} )
include_directories( ${LEPTONICA_INCLUDE_DIRS} )
link_directories( ${TESSERACT_LIBRARY_DIRS} )
link_directories( ${LEPTONICA_LIBRARY_DIRS} )
set(CMAKE_CXX_STANDARD 11)
set(SOURCE_FILES main.cpp)

add_executable(sudoku_solver ${SOURCE_FILES})
target_link_libraries( sudoku_solver ${OpenCV_LIBS} ${TESSERACT_LIBRARIES} ${LEPTONICA_LIBRARIES})