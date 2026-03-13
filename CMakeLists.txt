cmake_minimum_required(VERSION 3.14)
project(line)

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_compile_options(-faligned-new -march=native -g -O3 -msse2 -mlzcnt -mbmi)

include_directories(core)

add_executable(example examples/example.cpp)
add_executable(generatorvar dataset/generatorvar.cpp)
