cmake_minimum_required(VERSION 3.28)
project(multilayer_perceptron)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/binaries/${CMAKE_BUILD_TYPE})

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(SOURCES main.cpp
    Net/MultiLayerPerceptron.cpp
    Net/MultiLayerPerceptron.h)

add_executable(multilayer_perceptron ${SOURCES})