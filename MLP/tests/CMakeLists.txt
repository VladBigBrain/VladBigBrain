cmake_minimum_required(VERSION 3.5)
find_program(CMAKE_CXX_COMPILER NAMES g++)
find_program(CMAKE_C_COMPILER NAMES gcc)
project(Transactions VERSION 0.1 LANGUAGES CXX)

if(NOT CMAKE_CXX_COMPILER)
    message(FATAL_ERROR "g++ compiler not found. Please make sure it is installed or specify the compiler manually.")
endif()

set(CMAKE_CXX_COMPILER ${CMAKE_CXX_COMPILER})
#Flags
set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Werror -Wextra -pedantic")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -w --coverage")
#set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} --coverage")
#Flags
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")

#Additional Content

include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG release-1.12.1
)
FetchContent_MakeAvailable(googletest)

FetchContent_Declare(
  Eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG master
  GIT_SHALLOW TRUE
  GIT_PROGRESS TRUE)
set(EIGEN_BUILD_DOC OFF)

set(BUILD_TESTING OFF)
set(EIGEN_BUILD_PKGCONFIG OFF)
set( OFF)
FetchContent_MakeAvailable(Eigen)

enable_testing()

#commonSource
set(commonSource)

#MODELTEST
set(modelsource model_tests.cc ../src/model/layer.cc ../src/model/neural_network.cc ../src/model/neuron.cc)
add_executable(Model ${modelsource} ${commonSource})
target_link_libraries(Model PRIVATE  GTest::gtest_main Eigen3::Eigen)
target_include_directories(Model PRIVATE ../src/model ../src/libraries)

add_custom_target(build_model
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target Model
    COMMAND Model
    DEPENDS Model)

