cmake_minimum_required(VERSION 3.5)
project(Tests VERSION 0.1 LANGUAGES CXX)
set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(GTest REQUIRED)
find_package(GSL REQUIRED)
include_directories(${GSL_INCLUDE_DIRS})
# ------------------ADDITIONAL LIBRARIES--------------------------------------
# include(FetchContent)
# FetchContent_Declare(
#   googletest
#   GIT_REPOSITORY https://github.com/google/googletest.git
#   GIT_TAG release-1.12.1
# )
# set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
# FetchContent_MakeAvailable(googletest)
# FetchContent_Declare(GSL
#     GIT_REPOSITORY "https://github.com/microsoft/GSL"
#     GIT_TAG "v4.0.0"
#     GIT_SHALLOW ON
# )
# FetchContent_MakeAvailable(GSL)
add_executable(
  Tests
  tests.cc
)
enable_testing()
target_link_libraries(Tests
  GTest::gtest_main
  ${GSL_LIBRARIES}
)

include(GoogleTest)
gtest_discover_tests(Tests)
