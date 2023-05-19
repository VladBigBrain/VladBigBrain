cmake_minimum_required(VERSION 3.5)
project(Transactions VERSION 0.1 LANGUAGES CXX)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(GTest REQUIRED)
include(FetchContent)
# FetchContent_Declare(
#   googletest
#   GIT_REPOSITORY https://github.com/google/googletest.git
#   GIT_TAG release-1.12.1
# )

# set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
# FetchContent_MakeAvailable(googletest)


add_executable(
  Transactions
  tests.cc
)

enable_testing()
# target_include_directories(Transactions PRIVATE ../src/Common ../src/HashTable ../src/TreeBalance)
# target_link_directories(Transactions PRIVATE  ../src/Common ../src/HashTable ../src/TreeBalance)

target_link_libraries(Transactions
  GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(Transactions)
