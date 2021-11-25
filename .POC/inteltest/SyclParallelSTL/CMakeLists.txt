cmake_minimum_required (VERSION 3.2.2)
project (SyclSTL)

enable_testing()

option(PARALLEL_STL_BENCHMARKS "Build the internal benchmarks" OFF)
option(USE_COMPUTECPP "Use ComputeCPP" ON)

message(STATUS " Path to CMAKE source directory: ${CMAKE_SOURCE_DIR} ")
set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/Modules/)
find_package(Threads REQUIRED)

if (USE_COMPUTECPP)
  message(STATUS " Using ComputeCpp CMake")

  message(STATUS " Path to ComputeCpp implementation: ${COMPUTECPP_PACKAGE_ROOT_DIR} ")

  set(CMAKE_CXX_STANDARD 11)

  include(FindOpenCL)
  include(FindComputeCpp)

  add_definitions(-DSYCL_PSTL_USE_OLD_ALGO)
  set(COMPUTECPP_DEVICE_COMPILER_FLAGS "${COMPUTECPP_DEVICE_COMPILER_FLAGS} -DSYCL_PSTL_USE_OLD_ALGO")

  include_directories("${COMPUTECPP_INCLUDE_DIRECTORY}")


else()
  message(STATUS " Using triSYCL CMake")
  include(FindTriSYCL)
endif()


# PSTL specific
include_directories("include")

add_subdirectory (src)
add_subdirectory (examples)
add_subdirectory (tests)

if (PARALLEL_STL_BENCHMARKS)
  add_subdirectory (benchmarks)
endif()
