cmake_minimum_required(VERSION 2.6)
project(meshfix)

SET(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
option(MESHFIX_VERBOSE "Turn on verbose messages" OFF)
if(MESHFIX_VERBOSE)
  add_definitions(-DMESHFIX_VERBOSE)
endif()

if (UNIX)
    include_directories(/usr/include/superlu)
    include_directories(/usr/include/eigen3)
endif()

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-strict-aliasing")

add_definitions(-DIS64BITPLATFORM)

# We need C++11. Put this directive after CGAL's include.
if(NOT WIN32)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -g -Wall -Wno-reorder -Wno-unknown-pragmas -Wno-deprecated-declarations -Wno-format-security")
endif()
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
endif()


set(JMESHEXT_DIR "${PROJECT_SOURCE_DIR}/JMeshExt-1.0alpha_src")
set(JMESH_DIR "${JMESHEXT_DIR}/JMeshLib-1.2/")
set(OPENNL_DIR "${JMESH_DIR}/OpenNL3.2.1/")
set(SUPERLU_DIR "${OPENNL_DIR}/SuperLU/")

include_directories("${JMESHEXT_DIR}/include")
include_directories("${JMESH_DIR}/include")
add_subdirectory("${JMESHEXT_DIR}")
add_definitions(-DUSE_FORTRAN_BLAS)
if(APPLE)
  set(CMAKE_CXX_LINK_FLAGS "-framework Accelerate")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
endif()


add_library(meshfix STATIC meshfix.cpp)
#[=[
add_executable(meshfix-bin main.cpp)

target_link_libraries(
  meshfix-bin 
  meshfix 
  jmesh 
  jmeshext 
  nl 
  superlu
openblas lapack
)
set_target_properties(meshfix-bin PROPERTIES OUTPUT_NAME meshfix)
#]=]

#[=[
set(MESHFIX_WITH_LIBIGL OFF)

if(MESHFIX_WITH_LIBIGL)
  find_package(LIBIGL QUIET)
  if(LIBIGL_FOUND)
    option(LIBIGL_USE_STATIC_LIBRARY "Use libigl as static library" OFF)
    #add_subdirectory("${LIBIGL_INCLUDE_DIR}/../shared/cmake" "libigl")
    include_directories(${LIBIGL_INCLUDE_DIRS})
    add_executable(libigl_example libigl_example.cpp)
    target_link_libraries(
      libigl_example
      meshfix 
      jmesh 
      jmeshext 
      nl 
      superlu
openblas lapack
  )
  else()
    message(STATUS "NOT building libigl example (LIBIGL not found)")
  endif()
else()
  message(STATUS "NOT attempting libigl example")
endif()
#]=]
