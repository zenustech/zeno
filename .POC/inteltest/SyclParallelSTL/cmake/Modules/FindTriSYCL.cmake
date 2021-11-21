#.rst:
# FindTriSYCL
#---------------
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.

#########################
#  FindTriSYCL.cmake
#########################
#
# Tools for finding and building with triSYCL.
#
#  User must define TRISYCL_INCLUDE_DIR pointing to the triSYCL
#  include directory.
#
#  Latest version of this file can be found at:
#    https://github.com/triSYCL/triSYCL

# Requite CMake version 3.5 or higher
cmake_minimum_required (VERSION 3.5)

# Check that a supported host compiler can be found
if(CMAKE_COMPILER_IS_GNUCXX)
  # Require at least gcc 5.4
  if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 5.4)
    message(FATAL_ERROR
      "host compiler - Not found! (gcc version must be at least 5.4)")
  else()
    message(STATUS "host compiler - gcc ${CMAKE_CXX_COMPILER_VERSION}")
    add_compile_options("-std=c++1z") # Use latest available C++ standard
    add_compile_options("-Wall")   # Turn on all warnings
    add_compile_options("-Wextra") # Turn on all warnings

    # Disabling specific warnings
    # warning: ignoring attributes on template argument
    add_compile_options("-Wno-ignored-attributes")
    # warning: comparison between signed and unsigned integer expressions
    add_compile_options("-Wno-sign-compare")
    # warning: ‘<OpenCL func>’ is deprecated
    add_compile_options("-Wno-deprecated-declarations")
    # warning: type qualifiers ignored on function return type
    add_compile_options("-Wno-ignored-qualifiers")
    # warning: unused parameter ‘<id>’
    add_compile_options("-Wno-unused-parameter")

  endif()
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  # Require at least clang 3.9
  if (${CMAKE_CXX_COMPILER_VERSION} VERSION_LESS 3.9)
    message(FATAL_ERROR
      "host compiler - Not found! (clang version must be at least 3.9)")
  else()
    message(STATUS "host compiler - clang ${CMAKE_CXX_COMPILER_VERSION}")
    add_compile_options("-std=c++1z") # Use latest available C++ standard
    add_compile_options("-Wall")   # Turn on all warnings
    add_compile_options("-Wextra") # Turn on all warnings

    # Disabling specific warnings
    # warning: 'const' type qualifier on return type has no effect
    add_compile_options("-Wno-ignored-qualifiers")
    # warning: comparison between signed and unsigned integer expressions
    add_compile_options("-Wno-sign-compare")
    # warning: ‘<OpenCL func>’ is deprecated
    add_compile_options("-Wno-deprecated-declarations")
    # warning: unused parameter ‘<id>’
    add_compile_options("-Wno-unused-parameter")
    # warning: suggest braces around initialization of subobject
    add_compile_options("-Wno-missing-braces")
    # warning: unused variable '<id>'
    add_compile_options("-Wno-unused-variable")
    # warning: instantiation of variable '<templated id>' required here,
    # but no definition is available
    add_compile_options("-Wno-undefined-var-template")

  endif()
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
  # Change to /std:c++latest once Boost::funtional is fixed
  # (1.63.0 with toolset v141 not working)
  add_compile_options("/std:c++14")
  # Replace default Warning Level 3 with 4 (/Wall is pretty-much useless on MSVC
  # system headers are plagued with warnings)
  string(REGEX REPLACE "/W[0-9]" "/W4" CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})

  # Disabling (default) Warning Level 3 output
  # warning C4996: Call to '<algorithm name>' with parameters that may be
  # unsafe - this call relies on the caller to check that the passed values
  # are correct.
  add_compile_options("/wd4996")
  # warning C4267: '=': conversion from 'size_t' to 'int', possible loss of data
  add_compile_options("/wd4267")
  # warning C4244: '=': conversion from 'size_t' to 'double',
  # possible loss of data
  add_compile_options("/wd4244")
  # warning C4305: '<op>': truncation from 'double' to 'float'
  add_compile_options("/wd4305")
  # warning C4101: '<id>': unreferenced local variable
  add_compile_options("/wd4101")
  # warning C4700: uninitialized local variable '<id>' used
  add_compile_options("/wd4700")
  # warning C4189: '<id>': local variable is initialized but not referenced
  add_compile_options("/wd4189")

  # Disabling Warning Level 4 output
  # warning C4100: '<param>': unreferenced formal parameter
  add_compile_options("/wd4100")
  # warning C4459: declaration of '<id>' hides global declaration
  add_compile_options("/wd4459")
  # warning C4127: conditional expression is constant
  add_compile_options("/wd4127")
  # warning C4456: declaration of '<id>' hides previous local declaration
  add_compile_options("/wd4456")

else()
  message(WARNING
    "host compiler - Not found! (triSYCL supports GCC, Clang and MSVC)")
endif()

#triSYCL options
option(TRISYCL_OPENMP "triSYCL multi-threading with OpenMP" ON)
option(TRISYCL_OPENCL "triSYCL OpenCL interoperability mode" ON)
option(TRISYCL_NO_ASYNC "triSYCL use synchronous kernel execution" OFF)
option(TRISYCL_DEBUG "triSYCL use debug mode" OFF)
option(TRISYCL_DEBUG_STRUCTORS "triSYCL trace of object lifetimes" OFF)
option(TRISYCL_TRACE_KERNEL "triSYCL trace of kernel execution" OFF)

mark_as_advanced(TRISYCL_OPENMP)
mark_as_advanced(TRISYCL_OPENCL)
mark_as_advanced(TRISYCL_NO_ASYNC)
mark_as_advanced(TRISYCL_DEBUG)
mark_as_advanced(TRISYCL_DEBUG_STRUCTORS)
mark_as_advanced(TRISYCL_TRACE_KERNEL)

#triSYCL definitions
set(CL_SYCL_LANGUAGE_VERSION 121 CACHE VERSION
  "Host language version to be used by triSYCL (default is: 121)")
set(TRISYCL_CL_LANGUAGE_VERSION 121 CACHE VERSION
  "Device language version to be used by triSYCL (default is: 121)")
set(CMAKE_CXX_STANDARD 17)
set(CXX_STANDARD_REQUIRED ON)


# Find OpenCL package
if(TRISYCL_OPENCL)
  find_package(OpenCL REQUIRED)
  if(UNIX)
    set(BOOST_COMPUTE_INCPATH /usr/include/compute CACHE PATH
      "Path to Boost.Compute headers (default is: /usr/include/compute)")
  endif(UNIX)
endif()

# Find OpenMP package
if(TRISYCL_OPENMP)
  find_package(OpenMP REQUIRED)
endif()

# Find Boost
find_package(Boost 1.58 REQUIRED COMPONENTS chrono log)

# If debug or trace we need boost log
if(TRISYCL_DEBUG OR TRISYCL_DEBUG_STRUCTORS OR TRISYCL_TRACE_KERNEL)
  set(LOG_NEEDED ON)
else()
  set(LOG_NEEDED OFF)
endif()

find_package(Threads REQUIRED)

message(STATUS "triSYCL OpenMP:                   ${TRISYCL_OPENMP}")
message(STATUS "triSYCL OpenCL:                   ${TRISYCL_OPENCL}")
message(STATUS "triSYCL synchronous execution:    ${TRISYCL_NO_ASYNC}")
message(STATUS "triSYCL debug mode:               ${TRISYCL_DEBUG}")
message(STATUS "triSYCL object trace:             ${TRISYCL_DEBUG_STRUCTORS}")
message(STATUS "triSYCL kernel trace:             ${TRISYCL_TRACE_KERNEL}")

# Find triSYCL directory
if(NOT TRISYCL_INCLUDE_DIR)
  message(FATAL_ERROR
    "triSYCL include directory - Not found! (please set TRISYCL_INCLUDE_DIR")
else()
  message(STATUS "triSYCL include directory - Found ${TRISYCL_INCLUDE_DIR}")
endif()

#######################
#  add_sycl_to_target
#######################
#
#  Sets the proper flags and includes for the target compilation.
#
#  targetName : Name of the target to add a SYCL to.
#
function(add_sycl_to_target targetName)
  # Add include directories to the "#include <>" paths
  target_include_directories (${targetName} PUBLIC
    ${TRISYCL_INCLUDE_DIR}
    ${Boost_INCLUDE_DIRS}
    $<$<BOOL:${TRISYCL_OPENCL}>:${OpenCL_INCLUDE_DIRS}>
    $<$<BOOL:${TRISYCL_OPENCL}>:${BOOST_COMPUTE_INCPATH}>)

  # Link dependencies
  target_link_libraries(${targetName} PUBLIC
    $<$<BOOL:${TRISYCL_OPENCL}>:${OpenCL_LIBRARIES}>
    Threads::Threads
    $<$<BOOL:${LOG_NEEDED}>:Boost::log>
    Boost::chrono)

  # Compile definitions
  target_compile_definitions(${targetName} PUBLIC
    $<$<BOOL:${TRISYCL_NO_ASYNC}>:TRISYCL_NO_ASYNC>
    $<$<BOOL:${TRISYCL_OPENCL}>:TRISYCL_OPENCL>
    $<$<BOOL:${TRISYCL_OPENCL}>:BOOST_COMPUTE_USE_OFFLINE_CACHE>
    $<$<BOOL:${TRISYCL_DEBUG}>:TRISYCL_DEBUG>
    $<$<BOOL:${TRISYCL_DEBUG_STRUCTORS}>:TRISYCL_DEBUG_STRUCTORS>
    $<$<BOOL:${TRISYCL_TRACE_KERNEL}>:TRISYCL_TRACE_KERNEL>
    $<$<BOOL:${LOG_NEEDED}>:BOOST_LOG_DYN_LINK>)

  # C++ and OpenMP requirements
  target_compile_options(${targetName} PUBLIC
    ${TRISYCL_COMPILE_OPTIONS}
    $<$<BOOL:${TRISYCL_OPENMP}>:${OpenMP_CXX_FLAGS}>)

  if(${TRISYCL_OPENMP} AND (NOT WIN32))
    # Does not support generator expressions
    set_target_properties(${targetName}
      PROPERTIES
      LINK_FLAGS ${OpenMP_CXX_FLAGS})
  endif(${TRISYCL_OPENMP} AND (NOT WIN32))

endfunction(add_sycl_to_target)
