# PARTIO SOFTWARE
# Copyright 2010 Disney Enterprises, Inc. All rights reserved
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
# 
# * Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.
# 
# * Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in
# the documentation and/or other materials provided with the
# distribution.
# 
# * The names "Disney", "Walt Disney Pictures", "Walt Disney Animation
# Studios" or the names of its contributors may NOT be used to
# endorse or promote products derived from this software without
# specific prior written permission from Walt Disney Pictures.
# 
# Disclaimer: THIS SOFTWARE IS PROVIDED BY WALT DISNEY PICTURES AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING,
# BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE, NONINFRINGEMENT AND TITLE ARE DISCLAIMED.
# IN NO EVENT SHALL WALT DISNEY PICTURES, THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND BASED ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.

cmake_minimum_required(VERSION 3.15.0)
project(partio LANGUAGES CXX)

if(WIN32)
    option(PARTIO_BUILD_SHARED_LIBS "Enabled shared libraries" OFF)
else()
    option(PARTIO_BUILD_SHARED_LIBS "Enabled shared libraries" ON)
endif()

# Enable C++11
if (DEFINED ENV{CXXFLAGS_STD})
    set(CXXFLAGS_STD $ENV{CXXFLAGS_STD})
else()
    set(CXXFLAGS_STD "c++11")
endif()
if (${CXXFLAGS_STD} STREQUAL "c++17")
    set(CMAKE_CXX_STANDARD 17)
elseif (${CXXFLAGS_STD} STREQUAL "c++14")
    set(CMAKE_CXX_STANDARD 14)
else()
    set(CMAKE_CXX_STANDARD 11)
endif()

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(THREADS_PREFER_PTHREAD_FLAG OFF)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_FIND_PACKAGE_RESOLVE_SYMLINKS ON)
set(CMAKE_THREAD_PREFER_PTHREAD ON)
set(THREADS_PREFER_PTHREAD_FLAG ON)
set(CMAKE_SKIP_RPATH ON)
set(CMAKE_INSTALL_MESSAGE LAZY)

## Setup platform specific helper defines build variants
if (WIN32)
    if (MSVC)
        add_definitions(-DPARTIO_WIN32)
    endif()
    add_definitions(-D_USE_MATH_DEFINES)
else()
    add_compile_options(-Wextra -Wno-unused-parameter)
endif()

if (APPLE)
  set(CMAKE_SHARED_LINKER_FLAGS "-undefined dynamic_lookup")
endif()

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    # Detect the build type from the $FLAVOR environment variable
    # Default to optimized Release builds when unspecified.
    if ("$ENV{FLAVOR}" MATCHES "debug")
        set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "type of build" FORCE)
    elseif ("$ENV{FLAVOR}" MATCHES "profile")
        set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING "type of build" FORCE)
    else ()
        set(CMAKE_BUILD_TYPE "Release" CACHE STRING "type of build" FORCE)
    endif()
endif()

## Set install location
include(GNUInstallDirs)
if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    if (WIN32)
        set(VARIANT_DIRECTORY "Windows-x86_64")
    else()
        execute_process(COMMAND sh -c "echo `uname`-`uname -r | cut -d'-' -f1`-`uname -m`" OUTPUT_VARIABLE VARIANT_DIRECTORY OUTPUT_STRIP_TRAILING_WHITESPACE)
    endif()
    set(CMAKE_INSTALL_PREFIX "${CMAKE_SOURCE_DIR}/${VARIANT_DIRECTORY}")
endif()

## Search for useful libraries
find_package(Threads REQUIRED)
find_package(ZLIB)
if (ZLIB_FOUND)
    add_definitions(-DPARTIO_USE_ZLIB)
else()
    set(ZLIB_LIBRARY "")
endif()

# Make modules able to see partio library
set(PARTIO_LIBRARIES zspartio ${ZLIB_LIBRARY})
 

# build zspartio
file(GLOB io_cpp "io/*.cpp")
file(GLOB core_cpp "core/*.cpp")

if(PARTIO_BUILD_SHARED_LIBS)
    set(PARTIO_LIBRARY_TYPE SHARED)
else()
    set(PARTIO_LIBRARY_TYPE STATIC)
endif()
add_library(zspartio ${PARTIO_LIBRARY_TYPE} ${io_cpp} ${core_cpp})
set_target_properties(zspartio PROPERTIES OUTPUT_NAME zspartio POSITION_INDEPENDENT_CODE ON)

if (ZLIB_FOUND)
    target_link_libraries(zspartio PUBLIC ZLIB::ZLIB)
endif()

install(TARGETS zspartio DESTINATION ${CMAKE_INSTALL_LIBDIR})

file(GLOB public_includes "*.h")
install(FILES ${public_includes} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
