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

CMAKE_MINIMUM_REQUIRED( VERSION 3.8.0 )
SET( CMAKE_SKIP_RPATH TRUE )

## project name & version
PROJECT(partio LANGUAGES CXX VERSION 1.5.2)

# Enable C++11
SET(CMAKE_CXX_STANDARD 11)
SET(CMAKE_CXX_STANDARD_REQUIRED YES)
SET(CMAKE_CXX_EXTENSIONS NO)

## Setup platform specific helper defines build variants
IF(WIN32)
  IF(MSVC)
    ADD_DEFINITIONS (-DPARTIO_WIN32)
  ENDIF()
  ADD_DEFINITIONS (-D_USE_MATH_DEFINES)
ELSE()
  ADD_COMPILE_OPTIONS (-Wextra -Wno-unused-parameter)
ENDIF()
IF(APPLE)
  set (CMAKE_SHARED_LINKER_FLAGS "-undefined dynamic_lookup")
ENDIF()

#SET(CMAKE_BUILD_TYPE "Release" CACHE STRING "type of build" FORCE)

## Set install location
IF(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    IF(WIN32)
        set(VARIANT_DIRECTORY "Windows-x86_64")
    ELSE()
        execute_process(COMMAND sh -c "echo `uname`-`uname -r | cut -d'-' -f1`-`uname -m`" OUTPUT_VARIABLE VARIANT_DIRECTORY OUTPUT_STRIP_TRAILING_WHITESPACE)
    ENDIF()
    set(CMAKE_INSTALL_PREFIX "${CMAKE_SOURCE_DIR}/${VARIANT_DIRECTORY}")
ENDIF(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)

SET (ZLIB_LIBRARY "")

#option(BUILD_SHARED_LIBS "Enabled shared libraries" OFF)

####################################################
# FUN FACT: DID U KNOW THAT MICROSOFT LOVES LINUX? #
####################################################
if (MSVC)  # we must use this to fix bullet3 on win
    set(BUILD_SHARED_LIBS OFF)
    add_compile_options(
            $<$<CONFIG:>:/MD> #---------|
            $<$<CONFIG:Debug>:/MDd> #---|-- Statically link the runtime libraries
            $<$<CONFIG:Release>:/MD> #--|
    )  # cl: 命令行 warning D9025 :正在重写“/MT”(用“/MD”)
else()
    set(BUILD_SHARED_LIBS OFF)
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
endif()
####################################################
# FUN FACT: DID U KNOW THAT MICROSOFT BUYS GITHUB? #
####################################################


file(GLOB io_cpp "io/*.cpp")
file(GLOB core_cpp "core/*.cpp")

add_library(partio ${io_cpp} ${core_cpp})
set_target_properties(partio PROPERTIES OUTPUT_NAME partio POSITION_INDEPENDENT_CODE ON)
if (WIN32)
if (MSVC)
  #target_compile_options(partio PUBLIC /MT)
  target_link_options(partio PRIVATE /INCREMENTAL:NO /NODEFAULTLIB:MSVCRT)
endif()
endif()

target_include_directories(partio
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>)
