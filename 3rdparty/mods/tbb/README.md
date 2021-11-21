### Intel(R) Threading Building Blocks

[![Stable release](https://img.shields.io/badge/version-2020.2-green.svg)](https://github.com/01org/tbb/releases/tag/v2020.2)
[![Apache License Version 2.0](https://img.shields.io/badge/license-Apache_2.0-green.svg)](LICENSE)
[![Build Status](https://travis-ci.org/wjakob/tbb.svg?branch=master)](https://travis-ci.org/wjakob/tbb)
[![Build status](https://ci.appveyor.com/api/projects/status/fvepmk5nxekq27r8?svg=true)](https://ci.appveyor.com/project/wjakob/tbb/branch/master)

This is git repository is currently based on TBB 2020.2 and will be
updated from time to time to track the most recent release. The only
modification is the addition of a CMake-based build system.

This is convenient for other projects that use CMake and TBB because TBB can be
easily incorporated into their build process using git submodules and a simple
``add_subdirectory`` command.

Currently, the CMake-based build can create shared and static versions of
`libtbb`, `libtbbmalloc` and `libtbbmalloc_proxy` for the Intel `i386` and
`x86_64` architectures on Windows (Visual Studio, MinGW), MacOS (Clang) and
Linux (GCC & Clang). The `armv7` and `armv8` architectures are supported on
Linux (GCC & Clang). Other combinations may work but have not been tested.

See index.html for general directions and documentation regarding TBB.

See examples/index.html for runnable examples and directions.

See http://threadingbuildingblocks.org for full documentation
and software information.

Note: Intel, Thread Building Blocks, and TBB are either registered trademarks or
trademarks of Intel Corporation in the United States and/or other countries.

The CMake build contains the following additional/changed files that are not
part of the regular release: ``build/mingw_cross_toolchain.cmake``,
``build/version_string.ver.in``, ``.gitignore`` (modified), ``README.md`` (this
file), and ``Makefile.old`` (renamed from ``Makefile``).
