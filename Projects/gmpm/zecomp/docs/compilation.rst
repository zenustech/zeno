Compilation
=============

This is a cross-platform C++/CUDA cmake project. The minimum version requirement of cmake is 3.15, although the latest version is generally recommended. 

.. note::
    Currently tested C++ compilers (as the host compiler for **NVCC**) on different platforms include:

    +----------+------------------+
    | platform | Compilers        |
    +==========+==================+
    | Windows  | msvc142, clang-9 |
    +----------+------------------+
    | Linux    | gcc8.4, clang-9  |
    +----------+------------------+

    In short, the supported compilers should support **C++14** standard and be in compliance with **NVCC**.
    Since the future releases of **CUDA** are officially excluded on Mac OS, and there is a more suitable candidate **Metal** developed by **Apple**, the **Mac OS** platform is not discussed.

Build Commands
-------------------
Run the following command in the *root directory*.
.. code-block:: cpp
    mkdir build
    cd build
    cmake ..
    cmake --build . --config Release --target mgsp

.. note::
    The project can also be configured through other interfaces, e.g. using the *CMake Tools* extension in *Visual Studio Code*, or in *CMake GUI*.

