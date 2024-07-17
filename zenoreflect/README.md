# Zeno Reflect

This project should support C++17 and above C++ standards.

[English](docs/README-en.md) | [中文](docs/README-zh.md)

## Quickstart

1. Add this project as a submodule to your project.
2. Include this project in your CMake configuration:
    ```cmake
    add_subdirectory(/path/to/ZenoReflect)
    ```
3. Then, use it in the module where you need reflection (refer to the `examples` folder):
    ```cmake
    make_absolute_paths(REFLECTION_HEADERS 
        include/data.h
        include/test.h
    ) 
    zeno_declare_reflection_support(ReflectExample "${REFLECTION_HEADERS}")
    ```
