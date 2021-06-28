# Zensim Parallel Compute
*Zensim Parallel Compute* is the codebase **zecomp** (**Ze**nsim Parallel **Comp**ute) maintained by **Zensim Co. Ltd**, which delivers great parallel computing efficiency within a shared-memory heterogeneous architecture through a unified programming interface regardless of the actual running backend.

## **Document**
See [git wiki](https://github.com/zensim-dev/zpc/wiki) page for more details.

## **Compilation**
This is a cross-platform C++/CUDA cmake project. The minimum version requirement of cmake is 3.18, yet the latest version is generally recommended. Please install cmake through official website or python3-pip, since the cmake version in apt repo is behind.

When CUDA is enabled, the required CUDA version is 11.0+ (for c++17).

Currently, *supported OS* is Ubuntu 20.04+, and *tested compilers* includes >gcc10.0, >clang-11. 

### **Build**

Before building this framework, please first manually configure these external dependencies, i.e. [**openvdb**](https://github.com/AcademySoftwareFoundation/openvdb). Then pull all dependencies by

```
git submodule init
git submodule update
```

If CUDA (>=11) is installed and required, be sure to set *ZS_ENABLE_CUDA=On* first.

Configure the project using the *CMake Tools* extension in *Visual Studio Code* (recommended).

## **Code Usage**

> Use the codebase in another cmake c++ project.

Directly include the codebase as a submodule. Or install this repo then use *find_package(Zecomp)*.

> Develop upon the codebase.

Create a sub-folder in *Projects* with a cmake file at its root.

## **Credits**
This framework draws inspirations from [Taichi](https://github.com/taichi-dev/taichi), [MGMPM](https://github.com/penn-graphics-research/claymore), kokkos, vsg, raja.

### **Dependencies**
The following libraries are adopted in our project development:
- [fmt](https://fmt.dev/latest/index.html)
- spdlog
- magic_enum
- gcem
- catch2
- rapidjson
- cxxopts

For spatial data IO and generation, we use these libraries in addition:

- [partio](http://partio.us/)
- [openvdb](https://github.com/AcademySoftwareFoundation/openvdb) 

We import these following libraries as well:

- [function_ref](https://github.com/TartanLlama/function_ref)