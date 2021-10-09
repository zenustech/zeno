# How to build Zeno from source

## Software requirements

To work with Zeno, you need:
```cpp
Git >= 2.0 && CMake >= 3.12 && (MSVC >= 2019 || GCC >= 11 || Clang >= 12) && (Windows || Linux) && 64bit
```

* Windows

1. Download and install Git: https://git-scm.com/download/win
2. Download and install CMake: https://github.com/Kitware/CMake/releases/download/v3.21.3/cmake-3.21.3-windows-x86_64.zip
3. Download and install Visual Studio 2019 Community Edition (which is free): https://visualstudio.microsoft.com/zh-hans/downloads/

> As we all know, Visual Studio is a compiler that bundled with an IDE :)

* Linux (Ubuntu)

```bash
sudo apt-get install git cmake g++
```

## Install required libraries (optional)

Zeno depends on many C++ libraries, we recommend to use the cross-platform package manager `vcpkg` to install these requirements. (feel free to follow your own favor as long as the requirements below can be installed :)

```bash
git clone https://github.com/microsoft/vcpkg.git --depth=1
cd vcpkg
```

Now, let's install our dependencies via `vcpkg`:

* Windows (cmd)

```cmd
bootstrap-vcpkg.bat
vcpkg integrate install
vcpkg install openvdb:x64-windows
vcpkg install eigen3:x64-windows
vcpkg install cgal:x64-windows
vcpkg install openblas:x64-windows
vcpkg install lapack:x64-windows
```

> Windows users may need to **install the `English Pack` of Visual Studio** for vcpkg to work.

> For Chinese users, you may also need to follow the instruction in [this zhihu post](https://zhuanlan.zhihu.com/p/383683670) to **switch to domestic source** for faster download.

* Linux (bash)

```bash
./bootstrap-vcpkg.sh
./vcpkg install openvdb:x64-linux
./vcpkg install eigen3:x64-linux
./vcpkg install cgal:x64-linux
./vcpkg install openblas:x64-linux
./vcpkg install lapack:x64-linux
```

> Note that the `x64-windows` suffix **must be specified** otherwise vcpkg will install 32-bit packages by default :)

## Building Zeno from Source

Now that all requirements are satisfied, let's clone Zeno from GitHub:

```bash
git clone https://github.com/zenustech/zeno.git
cd zeno
```

Configure CMake:

```bash
cmake -B build -DCMAKE_TOOLCHAIN_FILE=[path to vcpkg]/scripts/buildsystems/vcpkg.cmake
```

> Please replace the `[path to vcpkg]` here to the path where we previously cloned vcpkg, for example: `C:/Users/Administrator/vcpkg`, with `\\` replaced with `/` to make cmake happy.

Starts to build (CMake will automatically find your VS2019 thanks to VCPKG):

```bash
cmake --build build --parallel 8
```

## Run Zeno

* Windows (cmd)

```cmd
build\main\zeno.main.exe
```

* Linux (bash)

```bash
build/main/zeno.main
```

## References

- [vcpkg user guide](https://github.com/microsoft/vcpkg/blob/master/README_zh_CN.md)
- [CMake documentation](https://cmake.org/cmake/help/latest/)
- [Git documentation](https://git-scm.com/doc)
- [C++ references](https://en.cppreference.com/w/)
- [OpenVDB cookbook](https://www.openvdb.org/documentation/doxygen/codeExamples.html)
- [Zeno bug report](https://github.com/zenustech/zeno/issues)
