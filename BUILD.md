# How to build Zeno from source

## Software requirements

To work with Zeno, you need:
```cpp
Git >= 2.0 && CMake >= 3.18 && Qt == 5 && (MSVC >= 2019 || GCC >= 11 || Clang >= 12) && (Windows || Linux) && 64bit
```

### Windows

1. Download and install Git: https://git-scm.com/download/win

2. Download and install CMake: https://github.com/Kitware/CMake/releases/download/v3.21.3/cmake-3.21.3-windows-x86_64.zip

3. Download and install Visual Studio 2019 Community Edition (which is free): https://visualstudio.microsoft.com/zh-hans/downloads/

> Note that we install Visual Studio **only to get the compiler bundled with it**. feel free to use your favorite editors like VSCode or CLion other than Visual Studio for coding.

> It's recommended to install Visual Studio in trivial locations, like `C:/Programs Files (x86)/xxx` or at least `D:/Program Files (x86)/xxx`, so that VCPKG can find it easier.

4. Download and install Qt via their online installer: https://www.qt.io/download-qt-installer

> WARN: You must click the `Qt 5.12.11` option to install the prebuilt Qt binaries, otherwise only the Qt Creator is selected by default.

> JOKE: Yes, we have to register a Qt account to install Qt... because the Qt company sucks :)

### Ubuntu

```bash
sudo apt-get install -y git cmake make g++
sudo apt-get install -y qt5-default
```

> We haven't tested Zeno on WSL, but please give feedback if you meet trouble there, we'd happy to help you resolve :)

### Arch Linux

```bash
sudo pacman -S git cmake make g++
sudo pacman -S qt5
```

### Mac OS X

https://www.bilibili.com/video/BV1uT4y1P7CX

## Building Zeno from Source

### Clone source code

Now that development tools are ready, let's clone the source code of Zeno from GitHub:

```bash
git clone https://github.com/zenustech/zeno.git
cd zeno
```

> If you find GitHub slow: use `git clone https://gitee.com/zenustech/zeno.git` instead, which is our [official Gitee mirror](https://gitee.com/zenustech/zeno).

### Fetch submodules

Update the submodules of Zeno (for downloading dependencies):
```bash
git submodule update --init --recursive
```

> If you find GitHub slow: edit `.gitmodules` and replace GitHub URLs by corresponding [Gitee](https://gitee.com) mirrors, and re-run the above command.

### Configure and build dependencies

> Hint: `git submodule update --init --recursive` should also be executed every time you `git pull` (when you'd like to synchronize with latest updates).

Configure CMake:

```bash
cmake -B build
```

#### Why is 'configure' building a butch of libraries

Note: All the dependencies of Zeno will be installed to `build/zpm/opt` during this step, so it will take some time to 'configure'...

What's going on here is, [our package manager](cmake/ZPM.cmake) will build and install [all the submodules](cmake/ZenoRequires.cmake) we just fetched,
which are libraries required by Zeno.

If you already have [these dependencies](3rdparty/README.md) on your system (via `apt-get` or `vcpkg`), feel free to use:

```bash
rm -rf build
cmake -B build -DZENO_WITH_ZPM:BOOL=OFF
```

to disable ZPM and use system-wide libraries directly.

### Build Zeno itself

Starts to build (`4` here means using 4 CPU threads):

```bash
cmake --build build --parallel 4
```

## Run Zeno

### Windows (cmd)

```cmd
build\zeno.exe
```

### Linux (bash)

```bash
build/zeno
```

## References

- [VCPKG user guide](https://github.com/microsoft/vcpkg/blob/master/README_zh_CN.md)
- [CMake documentation](https://cmake.org/cmake/help/latest/)
- [Git documentation](https://git-scm.com/doc)
- [GitHub documentation](https://docs.github.com/en)
- [Qt5 documentation](https://doc.qt.io/qt-5/)
- [C++ references](https://en.cppreference.com/w/)
- [OpenVDB cookbook](https://www.openvdb.org/documentation/doxygen/codeExamples.html)
- [hipSYCL install guide](https://github.com/illuhad/hipSYCL/blob/develop/doc/installing.md)
- [SYCL specification](https://www.khronos.org/registry/SYCL/specs/sycl-2020/html/sycl-2020.html)
- [Zeno bug report](https://github.com/zenustech/zeno/issues)
