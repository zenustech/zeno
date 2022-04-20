# How to build Zeno from source

## Software requirements

To work with Zeno, you need:

```cpp
Git >= 2.0 && CMake >= 3.16 && Qt >= 5.12 && (MSVC >= 2019 || GCC >= 9 || Clang >= 11) && (Windows || Linux) && 64bit
```

### Windows

1. Download and install Git: https://github.com/git-for-windows/git/releases/download/v2.35.1.windows.2/Git-2.35.1.2-64-bit.exe

2. Download and install CMake: https://github.com/Kitware/CMake/releases/download/v3.21.3/cmake-3.21.3-windows-x86_64.zip

3. Download and install Visual Studio 2019 Community Edition (which is free): https://visualstudio.microsoft.com/zh-hans/downloads/

> Note that we install Visual Studio **only to get the compiler bundled with it**. Feel free to use your favorite editors like VSCode or CLion other than Visual Studio for coding.

> It's recommended to install Visual Studio in trivial locations, like `C:/Programs Files (x86)/xxx` or at least `D:/Program Files (x86)/xxx`, so that VCPKG can find it easier.

> If you use VCPKG, you need to select the 'English Language Pack' when install, otherwise VCPKG will fail to work (it doesn't support Chinese characters in path).

4. Download and install Qt5 via their installer: https://download.qt.io/archive/qt/5.14/5.14.2/qt-opensource-windows-x86-5.14.2.exe

> Hint: Try this Tsinghua mirror if official site too slow: https://mirror.tuna.tsinghua.edu.cn/qt/archive/qt/5.14/5.14.2/qt-opensource-windows-x86-5.14.2.exe

> WARN: You must click the `Qt 5.14.2` option to install the prebuilt Qt binaries, otherwise only the Qt Creator is selected by default.

> JOKE: Yes, we have to register a Qt account to install Qt... because the Qt company sucks :)

5. If the install location is `C:\Qt\Qt5.14.2`, then add `C:\Qt\Qt5.14.2\msvc2017_64\bin` to the `PATH` environment variable.

> This is to allow the `zeno.exe` being able to find `Qt5Widgets.dll` there. Otherwise you need to manually copy `C:\Qt\Qt5.14.2\bin\msvc2017_64\bin\Qt5Widgets.dll` and other DLLs to `build\bin`. After this step rebooting the computer (or at least restart Visual Studio) would be best.

> JOKE: Because offcial Qt prebuilt binaries not matching VS2019, so we have to use `msvc2017_64` rather than `msvc2019_64`, it works too.

### Ubuntu

If you use Linux the setup would be much easier, simply run these commands:

```bash
sudo apt-get install -y git cmake make g++
sudo apt-get install -y qt5-default
sudo apt-get install -y libqt5svg5-dev
```

> NOTE: We highly recommend to use **Ubuntu 20.04 (or above)** whenever possible.
> Ubuntu 18.04 users may have to install `Qt >= 5.12` from the official installer manually.
> As a contrast, Ubuntu 20.04 users can easily get Qt 5.12 by running `apt install qt5-default`.

### Arch Linux

Arch Linux is my recommended environment, as it always provide latest packages.

```bash
sudo pacman -S git cmake make g++
sudo pacman -S qt5-base
```

### WSL

We haven't tested Zeno 2.0 on WSL (they doesn't have X11 by default). But please give feedback if you meet trouble there, we'd happy to help you resolve :)
My video about how to setup X11 in WSL1: https://www.bilibili.com/video/BV1u44y1N78v

### Mac OS X

Please refer to this video for installation guide :) https://www.bilibili.com/video/BV1uT4y1P7CX

## Get source code

Now that development tools are ready, let's clone the source code of Zeno from GitHub:

```bash
git clone https://github.com/zenustech/zeno.git
cd zeno
```

> If you find GitHub slow: use `git clone https://gitee.com/zenustech/zeno.git` instead, which is our [official Gitee mirror](https://gitee.com/zenustech/zeno).

> May also try `git clone https://github.com/zenustech/zeno.git --depth=1` for only fetching the latest commit, to reduce transmit data size for faster clone.

## Fetch submodules (optional)

You may *optionally* get the submodules of Zeno as well (for some extension modules like [ZPC](github.com/zenustech/zpc) or [LibIGL](https://github.com/zenustech/libigl)):

```bash
git submodule update --init --recursive
```

> The submodules are really huge (~40MB), if you find it too slow, you may give up for now and try it later. Zeno can still build without these submodules.

> Hint: `git submodule update --init --recursive` should also be executed every time you `git pull` (when you'd like to synchronize with latest updates).

> If you find GitHub slow: edit `.gitmodules` and replace GitHub URLs by your corresponding [Gitee](https://gitee.com) mirrors, and re-run the above command.

## Build Zeno

Quickly recall our CMake knowledge:

1. The first step `cmake -B build` called *configure*, it generates the `build/` directory containing `Makefile`.
2. The second step `cmake --build build` called *build*, equivalant to `make -C build` on Linux and call MSBuild on Windows.

### Windows

```bash
cmake -B build -DQt5_DIR="C:/Qt/Qt5.14.2/msvc2017_64/lib/cmake/Qt5"
cmake --build build --config Release
```

Please replace the `C:/Qt/Qt5.14.2` by your custom Qt install location. And make sure you use `/` instead of `\\`, since CMake doesn't recognize `\\`.

> The `--config Release` argument is **only required on Windows**, thank to the fact that MSBuild is a multi-config generator.
> If you use `-DCMAKE_BUILD_TYPE=Debug` in the *configure* phase, then you should also `--config Debug` in the *build* phase.

> Also, Windows doesn't support `--parallel` argument, which means MSBuild is a single-threaded build system, you have to wait.

### Linux

```bash
cmake -B build
cmake --build build --parallel 4
```

> `--parallel 4` here means using 4 CPU threads.

## Run Zeno

After build, you will find all the EXE and DLL files in `build/bin` directory.

### Windows

```cmd
build\bin\zenoedit.exe
```

### Linux

```bash
build/bin/zenoedit
```

This should shows up an node editor window if everything is working well.

## Building Zeno extensions (optional)

Zeno is a C++ project with thousands of `.cpp` files. However what you have just built is just some
hundreds of them. To build the full-featured version of Zeno, you need to build the Zeno extension modules.

> Why not build the extension modules by default? Because they require a lot of dedicated setup, which can
> be unfriendly to new users. E.g., some of them depends on `x86_64` architecture, or the `CUDA` toolkit.

> We decide to keep **the core part of Zeno** build without any dependencies to **make it easy for new users**.
> People with dependencies installed may turn on some of the extensions manually if they'd like to.
> (In fact, the Zeno editor which requires Qt may also be turned off, building only the API of Zeno core).

If you are ready to challage, please go ahead to [`docs/BUILD_EXT.md`](docs/BUILD_EXT.md).

## References

- [VCPKG user guide](https://github.com/microsoft/vcpkg/blob/master/README_zh_CN.md)
- [CMake documentation](https://cmake.org/cmake/help/latest/)
- [Git documentation](https://git-scm.com/doc)
- [GitHub documentation](https://docs.github.com/en)
- [Qt5 documentation](https://doc.qt.io/qt-5/)
- [C++ references](https://en.cppreference.com/w/)
- [C++ core guidelines](https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines)
- [TBB tutorial](https://www.inf.ed.ac.uk/teaching/courses/ppls/TBBtutorial.pdf)
- [OpenVDB cookbook](https://www.openvdb.org/documentation/doxygen/codeExamples.html)
- [OpenGL online references](http://docs.gl)
- [Jiayao's learning materials](https://github.com/jiayaozhang/OpenVDB_and_TBB)
- [My public course on CMake](https://www.bilibili.com/video/BV16P4y1g7MH?spm_id_from=333.999.0.0)
- [ZENO bug report & feedback](https://github.com/zenustech/zeno/issues)
