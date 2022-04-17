# How to build Zeno from source

## Software requirements

To work with Zeno, you need:
```cpp
Git >= 2.0 && CMake >= 3.18 && Qt >= 5.14 && (MSVC >= 2019 || GCC >= 9 || Clang >= 11) && (Windows || Linux) && 64bit
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

5. If the install location is `C:\Qt\Qt5.14.2`, then add `C:\Qt\Qt5.14.2\msvc2019_64\bin` to the `PATH` environment variable.

> This is to allow the `zeno.exe` being able to find `Qt5Widgets.dll` there. Otherwise you need to manually copy `C:\Qt\Qt5.12.12\bin\msvc2019_64\bin\Qt5Widgets.dll` and other DLLs to `build\bin`. After this step rebooting the computer (or at least restart Visual Studio) would be best.

### Ubuntu

If you use Linux the setup would be much easier, simply run these commands:

```bash
sudo apt-get install -y git cmake make g++
sudo apt-get install -y qt5-default
```

> We haven't tested Zeno on WSL (they doesn't have X11 by default), but please give feedback if you meet trouble there, we'd happy to help you resolve :)

### Arch Linux

```bash
sudo pacman -S git cmake make g++
sudo pacman -S qt5-base
```

### Mac OS X

Please refer to this video for installation guide :) https://www.bilibili.com/video/BV1uT4y1P7CX

## Building Zeno

### Clone source code

Now that development tools are ready, let's clone the source code of Zeno from GitHub:

```bash
git clone https://github.com/zenustech/zeno.git
cd zeno
```

> If you find GitHub slow: use `git clone https://gitee.com/zenustech/zeno.git` instead, which is our [official Gitee mirror](https://gitee.com/zenustech/zeno).

> May also try `git clone https://github.com/zenustech/zeno.git --depth=1` for only fetching the latest commit, to reduce transmit data size for faster clone.

### Fetch submodules (optional)

You may optionally get the submodules of Zeno as well (for some extension modules):

```bash
git submodule update --init --recursive
```

> Hint: `git submodule update --init --recursive` should also be executed every time you `git pull` (when you'd like to synchronize with latest updates).

> If you find GitHub slow: edit `.gitmodules` and replace GitHub URLs by your corresponding [Gitee](https://gitee.com) mirrors, and re-run the above command.

### Configure CMake

### Windows

```bash
cmake -B build -DQt5_DIR="C:/Qt/Qt5.14.2/msvc2019_64/lib/cmake"
```

Please replace the `C:/Qt/Qt5.14.2` by your custom Qt install location. And make sure you use `/` instead of `\\`, since CMake doesn't recognize `\\`.

### Linux

```bash
cmake -B build
```

### Build Zeno

Starts to build (`4` here means using 4 CPU threads):

```bash
cmake --build build --parallel 4
```

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

## References

- [VCPKG user guide](https://github.com/microsoft/vcpkg/blob/master/README_zh_CN.md)
- [CMake documentation](https://cmake.org/cmake/help/latest/)
- [Git documentation](https://git-scm.com/doc)
- [GitHub documentation](https://docs.github.com/en)
- [Qt5 documentation](https://doc.qt.io/qt-5/)
- [C++ references](https://en.cppreference.com/w/)
- [TBB tutorial](https://www.inf.ed.ac.uk/teaching/courses/ppls/TBBtutorial.pdf)
- [OpenVDB cookbook](https://www.openvdb.org/documentation/doxygen/codeExamples.html)
- [Jiayao's learning materials](https://github.com/jiayaozhang/OpenVDB_and_TBB)
- [ZENO bug report & feedback](https://github.com/zenustech/zeno/issues)
