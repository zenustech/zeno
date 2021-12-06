# Windows 10 Setup

## Installation requirements

1. Install Python 3.9 64-bit. IMPORTANT: make sure you **Add Python 3.9 to PATH**! After that rebooting your computer would be the best.

> WARNING: **Don't use Python 3.8**, PySide2 have a known bug with 3.8. Also **don't use 32-bit Python**.

2. Start CMD in **Administrator mode** and type these commands:
```cmd
python -m pip install pybind11 numpy PySide2
```
> (Fun fact: you will be redirected to Microsoft Store if `python` is not added to PATH properly :) Make sure it starts to downloading and installing successfully without `ERROR` (warnings are OK though).

> If you got `ERROR: Could not install packages due to an EnvironmentError: [Errno 13] Permission denied: 'c:\\python38\\Lib\\site-packages\\PySide2\\Qt5\\bin\\d3dcompiler_47.dll''`:
> **Quit anti-virus softwares** (e.g. 360), they probably prevent `pip` from copying DLL files.

> If you got `ImportError: DLL load failed while importing QtGui`:
> Try install [Microsoft Visual C++ Redistributable](https://aka.ms/vs/16/release/vc_redist.x64.exe).

3. Install Visual Studio 2019 Community Edition or later version (for C++17 support in MSVC).

4. (Optional) Install other dependencies via [vcpkg](https://github.com/microsoft/vcpkg):

```cmd
git clone https://github.com/microsoft/vcpkg.git --depth=1
cd vcpkg

@rem (Optional) integrate vcpkg into your VS2019 if necessary:
vcpkg integrate install

@rem (Optional) Install OpenVDB for the extension ZenVDB & FastFLIP:
vcpkg install openvdb:x64-windows

@rem (Optional) Install Eigen3 for the extension FastFLIP:
vcpkg install eigen3:x64-windows

@rem (Optional) Install CGAL for the extension CGMesh:
vcpkg install cgal:x64-windows

@rem (Optional) Install OpenBLAS for the extension CGMesh:
vcpkg install openblas:x64-windows

@rem (Optional) Install LAPACK for the extension CGMesh:
vcpkg install lapack:x64-windows

@rem (Optional) Install Alembic for the extension Alembic:
vcpkg install alembic[hdf5]:x64-windows
```

> Notice that you may need to install the `English Pack` for VS2019 for vcpkg to work.

> For Chinese users, you may also need to follow the instruction in [this zhihu post](https://zhuanlan.zhihu.com/p/383683670) to **switch to domestic source** for faster download.

> See also [their official guide](https://github.com/microsoft/vcpkg/blob/master/README_zh_CN.md) for other issues.


## Build ZENO

1. Minimal build:
```cmd
cmake -B build -DCMAKE_BUILD_TYPE=Release

@rem Use this if you are using vcpkg:
@rem cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=[path to vcpkg]/scripts/buildsystems/vcpkg.cmake
```
Then open ```build/zeno.sln``` in Visual Studio 2019, and **switch to Release mode in build configurations**, then run `build -> build all` (Ctrl+Shift+B).

2. (Optional) Enable OpenVDB support:
```
cmake -B build -DEXTENSION_FastFLIP:BOOL=ON -DEXTENSION_zenvdb:BOOL=ON -DZENOFX_ENABLE_OPENVDB:BOOL=ON
```
Then goto VS2019 and run `build -> build all` again.

> IMPORTANT: In MSVC, **Release** mode must **always be active** when building ZENO, since MSVC uses different allocators in Release and Debug mode. If a DLL of Release mode and a DLL in Debug mode are linked together in Windows, it will crash when passing STL objects.


## Run ZENO for development

```bash
python run.py
```

If you got:
```bash
This application failed to start because it could not find or load the Qt platform plugin "xxx"

Reinstalling the application may fix this problem.
```

Are you using Anaconda? Please try using the methods in: https://stackoverflow.com/questions/41994485/how-to-fix-could-not-find-or-load-the-qt-platform-plugin-windows-while-using-m


If you got:
```bash
ImportError: DLL load failed while importing shiboken2
```

Don't use Python 3.8, it's a PySide2 bug, use Python 3.9 (or even 3.6) instead. See also [this post](https://blog.csdn.net/sinat_37938004/article/details/106384172).

If you got:
```bash
  File "D:\zeno\zenqt\ui\visualize\zenvis.py", line 1, in <module>
    from ...bin import pylib_zenvis as core
ImportError: cannot import name 'pylib_zenvis' from 'zenqt.bin' (unknown location)
```

See [issues 243](https://github.com/zenustech/zeno/issues/243#issuecomment-979619095).
You might be using different Python version to run Zeno, from the version Pybind11 is compiled for.
May need to delete the `build` directory and rebuild to force Pybind11 re-search the `PATH`.
