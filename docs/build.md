# Developer Build

## Installation requirements

You need a C++17 compiler, CMake 3.12+, and Python 3.6+ to build ZENO; Pybind11, NumPy and PySide2 (Qt for Python) to run ZENO editor.
Other requirements like GLAD are self-contained and you don't have to worry installing them manually.

> Hint: for Python, please try avoid using virtualenv and Conda if possible.

### Ubuntu 20.04

```bash
# Install basic dependencies:
sudo apt-get install -y gcc make cmake python-is-python3 python-dev-is-python3 python3-pip qt5dxcb-plugin libglvnd libglapi

python --version  # make sure Python version >= 3.6
sudo python -m pip install -U pip
sudo python -m pip install pybind11 numpy PySide2

# (Optional) for easily altering cmake configurations from terminal:
sudo apt-get install -y cmake-curses-gui

# (Optional) Install Zlib, Eigen3, OpenBLAS:
sudo apt-get install -y zlib1g-dev libeigen3-dev libopenblas-dev

# (Optional) Installing OpenVDB dependencies (Boost, TBB, Blosc, OpenEXR):
sudo apt-get install -y libilmbase-dev libopenexr-dev libtbb-dev
sudo apt-get install -y libboost-iostreams-dev libboost-system-dev

git clone https://github.com/Blosc/c-blosc.git --branch=v1.5.0
cd c-blosc
mkdir build
cd build
cmake ..
make -j8
sudo make install
cd ../..

# (Optional) Install OpenVDB:
git clone https://github.com/AcademySoftwareFoundation/openvdb.git --branch=v7.2.1
cd openvdb
mkdir build
cd build
cmake ..
make -j8
sudo make install
cd ../..
```

See also [`Dockerfile`](/Dockerfile) as a reference for full installing steps.

### CentOS 7

```bash
# Install basic dependencies:
sudo yum -y install wget make python3 python3-devel

sudo python3 -m pip install pybind11 numpy PySide2

# Install CMake dependency (OpenSSL):
sudo yum -y install openssl openssl-devel

# Install CMake 3.17:
wget -c https://github.com/Kitware/CMake/releases/download/v3.17.0-rc3/cmake-3.17.0-rc3.tar.gz
tar zxvf cmake-3.17.0-rc3.tar.gz
cd cmake-3.17.0-rc3
./bootstrap
make -j8
sudo make install
cd ..

# Allowing CMake 3.17 to be launched directly from shell:
sudo ln -sf /usr/local/bin/cmake /usr/bin/

cmake --version  # make sure CMake version is 3.17 now

# (Optional) Install Zlib, Eigen3, OpenBLAS:
sudo yum -y install bzip2-devel zlib-devel

git clone https://github.com/eigenteam/eigen-git-mirror.git --branch=3.3.7
cd eigen-git-mirror
mkdir build
cd build
cmake ..
sudo make install
cd ../..

git clone https://github.com/xianyi/OpenBLAS.git --branch=v0.3.17
cd OpenBLAS
make FC=gfortran -j8
sudo make install PREFIX=/usr/local

# Install GCC 9.x
yum -y install centos-release-scl
yum -y install devtoolset-9-gcc
yum -y install devtoolset-9-gcc-c++

# Enable GCC 9.x (must be executed before build)
scl enable devtoolset-9 bash
g++ --version  # Make sure G++ version is 9.x now

# (Optional) Install OpenVDB dependencies (Boost, TBB, Blosc, OpenEXR):
git clone https://github.com/aforsythe/IlmBase.git --branch=v2.0.0
cd IlmBase
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j8
sudo make install
cd ../..

git clone https://github.com/AcademySoftwareFoundation/openexr.git --branch=v2.3.0
cd openexr
mkdir build
cd build
cmake .. -DOPENEXR_BUILD_PYTHON_LIBS:BOOL=OFF -DCMAKE_BUILD_TYPE=Release
make -j8
sudo make install
cd ../..

wget https://cfhcable.dl.sourceforge.net/project/boost/boost/1.67.0/boost_1_67_0.tar.gz
tar zxvf boost_1_67_0.tar.gz
cd boost_1_67_0
./bootstrap.sh
./b2 -j8 --without-python
sudo ./b2 install --without-python

git clone https://github.com/oneapi-src/oneTBB.git --branch=2017_U8
cd oneTBB
make cfg=release -j8
sudo cp -r include/tbb /usr/local/include/
sudo cp -r build/linux_*_release/*.so* /usr/local/lib64/
sudo cp cmake/*.cmake /usr/local/lib/pkgconfig/
sudo cp -r cmake/templates /usr/local/lib/pkgconfig/
cd ..

git clone https://github.com/Blosc/c-blosc.git --branch=v1.5.0
cd c-blosc
mkdir build
cd build
cmake ..
make -j8
sudo make install
cd ../..

# (Optional) Install OpenVDB:
git clone https://github.com/AcademySoftwareFoundation/openvdb.git --branch=v7.2.1
cd openvdb
mkdir build
cd build
cmake ..
make -j8
sudo make install
cd ../..
```

### Windows 10

1. Install Python 3.8 64-bit. IMPORTANT: make sure you **Add Python 3.8 to PATH**! After that rebooting your computer would be the best.

2. Start CMD in **Administrator mode** and type these commands:
```cmd
python -m pip install pybind11 numpy PySide2
```
(Fun fact: you will be redirected to Microsoft Store if `python` is not added to PATH properly :)
Make sure it starts to downloading and installing successfully without `ERROR` (warnings are OK though).

If you got `ERROR: Could not install packages due to an EnvironmentError: [Errno 13] Permission denied: 'c:\\python38\\Lib\\site-packages\\PySide2\\Qt5\\bin\\d3dcompiler_47.dll''`:
**Quit anti-virus softwares** (e.g. 360), they probably prevent `pip` from copying DLL files.

If you got `ImportError: DLL load failed while importing QtGui`:
Try install [Microsoft Visual C++ Redistributable](https://aka.ms/vs/16/release/vc_redist.x64.exe).

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
```

Notice that you may need to install the `English Pack` for VS2019 for vcpkg to work.

For Chinese users, you may also need to follow the instruction in [this zhihu post](https://zhuanlan.zhihu.com/p/383683670) to **switch to domestic source** for faster download.

See also [their official guide](https://github.com/microsoft/vcpkg/blob/master/README_zh_CN.md) for other issues.

### Arch Linux

```bash
sudo pacman -S gcc make cmake python python-pip python-numpy pyside2
```

See also [`Dockerfile.archlinux`](Dockerfile.archlinux) for full installing steps.

### Docker

```bash
./docker.sh
```


## Build ZENO

### Linux

```bash
cmake -B build
cmake --build build --parallel
```

> Optional: You can change some cmake configurations using `ccmake`.

```bash
cmake -B build
ccmake -B build  # will shows up a curses screen, c to save, q to exit
```

> Below is the suggested Extension Setup:

![extension](/images/extension1.png)
![extension](/images/extension2.png)

> if you have confidence with your GPU and CUDA version, also turn ON those CUDA related stuffs, see figures below: (change mesher, gmpm, ZS_CUDA, ZFXCUDA to OFF may skip cmake and gpu dependencies issue, while disable you from using GPU computing features)

<img src="/images/ccmake1.png" alt="ccmake1" style="zoom:98%;" />
<img src="/images/ccmake2.png" alt="ccmake2" style="zoom:50%;" />

### Windows

```cmd
cmake -B build -DCMAKE_BUILD_TYPE=Release

@rem Use this if you are using vcpkg:
@rem cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=[path to vcpkg]/scripts/buildsystems/vcpkg.cmake
```
Then open ```build/zeno.sln``` in Visual Studio 2019, and **switch to Release mode in build configurations**, then run `Build -> Build All` (Ctrl+Shift+B).

> Optional: You can change some cmake configurations using `cmake-gui`.

```cmd
cmake -B build
cmake-gui -B build
@rem will shows up a window to allow you turn ON/OFF the extensions build
```

IMPORTANT: In MSVC, **Release** mode must **always be active** when building ZENO, since MSVC uses different allocators in Release and Debug mode. If a DLL of Release mode and a DLL in Debug mode are linked together in Windows, it will crash when passing STL objects.


## Run ZENO for development

### Linux
```bash
./run.py
```

### Windows
```bash
python run.py
```

After successfully loading the editor, you may click `File -> Open` to play `graphs/LorenzParticleTrail.zsg` to confirm everything is working well :)

## Troubleshooting

If you got:
```bash
This application failed to start because it could not find or load the Qt platform plugin "xxx"

Reinstalling the application may fix this problem.
```

### Windows 10

Are you using Anaconda? Please try using the methods in: https://stackoverflow.com/questions/41994485/how-to-fix-could-not-find-or-load-the-qt-platform-plugin-windows-while-using-m

### Ubuntu 20.04

```bash
sudo apt-get install -y qt5dxcb-plugin
```

Please let me know if you have any trouble not mentioned above by opening an [issue](https://github.com/zenustech/zeno/issues) on GitHub, thanks for you support!

### WSL

WSL doesn't have X11 display by default :( Please try search the Web for how to enable it, sorry!
