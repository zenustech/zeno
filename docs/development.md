[[TOC]]

# Installation for Developers

## Pre-requisites

A computer with CUDA-enabled graphic card is fundamental for building Zeno. In addition, regardless of your operating system, the following are needed before building starts:

- C++ 17 compiler
- CMake 3.12+
- Python 3.8+
- Numpy
- PySide2

To install the software requirements above, run below depending on your operating system.

### Arch Linux

Run in terminal,

```bash
sudo pacman -S gcc make cmake python python-pip python-numpy python-pyqt5 qt5-base libglvnd mesa
```

### Ubuntu 20.04

Run in terminal,

```bash
sudo apt-get install gcc make cmake python-is-python3 python-dev-is-python3 python3-pip libqt5core5a qt5dxcb-plugin libglvnd-dev libglapi-mesa libosmesa6

python --version  # make sure Python version >= 3.8

sudo python -m pip install -U pip
sudo python -m pip install numpy PySide2
```

### Windows 10

1. Install Python 3.8 64-bit. IMPORTANT: make sure to **Add Python 3.8 to PATH**. Then better to reboot your computer. (If `python` is not added to PATH properly, in following steps you'll be redirected to Microsoft Store)
  
2. Start CMD in **Administrator mode** and run,
    ```cmd
        python -m pip install numpy PySide2
    ```
    Make sure it starts to downloading and installing successfully without  `ERROR`  (warnings are OK though). 
    
    - If get  `ERROR: Could not install packages due to an EnvironmentError: [Errno 13] Permission denied: 'c:\\python38\\Lib\\site-packages\\PySide2\\Qt5\\bin\\d3dcompiler_47.dll''` : 
    
        **Quit anti-virus softwares** since they probably prevent  `pip`  from copying DLL files.
    
    
    - If get  `ImportError: DLL load failed while importing QtGui` : 
    
        Try install  [Microsoft Visual C++ Redistributable](https://aka.ms/vs/16/release/vc_redist.x64.exe).

3. Install Visual Studio 2019 Community Edition or later version (for C++17 support).

## Build ZENO

### Arch Linux & Ubuntu 20.04

```bash
cmake -B build
make -C build -j8
```

### Windows 10

Run in CMD in administrator mode,

```cmd
cmake -B build
```

Then open ```build/zeno.sln``` in Visual Studio 2019, and **switch to Release mode in build configurations**, then run `Build -> Build All`.

Note: In MSVC, Release mode must **always be active** when building ZENO, since MSVC uses different allocators in Release and Debug mode. If a DLL of Release mode and a DLL in Debug mode are linked together in Windows, it will crash when passing STL objects.


## Run ZENO for Development

- Linux

```bash
./run.sh
```

- Windows

```cmd
run.bat
```
