# ZENO

Open-source node system framework for physics simulation and other CG applications.


# Features

Nowadays, many CG artists have reached an agreement that creating arts (especially
physics simulation and animation) using nodes is very convinent and flexible.

And ZENO aims to provide a great tool for both technical artists and CG developers.

## Intuitive

Without one line of code, artists may create animations by simply connecting nodes.
Here's an example of water simulation:

This is the charm of node programming, the direction of data flow is very clear
to you as the inputs & outputs are connected, making it very easy to create and
understand.

## Flexible

## Performant

ZENO nodes are mainly written in C++. By connecting nodes in our Qt5 editor,
you are invoking our highly optimized programs by our senior engineers. And
all you need to do is explore the art-space without bothering to tackle these
low-level details.
In fact, it's shown by @zhxx1987 that our FLIP solver is 4x faster than Houdini
at large scale.

## Unified

Despite we already have many node systems today, but the are usually bound to
specific softwares, e.g. Blender, Houdini, Unreal, etc.. These softwares usually
already contains a lot of nodes and assumptions and thus hard to use for developers
to extend it.
What's more, a user who wrote a cloth simulation node for Blender cannot couple
with a user who wrote a fluid simulation in Houdini.
So, we want to create a unified framework customized for simulation with nodes.

## Extensible

As a comparison, the ZENO node system is very extensible. Although ZENO itself
doesn't provide any solvers, instead it allows users to **write their own nodes**
using its C++ or Python API.
Here's some of the node libraries that have been implemented by our developers:

- basic primitive ops (by @archibate)
- OpenVDB FLIP fluids (by @zhxx1987)
- Tree-code N-body (by @archibate)
- Molocular Dynamics (by @victoriacity)
- GPU MPM with CUDA (by @littlemine)

Loading these libraries would add corresponding functional nodes into ZENO,
after which you can creating node graphs with them for simulation.
You may also add your own solver nodes to ZENO with this workflow if you'd like.

## Integratable

Not only you can play ZENO in our official Qt5 editor, but also we may install
ZENO as a **Blender addon**! With that, you can enjoy the flexibilty of ZENO
node system and all other powerful tools in Blender.


# Build & Run

## install requirements

You need a C++17 compiler, CMake 3.12+, and Python 3.6+ to build ZENO; NumPy and PyQt5 to run ZENO editor.
Other requirements like Pybind11 or GLAD are self-contained and you don't have to worry installing them manually.

- Arch Linux
```bash
sudo pacman -S gcc make cmake python python-pip python-numpy python-pyqt5 qt5-base libglvnd mesa
```

- Ubuntu 20.04
```bash
sudo apt-get install gcc make cmake python-is-python3 python-dev-is-python3 python3-pip libqt5core5a qt5dxcb-plugin libglvnd-dev libglapi-mesa libosmesa6

python --version  # make sure Python version >= 3.7
sudo python -m pip install -U pip
sudo python -m pip install numpy PyQt5
```

- Windows 10
1. Install Python 3.8 64-bit. IMPORTANT: make sure you **Add Python 3.8 to PATH**! After that rebooting your computer would be the best.
2. Start CMD in **Administrator mode** and type these commands:
```cmd
python -m pip install numpy PyQt5
```
(Fun fact: you will be redirected to Microsoft Store if `python` is not added to PATH properly :)
Make sure it starts to downloading and installing successfully without `ERROR` (warnings are OK though).

If you got `ERROR: Could not install packages due to an EnvironmentError: [Errno 13] Permission denied: 'c:\\python38\\Lib\\site-packages\\PyQt5\\Qt5\\bin\\d3dcompiler_47.dll''`:
**Quit anti-virus softwares** like 360, they are likely stopping `pip` from copying DLL files..

If you got `ImportError: DLL load failed while importing QtGui`:
Try install [Microsoft Visual C++ Redistributable](https://aka.ms/vs/16/release/vc_redist.x64.exe).

3. Install Visual Studio 2019 Community Edition or later version (for C++17 support in MSVC).


## build ZENO
- Linux
```bash
cmake -B build
make -C build -j8
```

- Windows
```cmd
cmake -B build
```
Then open ```build/zeno.sln``` in Visual Studio 2019, and **switch to Release in build configuration**, then click `Project -> Build All`.

IMPORTANT: you must always use Release mode in MSVC, cause they use different allocators in Release and Debug mode, if you link a DLL of Release mode and a DLL in Debug mode together on Windows, they will crash when passing STL objects...


## run ZENO for development
- Linux
```bash
./run.sh
```

- Windows
```cmd
run.bat
```

## install ZENO globally for Python
```bash
python setup.py install
```



## package ZENO into PyPI wheel
```bash
sudo python -m pip install wheel
python setup.py bdist_wheel
ls dist/*.whl
```


## upload ZENO to PyPI.org (needs password)
```bash
sudo python -m pip install twine
twine upload dist/*.whl
```


# Node libraries

ZENO is extensible which means we may write node libraries for it.


## Build requirements

Before building node libraries, you need to install ZENO first (or add zeno source root to PYTHONPATH), to do so:
```bash
python setup.py install
```

## Build official node libraries

- Linux
```bash
cd Projects
cmake -B build
make -C build
```

- Windows
```cmd
cd Projects
cmake -B build
```
Then open ```Projects/build/zeno_projects.sln``` in Visual Studio 2019, **switch to Release in build configuration**, click `Project -> Build All`.


## Write your own one!

See ```demo_project/``` for example on how to write custom nodes in ZENO.

## Installing node libraries

To install a node library for ZENO just copy the `.so` or `.dll` files to `zen/autoload/`. See ```demo_project/CMakeLists.txt``` for how to automate this in CMake.
