# ZENO

Open-source node system framework, to change your algorithmic code into useful tools to create much more complicated simulations!
![rigid3.zsg](images/rigid3.jpg)

## Features

Integrated Toolbox, from volumetric geometry process tools (OpenVDB), to state-of-art, commercially robust, highly optimized physics solvers and visualization
nodes, and various VFX and simulation solutions based on our nodes (provided by .zsg file).
//better have some screenshot here

## ZenCompute (@littlemine)

Open-source code development framework to easily develop high-performance physical simulation code that both run on cpu and gpu with out too much effort.
//better have some screenshot here


# Motivation

Nowadays, many CG artists have reached an agreement that creating arts (especially
physics simulation and animation) using visual-programming tool is very convinent and flexible.

This repo aims to provide a great tool for both technical artists and CG developers, and researchers from physics simulation.

## Easy Plug, Easy Go

One may create complicated simulation scenarios by simply connecting the nodes provided by the system:
For example:
//put a node editor screenshot here
This is the charm of visual-programming, not only the direction of data-flow, but also the logic-flow of the solution algorithm is presented at no more clarity.
In fact, building a molecular simulation from scratch took no more than 7 minutes!

![lennardjones.zsg](images/lennardjones.jpg)

## Flexible

One can easily change or adjust a solution by simply break and reconnect of the nodes.
Unlike many simulation softwares that have fixed functionality, we provide the building
blocks of algorithms in the form of **nodes** at a relatively low granularity.
By connecting these nodes, you can literally create your unique solver that best fits
your need, rather than being limited by the imagination of software programmers.

[TODO: ```rigid_pool.zsg``` to demo solid-fluid couple]

## Performant

ZENO nodes are mainly written in C++. By connecting nodes in our Qt5 editor,
you are invoking our highly optimized programs by our senior engineers. And
all you need to do is to explore in your mind-space without bothering to tackle 
low-level details.
Performance-wisely, it's shown by @zhxx1987 that our FLIP solver is 4x faster than
Houdini at large scale.

![FLIPSolver.zsg](images/FLIPSolver.jpg)

## Control-flows

Unlike many pure functional node systems (e.g. Blender), ZENO has a strong time-order
and provide a lot of control-flow nodes including CachedOnce, BeginForEach, EndFor, etc.
This enable you to make turing-equivalent programs that fit real-world problems.

![forloop.zsg](images/forloop.jpg)

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
- basic OpenVDB ops (by @zhxx1987)
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

[TODO: demostrate our blender addon here]


# Build & Run

## Installation requirements

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
python -m pip install numpy PyQt5 pywin32
```
(Fun fact: you will be redirected to Microsoft Store if `python` is not added to PATH properly :)
Make sure it starts to downloading and installing successfully without `ERROR` (warnings are OK though).

If you got `ERROR: Could not install packages due to an EnvironmentError: [Errno 13] Permission denied: 'c:\\python38\\Lib\\site-packages\\PyQt5\\Qt5\\bin\\d3dcompiler_47.dll''`:
**Quit anti-virus softwares** (e.g. 360), they probably prevent `pip` from copying DLL files.

If you got `ImportError: DLL load failed while importing QtGui`:
Try install [Microsoft Visual C++ Redistributable](https://aka.ms/vs/16/release/vc_redist.x64.exe).

3. Install Visual Studio 2019 Community Edition or later version (for C++17 support in MSVC).


## Build ZENO
- Linux
```bash
cmake -B build
make -C build -j8
```

- Windows
```cmd
cmake -B build
```
Then open ```build/zeno.sln``` in Visual Studio 2019, and **switch to Release mode in build configurations**, then run `Build -> Build All`.

IMPORTANT: In MSVC, Release mode must **always be active** when building ZENO, since MSVC uses different allocators in Release and Debug mode. If a DLL of Release mode and a DLL in Debug mode are linked together in Windows, it will crash when passing STL objects.


## Run ZENO for development
- Linux
```bash
./run.sh
```

- Windows
```cmd
run.bat
```

## Install ZENO globally for Python
```bash
python setup.py install
```



## Packing ZENO into PyPI wheel
```bash
sudo python -m pip install wheel
python setup.py bdist_wheel
ls dist/*.whl
```


## Upload ZENO to PyPI.org (password required)
```bash
sudo python -m pip install twine
twine upload dist/*.whl
```


# Extensions

ZENO is extensible which means we may write node libraries for it.
The source code of all our official libraries are provided in `Projects/`.

## Build libraries

For now, official node libraries will be built by default when running the
```ALL_BUILD``` target of CMake.

### Major dependencies

Building them require some dependencies, please comment the, for example:

```cmake
add_subdirectory(zenbase)
#add_subdirectory(FastFLIP)
```

in `Projects/CMakeLists.txt` if you meet trouble building them.
And here goes the dependencies:

- FastFLIP
  - OpenVDB
  - IlmBase (or OpenEXR)
  - Eigen3
  - TBB

- Zenbase
  - OpenMP C++ (optional)

- Zenvdb
  - OpenVDB
  - IlmBase (or OpenEXR)
  - TBB
  - OpenMP C++ (optional)

- ZMS
  - OpenMP C++ (optional)

<!--

### Requirements

Before building node libraries, you need to install ZENO first (or add zeno source root to PYTHONPATH), to do so:
```bash
python setup.py install
```

### Build libraries

- Linux
```bash
cd Projects
cmake -B build
make -C build
```

- Windows

For Windows users, **make sure to check the [more detailed README](./Projects/README.md)** for the most updated info and known issues before build starts.

```cmd
cd Projects
cmake -B build
```
Then open ```Projects/build/zeno_projects.sln``` in Visual Studio 2019, **switch to Release in build configuration**, run `Build -> Build All`.
-->

## Write your own one!

See ```demo_project/``` for an example on how to write custom nodes in ZENO.

### Installing node libraries

To install a node library for ZENO just copy the `.so` or `.dll` files to `zen/autoload/`. See ```demo_project/CMakeLists.txt``` for how to automate this in CMake.
