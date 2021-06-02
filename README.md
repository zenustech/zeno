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

- Arch Linux
```bash
sudo pacman -S gcc make cmake python python-pip pybind11 python-numpy python-pyqt5 qt5-base libglvnd mesa
```

- Ubuntu 20.04
```bash
sudo apt-get install gcc make cmake python-is-python3 python-dev-is-python3 python3-pip libqt5core5a qt5dxcb-plugin libglvnd-dev libglapi-mesa libosmesa6

python --version  # make sure Python version >= 3.7
python -m pip install -U pip
python -m pip install pybind11 numpy PyQt5
```

- Windows 10
1. Install Python 3.8 64-bit. IMPORTANT: make sure you **Add Python 3.8 to PATH**! After that rebooting your computer would be the best.
2. Start CMD in **Administrator mode** and type these commands:
```cmd
python -m pip install pybind11 numpy PyQt5
```
(Fun fact: you will be redirected to Microsoft Store if `python` is not added to PATH properly :)
Make sure it starts to downloading and installing successfully without `ERROR` (warnings are OK though).

If you got `ERROR: Could not install packages due to an EnvironmentError: [Errno 13] Permission denied: 'c:\\python38\\Lib\\site-packages\\PyQt5\\Qt5\\bin\\d3dcompiler_47.dll'``:
**Quit anti-virus softwares** like 360, they are likely stopping `pip` from copying DLL files..

If you got `ImportError: DLL load failed while importing QtGui: 找不到指定的模块。`:
Try install [Microsoft Visual C++ Redistributable](https://aka.ms/vs/16/release/vc_redist.x64.exe).

3. Install Visual Studio 2017 Community Edition (for free!).


## build ZENO
- Linux
```bash
cmake -B build
make -C build -j8
```

- Windows
Open ZENO repo in Visual Studio 2017, click `Project -> Build All`.


## run ZENO for development
- Linux
```bash
./run.sh
```

- Windows
```cmd
run.bat
```


## package ZENO into PyPI wheel
```bash
python -m pip install wheel twine ninja
python python/setup.py bdist_wheel
echo python/dist/*.whl
```
