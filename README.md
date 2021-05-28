# ZENO

ZEn NOde system - a simple & unified way of computation by connecting node graphs


## Build & Run

### install dependencies

* Git
* GCC
* Make
* CMake
* Python
* Python Pybind11
* Python Numpy
* Python PyQt5
* OpenGL
* Qt5

we have figure out how to install these packages on some platforms as follows:

- Arch Linux
```bash
sudo pacman -S git gcc make cmake python python-pip pybind11 python-numpy python-pyqt5 qt5-base libglvnd mesa
```

- Ubuntu 20.04
```bash
sudo apt-get install git gcc make cmake python-is-python3 python-dev-is-python3 python3-pip libqt5core5a qt5dxcb-plugin libglvnd-dev libglapi-mesa libosmesa6

python --version  # make sure Python version >= 3.7
python -m pip install -U pip
python -m pip install pybind11 numpy PyQt5
```

- Windows

Download and install [MSYS2 20210419](https://repo.msys2.org/distrib/x86_64/msys2-x86_64-20210419.exe), then start MSYS2 shell and type these commands:

```bash
pacman -Sy
pacman -S git gcc make cmake python python-devel python-pip mingw-w64-x86_64-mesa mingw-w64-x86_64-qt5 mingw-w64-x86_64-python-pyqt5

python -m pip install -U pip
python -m pip install pybind11 numpy
```

- Mac OS X

Comming soon... before that please try install all these dependencies manually. If you succeed, please let us know your build step, thanks in advance!


### clone ZENO repository
```bash
git clone https://github.com/zensim-dev/zeno.git --depth=10
cd zeno
```


### build ZENO into binary
```bash
cmake -B build
make -C build -j8
```


### run ZENO for development
```bash
./run.sh
```


## Build & Run with Docker
```bash
./docker.sh
```
