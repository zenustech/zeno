# ZENO

ZEn NOde system - a simple & unified way of computation by connecting node graphs


## Build & Run

### install dependencies

- Arch Linux
```bash
sudo pacman -S git
sudo pacman -S gcc
sudo pacman -S make
sudo pacman -S cmake
sudo pacman -S python
sudo pacman -S python-pip
sudo pacman -S pybind11
sudo pacman -S python-numpy
sudo pacman -S python-pyqt5
sudo pacman -S qt5-base
sudo pacman -S libglvnd
sudo pacman -S mesa
```

- Ubuntu 20.04
```bash
sudo apt-get install git
sudo apt-get install gcc
sudo apt-get install make
sudo apt-get install cmake
sudo apt-get install python-is-python3
sudo apt-get install python-dev-is-python3
sudo apt-get install python-pip-whl
sudo apt-get install libqt5core5a
sudo apt-get install qt5dxcb-plugin
sudo apt-get install libglvnd-dev
sudo apt-get install libglapi-mesa
sudo apt-get install libosmesa6

python --version  # make sure Python version >= 3.7
python -m pip install -U pip
python -m pip install pybind11
python -m pip install numpy
python -m pip install PyQt5
```

- Windows

Download and install [MSYS2 20210419](https://repo.msys2.org/distrib/x86_64/msys2-x86_64-20210419.exe), then start MSYS2 and type these commands:

```bash
pacman -Sy
pacman -S gcc
pacman -S make
pacman -S cmake
pacman -S python
pacman -S python-devel
pacman -S python-pip
pacman -S mingw-w64-x86_64-mesa
pacman -S mingw-w64-x86_64-qt5
pacman -S mingw-w64-x86_64-python-pyqt5

python -m pip install -U pip
python -m pip install pybind11
python -m pip install numpy
```


- Mac OS X

Comming soon... before that please try install Python, CMake, GCC, OpenGL, Qt5, and Pybind11 manually. If you succeed, please let us know your build step, thanks in advance!


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
