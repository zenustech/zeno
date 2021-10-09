# Ubuntu 20.04 Setup

## Installation requirements

```bash
# Install basic dependencies:
sudo apt-get install -y gcc make cmake python-is-python3 python-dev-is-python3 python3-pip qt5dxcb-plugin libglvnd libglapi

python --version  # make sure Python version >= 3.6
sudo python -m pip install -U pip
sudo python -m pip install pybind11 numpy PySide2

# (Optional) for easily altering cmake configurations from terminal (ccmake):
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

# (Optional) Install CGAL:
sudo apt-get install -y cgal-dev
```

See also [`Dockerfile`](/Dockerfile) as a reference for full installing steps.

## Build ZENO

```bash
# Minimal build:
cmake -B build
cmake --build build --parallel

# (Optional) Enable OpenVDB support:
cmake -B build -DEXTENSION_FastFLIP:BOOL=ON -DEXTENSION_zenvdb:BOOL=ON -DZENOFX_ENABLE_OPENVDB:BOOL=ON
cmake --build build --parallel

# (Optional) Enable CUDA support (for NVIDIA users):
cmake -B build -DEXTENSION_gmpm:BOOL=ON -DEXTENSION_mesher:BOOL=ON -DZFX_ENABLE_CUDA:BOOL=ON
cmake --build build --parallel

# (Optional) Enable CGAL support:
cmake -B build -DEXTENSION_cgmesh:BOOL=ON
cmake --build build --parallel

# (Optional) Enable Bullet support:
cmake -B build -DEXTENSION_Rigid:BOOL=ON
cmake --build build --parallel
```

## Run ZENO

```bash
./run.py
```

If you got:
```bash
This application failed to start because it could not find or load the Qt platform plugin "xxx"

Reinstalling the application may fix this problem.
```

Try this:
```bash
sudo apt-get install -y qt5dxcb-plugin
```

