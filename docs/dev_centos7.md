# CentOS 7 Setup

## Installation requirements

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

## Build ZENO

```bash
cmake -B build
cmake --build build --parallel

# (Optional) with OpenVDB support:
cmake -B build -DEXTENSION_FastFLIP:BOOL=ON -DEXTENSION_zenvdb:BOOL=ON -DZENOFX_ENABLE_OPENVDB:BOOL=ON
cmake --build build --parallel
```

## Run ZENO

```bash
./run.py
```
