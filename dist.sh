#!/bin/bash
set -e

rm -rf /tmp/zenv-build /tmp/zenv
mkdir -p /tmp/zenv-build /tmp/zenv
cd /tmp/zenv-build

PREFIX=/tmp/zenv
NCPU=48

mkdir -p $PREFIX/lib

# patchelf
git clone https://github.com/zensim-dev/patchelf.git --depth=1
cd patchelf
./bootstrap.sh
./configure --prefix=$PREFIX
make -j $NCPU
make install
cd ..

# cpython
git clone https://github.com/zensim-dev/cpython.git --branch=3.8 --depth=1
cd cpython
./configure --enable-shared --enable-optimizations --prefix=$PREFIX
make -j $NCPU build_all
make install
cd ..
$PREFIX/bin/patchelf --set-rpath $PREFIX/lib $PREFIX/bin/python3.8

# boost
cp -d /usr/lib/x86_64-linux-gnu/libboost_iostreams.so* $PREFIX/lib
cp -d /usr/lib/x86_64-linux-gnu/libboost_system.so* $PREFIX/lib

# zlib
#cp -d /usr/lib/x86_64-linux-gnu/libz.so* $PREFIX/lib
#cp -d /usr/lib/x86_64-linux-gnu/liblzma.so* $PREFIX/lib

# c-blosc
git clone https://github.com/Blosc/c-blosc.git --branch=v1.5.0 --depth=1
cd c-blosc
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$PREFIX
make -C build -j $NCPU
make -C build install
cd ..

# openvdb
git clone https://github.com/AcademySoftwareFoundation/openvdb.git --depth=1
cd openvdb
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$PREFIX
make -C build -j $NCPU
make -C build install
cd ..

# zeno
git clone https://github.com/zensim-dev/zeno.git --depth=1
cd zeno
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$PREFIX -DUSE_PYTHON_INCLUDE_DIR:BOOL=ON -DPYTHON_INCLUDE_DIR=$PREFIX/include/python3.8 -DPYTHON_EXECUTABLE=$PREFIX/bin/python3.8
make -C build -j $NCPU
make -C build install
python3.8 -m pip install -t $PREFIX/lib/python3.8 PyQt5 numpy
$PREFIX/bin/python3.8 setup.py install
cd ..
