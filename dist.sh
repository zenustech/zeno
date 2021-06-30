#!/bin/bash
set -e

rm -rf /tmp/zenv-build /tmp/zenv
mkdir -p /tmp/zenv-build /tmp/zenv
cd /tmp/zenv-build

PREFIX=/tmp/zenv
NCPU=48

mkdir -p $PREFIX/lib

# apt-get install -y libffi-dev zlib1g-dev patchelf

## openblas
#cp -d /usr/lib/x86_64-linux-gnu/openblas*-pthread/libopenblas*.so $PREFIX/lib
#
## boost
#cp -d /usr/lib/x86_64-linux-gnu/libboost_iostreams.so* $PREFIX/lib
#cp -d /usr/lib/x86_64-linux-gnu/libboost_system.so* $PREFIX/lib
#
## openexr
#git clone https://github.com/zensim-dev/openexr.git --depth=1
#cd openexr
#cmake . -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$PREFIX
#make -j $NCPU
#make install
#cd ..
#
## c-blosc
#git clone https://github.com/zensim-dev/c-blosc.git --branch=v1.5.0 --depth=1
#cd c-blosc
#cmake . -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$PREFIX
#make -j $NCPU
#make install
#cd ..
#
## openvdb
#git clone https://github.com/zensim-dev/openvdb.git --depth=1
#cd openvdb
#mkdir -p build && cd build
#cmake . -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$PREFIX
#make -j $NCPU
#make install
#cd ..

# patchelf
#git clone https://github.com/zensim-dev/patchelf.git --depth=1
#cd patchelf
#./bootstrap.sh
#./configure --enable-optimizations --prefix=$PREFIX
#make -j $NCPU
#make install
#cd ..

# cpython
git clone https://gitee.com/mirrors/cpython.git --branch=3.6 --depth=1
cd cpython
./configure --enable-shared --enable-optimizations --prefix=$PREFIX
make -j $NCPU build_all
make install
cd ..
patchelf --set-rpath $PREFIX/lib $PREFIX/bin/python3.6

# zeno
git clone https://gitee.com/archibate/zeno.git --depth=1
cd zeno
cmake . -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$PREFIX -DUSE_PYTHON_INCLUDE_DIR:BOOL=ON -DPYTHON_INCLUDE_DIR=$PREFIX/include/python3.6 -DPYTHON_EXECUTABLE=$PREFIX/bin/python3.6 #-DEXTENSION_FastFLIP:BOOL=ON -DEXTENSION_zenvdb:BOOL=ON
make -j $NCPU
make install
python3.6 -m pip install -t $PREFIX/lib/python3.6 PyQt5 numpy
$PREFIX/bin/python3.6 setup.py install
cd ..

cat > $PREFIX/start.sh <<EOF
#!/bin/bash

oldwd="\$(pwd)"
cd -- "\$(dirname "\$0")"
newwd="\$(pwd)"
cd -- "\$oldwd"
export LD_LIBRARY_PATH="\$LD_LIBRARY_PATH:\$newwd/lib"
exec -- "\$newwd/bin/python3.6" -m zenqt "\$@"
EOF

chmod +x $PREFIX/start.sh
