#!/bin/bash
set -e

mkdir -p /tmp/zenv-build /tmp/zenv
cd /tmp/zenv-build

PREFIX=/tmp/zenv
NCPU=48

mkdir -p $PREFIX/lib

cp -d /usr/lib/x86_64-linux-gnu/libffi.so* $PREFIX/lib
cp -d /usr/lib/x86_64-linux-gnu/libz.so* $PREFIX/lib
cp -d /usr/lib/x86_64-linux-gnu/libcrypt.so* $PREFIX/lib
cp -d /usr/lib/x86_64-linux-gnu/libgomp.so* $PREFIX/lib
cp -d /usr/lib/x86_64-linux-gnu/libgthread-2.0.so* $PREFIX/lib
cp -d /usr/lib/x86_64-linux-gnu/libglib-2.0.so* $PREFIX/lib

cp -d /usr/lib/x86_64-linux-gnu/libbz2.so* $PREFIX/lib
cp -d /usr/lib/x86_64-linux-gnu/liblzma.so* $PREFIX/lib
cp -d /usr/lib/x86_64-linux-gnu/libzstd.so* $PREFIX/lib
cp -d /usr/lib/x86_64-linux-gnu/libopenblas.so* $PREFIX/lib

cp -d /usr/local/lib/libblosc.so* $PREFIX/lib

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

echo "If ZENO Qt editor failed to start, consider run: apt install qt5dxcb-plugin"
oldwd="\$(pwd)"
cd -- "\$(dirname "\$0")"
newwd="\$(pwd)"
cd -- "\$oldwd"
export LD_LIBRARY_PATH="\$LD_LIBRARY_PATH:\$newwd/lib"
exec -- "\$newwd/bin/python3.6" -m zenqt "\$@"
EOF

chmod +x $PREFIX/start.sh
