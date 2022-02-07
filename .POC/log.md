```bash
git clone https://gitee.com/mirrors/cpython.git --branch=3.9 --depth=1
cd cpython
./configure --enable-optimizations --enable-shared --prefix=/opt/python3.9
make -j`nproc` build_all
```
