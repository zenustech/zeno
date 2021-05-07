1. install dependencies (Arch Linux)

```bash
sudo pacman -S cmake python-pip pybind11
sudo pacman -S tbb cblas glm glew glfw boost
sudo pacman -S openvdb eigen openblas lapack
```

2. build and install IlmBase
```bash
git clone https://github.com/aforsythe/IlmBase.git --depth=1
cd IlmBase
./bootstrap
./configure
make -j8
sudo make install
```

3. clone and build ZENO
```bash
git clone https://github.com/archibate/zeno.git --depth=10
cd zeno
pip install -r python/requirements.txt
make
```

4. run ZENO for development
```bash
./run.sh
```

5. package ZENO for release
```bash
./release.sh
```
