1. clone ZENO repository
```bash
git clone https://github.com/zensim-dev/zeno.git --depth=10
cd zeno
```

2. install dependencies (Arch Linux)

```bash
sudo pacman -S cmake python-pip
sudo pacman -S make gcc
sudo pacman -S qt5-base

pip install pybind11 numpy
pip install PyQt5
```

3. build ZENO to binary
```bash
cmake -B build
make -C build -j8
```

4. run ZENO for development
```bash
./run.sh
```
