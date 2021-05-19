# ZENO

ZEn NOde system - a simple & unified way of computation by connecting node graphs


## Build & Run

1. clone ZENO repository
```bash
git clone https://github.com/zensim-dev/zeno.git --depth=10
cd zeno
```

2. install dependencies

- Arch Linux
```bash
sudo pacman -S gcc
sudo pacman -S make
sudo pacman -S cmake
sudo pacman -S python
sudo pacman -S python-pip
sudo pacman -S qt5-base
sudo pacman -S libglvnd
```

- Ubuntu 20.04
```bash
sudo apt-get install gcc
sudo apt-get install make
sudo apt-get install cmake
sudo apt-get install python-is-python3 python-dev-is-python3
sudo apt-get install python-pip-whl
sudo apt-get install libqt5core5a qt5dxcb-plugin
sudo apt-get install libglvnd-dev

# make sure python version is 3.7 or above
python -V
pip -V
```

3. install Python dependencies
```bash
pip install pybind11
pip install numpy
pip install PyQt5
```

4. build ZENO into binary
```bash
cmake -B build
make -C build -j8
```

5. run ZENO for development
```bash
./run.sh
```

6. run ZENO in docker
```bash
./docker.sh
```

7. distribute ZENO for end-user
```bash
./dist.sh
```
