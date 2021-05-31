# ZENO

用于物理仿真和其他计算机图形学应用的开源节点系统框架。


# 特性


# 构建与运行

## 安装依赖项

- Arch Linux
```bash
sudo pacman -S git gcc make cmake python python-pip pybind11 python-numpy python-pyqt5 qt5-base libglvnd mesa
```

- Ubuntu 20.04
```bash
sudo apt-get install git gcc make cmake python-is-python3 python-dev-is-python3 python3-pip libqt5core5a qt5dxcb-plugin libglvnd-dev libglapi-mesa libosmesa6

python --version  # 确保 Python 版本 >= 3.7
python -m pip install -U pip
python -m pip install pybind11 numpy PyQt5
```


## 克隆 ZENO 仓库
```bash
git clone https://gitee.com/archibate/zeno.git --depth=10
cd zeno
```


## 构建 ZENO
```bash
cmake -B build
make -C build -j8
```


## 以开发者模式运行 ZENO
```bash
./run.sh
```


## 在 Docker 中运行 ZENO
```bash
./docker.sh
```
