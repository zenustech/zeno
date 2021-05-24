# ZENO

禅节点系统 - 一种基于节点图简洁而统一的计算方式


## 构建与运行

### 安装依赖项

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

- Windows

下载并安装 [MSYS2 20210419](https://mirrors.tuna.tsinghua.edu.cn/msys2/distrib/x86_64/msys2-x86_64-20210419.exe), 然后启动 MSYS2 命令行并输入以下命令:

```bash
echo 'Server = http://mirrors.ustc.edu.cn/msys2/mingw/i686' > /etc/pacman.d/mirrorlist.mingw32
echo 'Server = http://mirrors.ustc.edu.cn/msys2/mingw/x86_64' > /etc/pacman.d/mirrorlist.mingw64
echo 'Server = http://mirrors.ustc.edu.cn/msys2/msys/$arch' > /etc/pacman.d/mirrorlist.msys
echo 'Server = http://mirrors.ustc.edu.cn/msys2/mingw/ucrt64' > /etc/pacman.d/mirrorlist.ucrt64

pacman -Sy
pacman -S git cmake
pacman -S gcc make
pacman -S python python-devel python-pip
pacman -S mingw-w64-x86_64-mesa mingw-w64-x86_64-qt5 mingw-w64-x86_64-python-pyqt5

python -m pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
python -m pip install -U pip
python -m pip install pybind11 numpy
```


### 克隆 ZENO 仓库
```bash
git clone https://gitee.com/archibate/zeno.git --depth=10
cd zeno
```


### 构建 ZENO
```bash
cmake -B build
make -C build -j8
```


### 以开发者模式运行 ZENO
```bash
./run.sh
```


## 在 Docker 中运行 ZENO
```bash
./docker.sh
```