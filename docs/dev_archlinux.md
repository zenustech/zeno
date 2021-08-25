# Arch Linux Setup

## Installation requirements

```bash
# Install basic dependencies:
sudo pacman -S --noconfirm gcc make cmake python python-pip python-numpy pyside2
```

See also [`/scripts/tools/Dockerfile.archlinux`](/scripts/tools/Dockerfile.archlinux) for full installing steps.

## Build ZENO

```bash
# Minimal build:
cmake -B build
cmake --build build --parallel

# (Optional) Enable OpenVDB support:
cmake -B build -DEXTENSION_FastFLIP:BOOL=ON -DEXTENSION_zenvdb:BOOL=ON -DZENOFX_ENABLE_OPENVDB:BOOL=ON
cmake --build build --parallel

# (Optional) Enable CUDA support:
cmake -B build -DEXTENSION_gmpm:BOOL=ON -DEXTENSION_mesher:BOOL=ON
cmake --build build --parallel
```

## Run ZENO

```bash
./run.py
```
