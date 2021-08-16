# Arch Linux Setup

## Installation requirements

```bash
# Install basic dependencies:
sudo pacman -S --noconfirm gcc make cmake python python-pip python-numpy pyside2
```

See also [`/scripts/Dockerfile.archlinux`](/scripts/Dockerfile.archlinux) for full installing steps.

## Build ZENO

```bash
cmake -B build
cmake --build build --parallel

# (Optional) with OpenVDB support:
cmake -B build -DEXTENSION_FastFLIP:BOOL=ON -DEXTENSION_zenvdb:BOOL=ON -DZENOFX_ENABLE_OPENVDB:BOOL=ON
cmake --build build --parallel
```

## Run ZENO

```bash
./run.py
```
