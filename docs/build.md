# Developer Build

To build ZENO, you need:

- A C++17 compiler, CMake 3.12+, and Python 3.6+ to build ZENO.
- Pybind11, NumPy and PySide2 (Qt for Python) to run ZENO editor.
- (Optional) OpenVDB for building volume nodes; CUDA for GPU nodes.

> Hint: for Python, please try avoid using virtualenv and Conda if possible.
> WSL is also not recommended because of its limited GUI and OpenGL support.

Click links below for detailed setup for each platform:

- [Windows 10](dev_win10.md)
- [Ubuntu 20.04](dev_ubuntu20.md)
- [CentOS 7](dev_centos7.md)
- [Arch Linux](dev_archlinux.md)

After successfully loading the editor, you may click `File -> Open` to play `graphs/LorenzParticleTrail.zsg` to confirm everything is working well :)
