# ZENO node system

[![CMake](https://github.com/zenustech/zeno/actions/workflows/cmake.yml/badge.svg)](https://github.com/zenustech/zeno/actions/workflows/cmake.yml) [![License](https://img.shields.io/badge/license-MPLv2-blue)](LICENSE) [![Version](https://img.shields.io/github/v/release/zenustech/zeno)](https://github.com/zenustech/zeno/releases)

[Download](https://github.com/zenustech/zeno/releases) | [Repo](https://github.com/zenustech/zeno) | [About us](https://zenustech.com) | [Tutorial](https://zenustech.com/tutorial) | [Videos](https://space.bilibili.com/263032155) | [Q&A Forum](https://github.com/zenustech/zeno/discussions) | [Bug report](https://github.com/zenustech/zeno/issues)

[国内高速下载](https://gitee.com/zenustech/zeno/releases) | [Gitee 镜像仓库](https://gitee.com/zenustech/zeno) | [公司主页](https://zenustech.com) | [中文教程](https://zenustech.com/tutorial) | [视频教程](https://space.bilibili.com/263032155) | [问答论坛](https://github.com/zenustech/zeno/discussions) | [BUG 反馈](https://github.com/zenustech/zeno/issues)

Open-source node system framework, to change your algorithmic code into useful tools to create much more complicated simulations!

 <img src="http://zenustech.oss-cn-beijing.aliyuncs.com/Place-in-Github/WelcomePic2.jpg" width="640" position="left">

ZENO is an OpenSource, Node based 3D system able to produce cinematic physics effects at High Efficiency, it was designed for large scale simulations and has been tested on complex setups.
Aside of its simulation Tools, ZENO provides necessary visualization nodes for users to import and run simulations if you feel that the current software you are using is too slow.

- [Why a new node system?](/docs/motivation.md)


## Features

Integrated Toolbox, from volumetric geometry process tools (OpenVDB), to state-of-art, commercially robust, highly optimized physics solvers and visualization
nodes, and various VFX and simulation solutions based on our nodes (provided by .zsg file in `graphs/` folder).

## Gallery

Fig.1 - Robot hit water

<img src="http://zenustech.oss-cn-beijing.aliyuncs.com/Place-in-Github/GIF/crag_hit_water.gif" width="640" position="left">

Fig.2 - SuperSonic Flow

<img src="http://zenustech.oss-cn-beijing.aliyuncs.com/Place-in-Github/GIF/shock.gif" width="640" position="left">

Fig.3 - Fluid-structure interaction in ZENO

<img src="http://zenustech.oss-cn-beijing.aliyuncs.com/Place-in-Github/GIF/crush-water2.gif" width="640" position="left">

Fig.4 - Rigid fracture in ZENO

<img src="http://zenustech.oss-cn-beijing.aliyuncs.com/Place-in-Github/GIF/mid-autumn-festival.gif" width="640" position="left">

Fig.5 - Muscular-skeleton simulation in ZENO

<img src="http://zenustech.oss-cn-beijing.aliyuncs.com/Place-in-Github/GIF/muscle2.gif" width="640" position="left">

Fig.6 - Large scale Fluids in ZENO

<img src="http://zenustech.oss-cn-beijing.aliyuncs.com/Place-in-Github/GIF/pyramid.gif" width="640" position="left">

# End-user Installation

## Download binary release

Go to the [release page](https://github.com/zenustech/zeno/releases/), and click Assets -> download `zeno-linux-20xx.x.x.tar.gz`.
Then, extract this archive, and simply run `./launcher` (`launcher.exe` for Windows), then the node editor window will shows up if everything is working well.

## How to play

There are some example graphs in the `graphs/` folder, you may open them in the editor and have fun!
Hint: To run an animation for 100 frames, change the `1` on the top-left of node editor to `100`, then click `Run`.
Also MMB to drag in the node editor, LMB click on sockets to create connections. MMB drag in the viewport to orbit camera, Shift+MMB to pan camera.
More details are available in [our official tutorial](https://zenustech.com/tutorial).

## Bug report

If you find the binary version didn't worked properly or some error message has been thrown on your machine, please let me know by opening an [issue](https://github.com/zenustech/zeno/issues) on GitHub, thanks for you support!


# Developer Build

To build ZENO, you need:

- GCC 9+ or MSVC 19+, CMake 3.12+, and Python 3.6+ to build ZENO.
- Pybind11, NumPy and PySide2 (Qt for Python) to run ZENO editor.
- (Optional) OpenVDB for building volume nodes; CUDA for GPU nodes.

> Hint: for Python, please try avoid using virtualenv and Conda if possible.
> WSL is also not recommended because of its limited GUI and OpenGL support.

Click links below for detailed setup for each platform:

- [Windows 10](/docs/dev_win10.md)
- [Ubuntu 20.04](/docs/dev_ubuntu20.md)
- [CentOS 7](/docs/dev_centos7.md)
- [Arch Linux](/docs/dev_archlinux.md)

After finishing building, use `run.py` to run ZENO for development! You may click `File -> Open` to play `graphs/LorenzParticleTrail.zsg` to confirm everything is working well :)


# Miscellaneous

## Write your own extension!

See [zenustech/zeno_addon_wizard](https://github.com/zenustech/zeno_addon_wizard) for an example on how to write custom nodes in ZENO.

## Contributors

Thank you to all the people who have already contributed to ZENO!

[![Contributors](https://contrib.rocks/image?repo=zenustech/zeno)](https://github.com/zenustech/zeno/graphs/contributors)

## License

ZENO is licensed under the Mozilla Public License Version 2.0, see [LICENSE](/LICENSE) for more information.

## Contact us

You may contact us via WeChat:

* @zhxx1987: shinshinzhang

* @archibate: tanh233

* 