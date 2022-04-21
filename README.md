# ZENO node system

[![CMake](https://github.com/zenustech/zeno/actions/workflows/cmake.yml/badge.svg)](https://github.com/zenustech/zeno/actions/workflows/cmake.yml)
[![License](https://img.shields.io/badge/license-MPLv2-blue)](LICENSE)
[![Version](https://img.shields.io/github/v/release/zenustech/zeno)](https://github.com/zenustech/zeno/releases) 

![Lines of code](https://img.shields.io/tokei/lines/github/zenustech/zeno)
![Code size](https://img.shields.io/github/languages/code-size/zenustech/zeno)
![Repo size](https://img.shields.io/github/repo-size/zenustech/zeno)

![Commit activity](https://img.shields.io/github/commit-activity/m/zenustech/zeno)
![Commits since latest release](https://img.shields.io/github/commits-since/zenustech/zeno/latest)
![GitHub contributors](https://img.shields.io/github/contributors/zenustech/zeno)

![GitHub release downloads](https://img.shields.io/github/downloads/zenustech/zeno/total)
![GitHub latest release downloads](https://img.shields.io/github/downloads/zenustech/zeno/latest/total)
![Repo stars](https://img.shields.io/github/stars/zenustech/zeno?style=social)

[Download](https://github.com/zenustech/zeno/releases) | [Repo](https://github.com/zenustech/zeno) | [About us](https://zenustech.com) | [Tutorial](https://zenustech.com/tutorial) | [Videos](https://space.bilibili.com/263032155) | [Q&A Forum](https://github.com/zenustech/zeno/discussions) | [Build from source](https://github.com/zenustech/zeno/blob/master/BUILD.md) | [Contributor Guidelines](https://github.com/zenustech/zeno/blob/master/docs/CONTRIBUTING.md) | [Bug report](https://github.com/zenustech/zeno/issues)

[国内高速下载](https://zenustech.com/d/) | [Gitee 镜像仓库](https://gitee.com/zenustech/zeno) | [公司主页](https://zenustech.com) | [中文教程](https://zenustech.com/tutorial) | [视频教程](https://space.bilibili.com/263032155) | [问答论坛](https://github.com/zenustech/zeno/discussions) | [从源码构建](https://github.com/zenustech/zeno/blob/master/BUILD.md) | [贡献者指南](https://github.com/zenustech/zeno/blob/master/docs/CONTRIBUTING.md) | [BUG 反馈](https://github.com/zenustech/zeno/issues)

Open-source node system framework, to change your algorithmic code into useful tools to create much more complicated simulations!

<img src="http://zenustech.oss-cn-beijing.aliyuncs.com/Place-in-Github/WelcomePic2.jpg" width="640" position="left">

ZENO is an open-source, Node based 3D system able to produce cinematic physics effects at High Efficiency, it was designed for large scale simulations and has been tested on complex setups.
Aside of its simulation Tools, ZENO provides necessary visualization nodes for users to import and run simulations if you feel that the current software you are using is too slow.

- [Contributor guidelines](docs/CONTRIBUTING.md)
- [How to build from source](BUILD.md)
- [Introduction on Zeno](docs/introduction.md)
- [Tutorial on the editor (WIP)](docs/node_editor.md)

## Features

Integrated Toolbox, from volumetric geometry process tools (OpenVDB), to state-of-art, commercially robust, highly optimized physics solvers and visualization nodes, and various VFX and simulation solutions based on our nodes (provided by .zsg file in `graphs/` folder).

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

Go to the [release page](https://github.com/zenustech/zeno/releases/), and click Assets -> download `zeno-windows-20xx.x.x.zip` (`zeno-linux-20xx.x.x.tar.gz` for Linux).

Then, extract this archive, and simply run `000_start.bat` (`./000_start.sh` for Linux), then the node editor window will shows up if everything is working well.

## How to play

There are some example graphs in the `graphs/` folder, you may open them in the editor and have fun!
Hint: To run an animation for 100 frames, change the `1` on the top-left of node editor to `100`, then click `Run`.
Also MMB to drag in the node editor, LMB click on sockets to create connections. MMB drag in the viewport to orbit camera, Shift+MMB to pan camera.
More details are available in [our official tutorial](https://zenustech.com/tutorial) and [my video tutorials](https://space.bilibili.com/263032155).

## Bug report

If you find the binary version didn't worked properly or some error message has been thrown on your machine, please let me know by opening an [issue](https://github.com/zenustech/zeno/issues) on GitHub, thanks for you support!


# Developer Build

To build ZENO from source, you need:

- GCC 9+ or MSVC 19.28+, and CMake 3.16+ to build ZENO.
- Qt 5.12+ to build the ZENO Qt editor.
- (Optional) OpenVDB for volume nodes.
- (Optional) Eigen3 for solver nodes.
- (Optional) CUDA 11 for GPU nodes.

> Hint: WSL is not recommended because of its limited GUI and OpenGL support.

- [Click me for detailed build instructions](BUILD.md)


## Install from package manager

Arch Linux users may install Zeno from [AUR](https://aur.archlinux.org):
```bash
yay -S zeno
```
The package is maintained by @archibate.


## Contributors

Thank you to all the people who have already contributed to ZENO!

[![Contributors](https://contrib.rocks/image?repo=zenustech/zeno)](https://github.com/zenustech/zeno/graphs/contributors)

- [Contributor guidelines and helps](docs/CONTRIBUTING.md)


# Miscellaneous

## Write your own extension!

See [`zenustech/zeno_addon_wizard`](https://github.com/zenustech/zeno_addon_wizard) for an example on how to write custom nodes in ZENO.

## Legacy version of Zeno

Currently the [`master`](https://github.com/zenustech/tree/master) branch is for Zeno 2.0.
You may find Zeno 1.0 in the [`legacy`](https://github.com/zenustech/tree/legacy) branch.

## License

ZENO is licensed under the Mozilla Public License Version 2.0, see [LICENSE](LICENSE) for more information.

## Code of Conduct

See [Code of Conduct](docs/code_of_conduct.md).

## Contact us

You may contact us via WeChat:

* @zhxx1987: shinshinzhang

* @archibate: tanh233
