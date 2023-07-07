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

[Download](https://github.com/zenustech/zeno/releases) | [Repo](https://github.com/zenustech/zeno) | [About us](https://zenustech.com) | [Docs](https://doc.zenustech.com/) | [Videos](https://space.bilibili.com/263032155) | [Q&A Forum](https://github.com/zenustech/zeno/discussions) | [Build from source](https://github.com/zenustech/zeno/blob/master/BUILD.md) | [FAQs](https://github.com/zenustech/zeno/blob/master/docs/FAQ.md) | [Contributor Guidelines](https://github.com/zenustech/zeno/blob/master/docs/CONTRIBUTING.md) | [Bug report](https://github.com/zenustech/zeno/issues)

[国内高速下载](https://zenustech.com/d/) | [Gitee 镜像仓库](https://gitee.com/zenustech/zeno) | [公司主页](https://zenustech.com) | [中文文档](https://doc.zenustech.com/) | [视频教程](https://space.bilibili.com/263032155) | [问答论坛](https://github.com/zenustech/zeno/discussions) | [从源码构建](https://github.com/zenustech/zeno/blob/master/BUILD.md) | [常见问题](https://github.com/zenustech/zeno/blob/master/docs/FAQ.md) | [贡献者指南](https://github.com/zenustech/zeno/blob/master/docs/CONTRIBUTING.md) | [BUG 反馈](https://github.com/zenustech/zeno/issues)

Open-source node system framework, to change your algorithmic code into useful tools to create much more complicated simulations!

<img src="https://zenustech.oss-cn-beijing.aliyuncs.com/Place-in-Github/202304/zeno_screenshot.png" width="640" position="left">

ZENO is an open-source, Node based 3D system able to produce cinematic physics effects at High Efficiency, it was designed for large scale simulations and has been tested on complex setups.
Aside of its simulation Tools, ZENO provides necessary visualization nodes for users to import and run simulations if you feel that the current software you are using is too slow.

- [Contributor guidelines](docs/CONTRIBUTING.md)
- [How to build from source](BUILD.md)
- [FAQ & troubleshooting](docs/FAQ.md)
- [Introduction on Zeno](docs/introduction.md)
- [Video tutorial series](https://space.bilibili.com/263032155)

## Features

Integrated Toolbox, from volumetric geometry process tools (OpenVDB), to state-of-art, commercially robust, highly optimized physics solvers and visualization nodes, and various VFX and simulation solutions based on our nodes (provided by .zsg file in `graphs/` folder).

## New

Multi Importance Sampling

<img src="https://zenustech.oss-cn-beijing.aliyuncs.com/Place-in-Github/202307/multi_importace_sampling.jpg" width="640" position="left">

## Gallery

Fig.1 - Cloth simulation

<img src="https://zenustech.oss-cn-beijing.aliyuncs.com/Place-in-Github/202304/cloth.gif" width="640" position="left">

Fig.2 - Fluid simulation

<img src="https://zenustech.oss-cn-beijing.aliyuncs.com/Place-in-Github/202304/flip.png" width="640" position="left">
<img src="https://zenustech.oss-cn-beijing.aliyuncs.com/Place-in-Github/202304/liulang.gif" width="640" position="left">

Fig.3 - Rigid simulation

<img src="https://zenustech.oss-cn-beijing.aliyuncs.com/Place-in-Github/202208/Bullet_Simulation.gif" width="640" position="left">

Fig.4 - Biological simulation

<img src="https://zenustech.oss-cn-beijing.aliyuncs.com/Place-in-Github/202208/Biological_Simulation.gif" width="640" position="left">

Fig.5 - Procedural material

<img src="https://zenustech.oss-cn-beijing.aliyuncs.com/Place-in-Github/202208/Procedural_Material.gif" width="640" position="left">

Fig.6 - Procedural modeling

<img src="https://zenustech.oss-cn-beijing.aliyuncs.com/Place-in-Github/202304/programmatic.gif" width="640" position="left">

Fig.7 - Human rendering

<img src="https://zenustech.oss-cn-beijing.aliyuncs.com/Place-in-Github/202304/face.png" width="640" position="left">


https://user-images.githubusercontent.com/25457920/234779878-a2f43b2f-5b9b-463b-950b-8842dad0c651.MP4



# End-user Installation

## Download binary release

Go to the [release page](https://github.com/zenustech/zeno/releases/), and click Assets -> download `zeno-windows-20xx.x.x.zip` (`zeno-linux-20xx.x.x.tar.gz` for Linux).

Then, extract this archive, and simply run `000_start.bat` (`./000_start.sh` for Linux), then the node editor window will shows up if everything is working well.

Apart from the GitHub release page, we also offer binary download from our official site for convinence of Chinese users: https://zenustech.com/d/

## How to play

There are some example graphs in the `misc/graphs/` folder, you may open them in the editor and have fun!
Hint: To run an animation for 100 frames, change the `1` on the bottom-right of the viewport to `100`, then click `Run`.
Also MMB to drag in the node editor, LMB click on sockets to create connections.
MMB drag in the viewport to orbit camera, Shift+MMB to pan camera.
More details are available in [our official tutorial](https://doc.zenustech.com/) and [my video tutorials](https://space.bilibili.com/263032155).

## Bug report

If you find the binary version didn't worked properly or some error message has been thrown on your machine, please let me know by opening an [issue](https://github.com/zenustech/zeno/issues) on GitHub, thanks for you support!


# Developer Build

To build ZENO from source, you need:

- GCC 9+ or MSVC 19.28+, and CMake 3.16+ to build ZENO.
- Qt 5.14+ to build the ZENO Qt editor.
- (Optional) TBB for parallel support.
- (Optional) OpenVDB for volume nodes.
- (Optional) Eigen3 for solver nodes.
- (Optional) CGAL for geometry nodes.
- (Optional) CUDA 11.6 for GPU nodes.

> Hint: WSL is not recommended because of its limited GUI and OpenGL support.

- [Click me for detailed build instructions](BUILD.md)


# Miscellaneous

## Contributors

Thank you to all the people who have already contributed to ZENO!

[![Contributors](https://contrib.rocks/image?repo=zenustech/zeno)](https://github.com/zenustech/zeno/graphs/contributors)

- [Contributor guidelines and helps](docs/CONTRIBUTING.md)

## Write your own extension!

See [`projects/FBX`](https://github.com/zenustech/zeno/projects/FBX) for an example on how to write custom nodes in ZENO.

## Legacy version of Zeno

Currently the [`master`](https://github.com/zenustech/tree/master) branch is for Zeno 2.0.
You may find Zeno 1.0 in the [`zeno_old_stable`](https://github.com/zenustech/tree/zeno_old_stable) branch.

## License

ZENO is licensed under the Mozilla Public License Version 2.0, see [LICENSE](LICENSE) for more information.

ZENO have also used many third-party libraries, some of which has little modifications. Their licenses could be found at [docs/licenses](docs/licenses).

## Contact us

You may contact us via WeChat:

* @zhxx1987: shinshinzhang

* @archibate: tanh233

... or sending E-mail:

* @archibate: pengyb@zenustech.com

Jobs offering: zenustech.com/jobs
