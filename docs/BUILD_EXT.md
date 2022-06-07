# Building Zeno extensions

You have just successfully built the plain Zeno with no extensions.

However Zeno has many useful extension modules in the `projects/` folder. Despite the
great power of these extensions, they require a lot of dependencies like OpenVDB,
therefore a very complicated setup is needed.

## Fetch submodules

You may need to get the submodules of Zeno to build some of our extension modules:

```bash
git submodule update --init --recursive
```

> The submodules are really huge (~40MB), if you find it too slow, you may give up for now and try it later. Zeno can still build without these submodules.

> Hint: `git submodule update --init --recursive` should also be executed every time you `git pull` (when you'd like to synchronize with latest updates).

> If you find GitHub slow: edit `.gitmodules` and replace GitHub URLs by your corresponding [Gitee](https://gitee.com) mirrors, and re-run the above command.

## Enable extensions by CMake arguments

Due to the complexity of extension modules, the are **not built by default** to be
friendly to new users. You may manually enable building them by passing CMake arguments.
For example, to enable the ZenoFX extension in the `projects/ZenoFX` folder:

```bash
cmake -B build -DZENO_WITH_ZenoFX:BOOL=ON
```

> If you are using IDE, specify `-DZENO_WITH_ZenoFX:BOOL=ON` in its CMake settings panel.
> To disable this extension, specify `-DZENO_WITH_ZenoFX:BOOL=OFF` again instead.

Next, run the build step again:

```bash
cmake --build build --config Release --parallel 4
```

> If you meet any CMake problems, try `rm -rf build` and re-run all the steps.

Actually ZenoFX is a dependency-free extension of Zeno, it only depends on `x86_64` architecture.
If you succeed to build ZenoFX, let's move on to build other more complicated extensions.

## Getting dependencies of extensions

### Arch Linux (recommended)

```bash
pacman -S tbb blosc boost zlib eigen cgal lapack openblas hdf5
```

### Ubuntu

```bash
sudo apt-get install -y libblosc-dev libboost-iostreams-dev zlib1g-dev libtbb-dev
sudo apt-get install -y libeigen3-dev libcgal-dev liblapack-dev libopenblas-dev libhdf5-dev
```

> It's highly recommended to use **Ubuntu 20.04 or above** for getting latest version of libraries.

> OpenVDB 9.0.0 is now bundled as a submodule of Zeno, so no need to install `libopenvdb-dev`
> (which is only 7.2.0) from `apt` anymore. But as a result we need to install the dependencies
> of OpenVDB like the `blosc`, `zlib`, `tbb`, and `boost` here.

### Windows

We (have no choice but to) use [`vcpkg`](https://github.com/microsoft/vcpkg) as package manager on Windows.

It requires fast GitHub connections to work, and may **randomly fail**. If you meet trouble
about `vcpkg`, please [creating an issue](https://github.com/microsoft/vcpkg/issues) in the
`vcpkg` repository (instead of `zeno`), they will help you resolve it.

```bash
cd C:\
git clone https://github.com/microsoft/vcpkg.git --depth=1
cd vcpkg
.\bootstrap-vcpkg.bat
.\vcpkg integrate install
.\vcpkg install zlib:x64-windows
.\vcpkg install blosc:x64-windows
.\vcpkg install tbb:x64-windows
.\vcpkg install boost-iostreams:x64-windows
.\vcpkg install boost-any:x64-windows
.\vcpkg install boost-algorithm:x64-windows
.\vcpkg install boost-uuid:x64-windows
.\vcpkg install boost-interprocess:x64-windows
.\vcpkg install eigen3:x64-windows
.\vcpkg install cgal:x64-windows
.\vcpkg install lapack:x64-windows
.\vcpkg install openblas:x64-windows
.\vcpkg install hdf5:x64-windows
```

> Notice that you must install the `English Pack` for VS2019 for vcpkg to work. This can be done by clicking the `Language` panel in the VS2019 installer. (JOKE: the maintainer of vcpkg speaks Chinese too..)

> Chinese users may also need to follow the instruction in [this zhihu post](https://zhuanlan.zhihu.com/p/383683670) to **switch to domestic source** for faster download.

> See also [their official guide](https://github.com/microsoft/vcpkg/blob/master/README_zh_CN.md) for other issues.

Then, please **delete the `build` directory completely** if you have previous builds,
this is to refresh the CMake cache otherwise it won't find these packages.

Now, get back to `zeno`, run a fresh CMake *configure* phase with the following argument:

```bash
cd C:\zeno
cmake -B build -DCMAKE_TOOLCHAIN_FILE="C:/vcpkg/scripts/buildsystems/vcpkg.cmake"
```

> Please replace the `C:/vcpkg` to your location of `vcpkg` you have just cloned.
> Also make sure you replace `\\` with `/` since CMake doesn't recognize `\\`.
> Delete the `build` directory completely and re-run if you meet CMake problems.

## Building all extensions

The full-featured version of Zeno can be built as follows:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DZENO_WITH_ZenoFX:BOOL=ON -DZENOFX_ENABLE_OPENVDB:BOOL=ON -DZENOFX_ENABLE_LBVH:BOOL=ON -DZENO_WITH_zenvdb:BOOL=ON -DZENO_WITH_FastFLIP:BOOL=ON -DZENO_WITH_FEM:BOOL=ON -DZENO_WITH_Rigid:BOOL=ON -DZENO_WITH_cgmesh:BOOL=ON -DZENO_WITH_oldzenbase:BOOL=ON -DZENO_WITH_TreeSketch:BOOL=ON -DZENO_WITH_Skinning:BOOL=ON -DZENO_WITH_Euler:BOOL=ON -DZENO_WITH_Functional:BOOL=ON -DZENO_WITH_LSystem:BOOL=ON -DZENO_WITH_mesher:BOOL=ON -DZENO_WITH_Alembic:BOOL=ON -DZENO_WITH_FBX:BOOL=ON
```

> See also `misc/run.sh` (you can use this script instead for the full-featured build).

### Enabling CUDA extensions

NVIDIA users may additionally specify `-DZENO_WITH_gmpm:BOOL=ON` in arguments for building CUDA support.

> NOTE: **CUDA 11.x requried**.
> NOTE: `gmpm` is work in progress, may not work.
> NOTE: ZenoFX must be enabled when gmpm is enabled, because gmpm depends on ZenoFX.

### Enabling OptiX extensions

NVIDIA users may also additionally specify `-DZENO_ENABLE_OPTIX:BOOL=ON` in argument for building OptiX real-time ray-tracing for the Zeno renderer.

You can download and install OptiX from this site: https://developer.nvidia.com/designworks/optix/download

Then, please define the path to your OptiX installation, for example: `-DOPTIX_PATH=/home/bate/Downloads/NVIDIA-OptiX-SDK-7.4.0-linux64-x86_64`

### Enabling subgraph extensions

Some of the extensions are purely made with Zeno subgraphs, they lays in the directory
`projects/tools` and their contents are basically hard-encoded subgraph JSON strings.
To enable them, just specify `-DZENO_WITH_TOOL_FLIPtools:BOOL=ON -DZENO_WITH_TOOL_cgmeshTools:BOOL=ON -DZENO_WITH_TOOL_BulletTools:BOOL=ON -DZENO_WITH_TOOL_HerculesTools:BOOL=ON`.
Enabling them you will find our well-packaged high-level nodes like `FLIPSimTemplate`,
they were exported from another subgraph file using Ctrl-Shfit-E by the way, see the
source code of `FLIPtools` for the original graph file name.

<!-- deprecated, see misc/ci/CMakePresets.json, now only used by CI
### Using CMake presets (experimental)

Latest version of CMake supports `CMakePresets.json` and `--preset`, so you may use the following command instead of above huge command lines:

```bash
cmake --preset default
cmake --build --preset default
```

And for people who would like to build with CUDA support:

```bash
cmake --preset cuda
cmake --build --preset cuda
```

The `default` or `cuda` here is called the preset name, see `CMakePresets.json` at the root of project directory for more presets and their details.

Note that you may still specify extra arguments under preset mode, for example:

```bash
cmake --preset default -G Ninja -DCMAKE_INSTALL_PREFIX:BOOL=/opt/zeno
cmake --build --preset default --parallel
```
-->
