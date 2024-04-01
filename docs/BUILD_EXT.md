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
sudo apt-get install -y libblosc-dev libboost-iostreams-dev zlib1g-dev libtbb-dev libopencv-dev
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
.\vcpkg install blosc:x64-windows
.\vcpkg install tbb:x64-windows
.\vcpkg install boost-uuid:x64-windows boost-interprocess:x64-windows
.\vcpkg install eigen3:x64-windows
.\vcpkg install cgal:x64-windows
.\vcpkg install lapack:x64-windows
.\vcpkg install openblas:x64-windows
.\vcpkg install hdf5:x64-windows
.\vcpkg install opencv4[core,jpeg,png,tiff,webp]:x64-windows
.\vcpkg install cryptopp:x64-windows
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
cmake -B build -DCMAKE_BUILD_TYPE=Release -DZENO_WITH_ZenoFX:BOOL=ON -DZENOFX_ENABLE_OPENVDB:BOOL=ON -DZENOFX_ENABLE_LBVH:BOOL=ON -DZENO_WITH_zenvdb:BOOL=ON -DZENO_WITH_FastFLIP:BOOL=ON -DZENO_WITH_FEM:BOOL=ON -DZENO_WITH_Rigid:BOOL=ON -DZENO_WITH_cgmesh:BOOL=ON -DZENO_WITH_oldzenbase:BOOL=ON -DZENO_WITH_TreeSketch:BOOL=ON -DZENO_WITH_Skinning:BOOL=ON -DZENO_WITH_Euler:BOOL=ON -DZENO_WITH_Functional:BOOL=ON -DZENO_WITH_LSystem:BOOL=ON -DZENO_WITH_mesher:BOOL=ON -DZENO_WITH_Alembic:BOOL=ON -DZENO_WITH_FBX:BOOL=ON -DZENO_WITH_DemBones:BOOL=ON -DZENO_WITH_SampleModel:BOOL=ON -DZENO_WITH_CalcGeometryUV:BOOL=ON -DZENO_WITH_MeshSubdiv:BOOL=ON -DZENO_WITH_Audio:BOOL=ON -DZENO_WITH_PBD:BOOL=ON -DZENO_WITH_GUI:BOOL=ON -DZENO_WITH_ImgCV:BOOL=ON
```

> See also `misc/run.sh` (you can use this script instead for the full-featured build on Linux).

### Enabling CUDA extensions

NVIDIA users may additionally specify `-DZENO_WITH_CUDA:BOOL=ON -DZENO_ENABLE_OPTIX:BOOL=ON` in arguments for building CUDA support.

> This will also builds the OptiX real-time ray-tracing for the Zeno renderer (RTX20xx above required).

Notice that **CUDA 11.6 (or above) is requried**, thanks to @littlemine's modern-fancy-cuda skills :(

> But if you only use the OptiX part, Simply CUDA 11 is enough, thanks to @zhxx1987 not using modern-fancy-cuda features :)

> NOTE: The CUDA extension is work in progress, may not work.
> NOTE: ZenoFX must be enabled when CUDA is enabled, because CUDA depends on ZenoFX.
> NOTE: Windows user must install the `CUDA Visual Studio integration`, otherwise CMake will complains `No CUDA toolset found`.

### Enabling tool extensions

Some of the extensions are purely made with Zeno subgraphs, they lays in the directory
`projects/tools` and their contents are basically hard-encoded subgraph JSON strings.
To enable them, just additionally specify `-DZENO_WITH_TOOL_FLIPtools:BOOL=ON -DZENO_WITH_TOOL_cgmeshTools:BOOL=ON -DZENO_WITH_TOOL_BulletTools:BOOL=ON -DZENO_WITH_TOOL_HerculesTools:BOOL=ON`.
Enabling them you will find our well-packaged high-level nodes like `FLIPSimTemplate`,
they were exported from another subgraph file using Ctrl-Shfit-E by the way, see the
source code of `FLIPtools` for the original graph file name.

## Enabling the Python extension

You may optionally enable the embedded Python interpreter extension for Zeno by specifying `-DZENO_WITH_python:BOOL=ON` in arguments.

> Note that we already embed a [modified version CPython](https://github.com/zenustech/python-cmake-buildsystem) in this repository (as a submodule), you don't need to pre-install Python in your system at all.

> The default version is 3.9.10, you may specify e.g. `-DPYTHON_VERSION=3.9.9` for using different version.

### Ubuntu

These packages are required to build the Python extension on Ubuntu:

```bash
sudo apt-get install -y build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev libsqlite3-dev wget libbz2-dev
```

## Advanced build configuration (optional)

Here we introduce some CMake options that professional users may need.

1. To enable *stack traceback* in Zeno when fault encountered (OFF by default):

```bash
cmake -B build -DZENO_ENABLE_BACKWARD:BOOL=ON
```

2. To enable *parallel STL* in Zeno for better performance (OFF by default):

```bash
cmake -B build -DZENO_PARALLEL_STL:BOOL=ON
```

> This would require `apt-get install libtbb-dev` on Linux (GCC), while Windows (MSVC) doesn't need to do anything.

3. To disable *OpenMP* in Zeno to prevent multi-threading (ON by default):

```bash
cmake -B build -DZENO_ENABLE_OPENMP:BOOL=OFF
```

4. To enable `-march=native` in Zeno to utilize native instruction set (OFF by default):

```bash
cmake -B build -DZENO_MARCH_NATIVE:BOOL=ON
```

> WARN: This would make the binary unable to deploy to other machine lower than yours.
> Suppose you have AVX512, but your customer doesn't have AVX512. Then if you copy your binary to them, they will get `Illegal Instruction`.

5. To enable `-ffast-math` in Zeno for fast but non-IEEE-compatibile math (OFF by default):

```bash
cmake -B build -DZENO_FAST_MATH:BOOL=ON
```

> WARN: May have inpredictible behavior when dealing with NaN and Infinity.

6. To disable *multi-processing* in Zeno editor to make it easier to debug (ON by default):

```bash
cmake -B build -DZENO_MULTIPROCESS:BOOL=OFF
```

7. To not build the *Zeno editor* (which requires Qt), but only the *Zeno core* (ON by default):

```bash
cmake -B build -DZENO_BUILD_EDITOR:BOOL=OFF
```

## What's next?

If you are the project maintainer, you may also checkout [`docs/MAINTAINERS.md`](/docs/MAINTAINERS.md) for even more advanced skills.
