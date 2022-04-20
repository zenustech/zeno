# Building Zeno extensions

You have just successfully built the plain Zeno with no extensions.

However Zeno has many useful extension modules in the `projects/` folder. Despite the
great power of these extensions, they require a lot of dependencies like OpenVDB,
therefore a very complicated setup is needed.

### Enable extensions by CMake arguments

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
pacman -S tbb openvdb eigen3 cgal lapack openblas alembic
```

### Ubuntu

```bash
sudo apt-get install -y libtbb-dev libopenvdb-dev libeigen3-dev libcgal-dev liblapack-dev libopenblas-dev

# Install alembic from source:
sudo apt-get install -y libhdf5-dev
git clone https://github.com/zenustech/alembic.git --depth=1
cd alembic
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 4
sudo cmake --build build --target install
cd ..
```

> It's highly recommended to use Ubuntu 20.04 or above, otherwise you have to build some libraries from source.

> If you install some libraries from source, it will be likely installed in the `/usr/local` directory.
> Say OpenVDB for example, CMake only searches `/usr/lib/cmake/OpenVDB/FindOpenVDB.cmake`.
> It will not find your actual OpenVDB in `/usr/local/lib/cmake/OpenVDB/FindOpenVDB.cmake`.
> you may need to specify `-DOpenVDB_DIR=/usr/local/lib/cmake/OpenVDB` in this case.

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
.\vcpkg install tbb:x64-windows
.\vcpkg install openvdb:x64-windows
.\vcpkg install eigen3:x64-windows
.\vcpkg install cgal:x64-windows
.\vcpkg install lapack:x64-windows
.\vcpkg install openblas:x64-windows
.\vcpkg install alembic[hdf5]:x64-windows
.\vcpkg install libigl:x64-windows
```

> Notice that you must install the `English Pack` for VS2019 for vcpkg to work. This can be done by clicking the `Language` panel in the VS2019 installer. (JOKE: the maintainer of vcpkg speaks Chinese too)

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
cmake -B build -DCMAKE_BUILD_TYPE=Release -DZENO_WITH_ZenoFX:BOOL=ON -DZENOFX_ENABLE_OPENVDB:BOOL=ON -DZENOFX_ENABLE_LBVH:BOOL=ON -DZENO_WITH_zenvdb:BOOL=ON -DZENO_WITH_FastFLIP:BOOL=ON -DZENO_WITH_FEM:BOOL=ON -DZENO_WITH_Rigid:BOOL=ON -DZENO_WITH_cgmesh:BOOL=ON -DZENO_WITH_oldzenbase:BOOL=ON -DZENO_WITH_TreeSketch:BOOL=ON -DZENO_WITH_Skinning:BOOL=ON -DZENO_WITH_Euler:BOOL=ON -DZENO_WITH_Functional:BOOL=ON -DZENO_WITH_LSystem:BOOL=ON -DZENO_WITH_Alembic:BOOL=ON -DZENO_WITH_gmpm:BOOL=ON -DZENO_WITH_mesher:BOOL=ON
```

> See also `misc/run.sh`.

### Enabling GPU extensions

NVIDIA users may additionally specify `-DZENO_WITH_gmpm:BOOL=ON -DZENO_WITH_mesher:BOOL=ON` in arguments for building CUDA support.

> NOTE: **CUDA 11.x requried**.

> NOTE: ZenoFX must be enabled when gmpm is enabled, because gmpm depends on ZenoFX.
