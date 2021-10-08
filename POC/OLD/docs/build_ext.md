# Building ZENO Extensions

ZENO is extensible which means we may write extensions (node libraries) for it.
The source code of all our official extensions are provided in `projects/`.

For now, official extensions will be built by default when running the
```ALL_BUILD``` target of CMake.

But some of the extensions are not **enabled by default** because they requries optional dependencies, don't worry, you can enable them with the following commands:

## Building Rigid

```bash
cmake -B build -DEXTENSION_Rigid:BOOL=ON
```

## Building ZenVDB & FastFLIP

```bash
cmake -B build -DEXTENSION_zenvdb:BOOL=ON -DEXTENSION_FastFLIP:BOOL=ON -DZENOFX_ENABLE_OPENVDB:BOOL=ON
```

> **The FastFLIP solver we know work well with OpenVDB 7.2.3, and have problem with OpenVDB 8.1.**

## Building GMPM & Mesher

```bash
# update git submodules to fetch @littlemine's ZPC submodule:
git submodule update --init --recursive
cmake -B build -DEXTENSION_gmpm:BOOL=ON -DEXTENSION_mesher:BOOL=ON
```

## Building Euler

```bash
cmake -B build -DEXTENSION_Euler:BOOL=ON
```

## Major dependencies

Building them require some dependencies:

- ZenoFX (ZFX expression wrangler)
  - OpenMP (optional)
  - OpenVDB (optional)

- Rigid (bullet3 rigid dynamics)
  - OpenMP

- ZMS (molocular dynamics)
  - OpenMP (optional)

- OldZenBASE (deprecated mesh operations)
  - OpenMP (optional)

- ZenVDB (OpenVDB ops and tools)
  - OpenVDB
  - IlmBase
  - TBB
  - OpenMP (optional)

- FastFLIP (OpenVDB FLIP solver)
  - OpenVDB
  - IlmBase
  - Eigen3
  - TBB
  - OpenBLAS
  - ZenVDB (see above)
  - OldZenBASE (see above)
  - OpenMP (optional)

- GMPM (GPU MPM solver)
  - CUDA toolkit
  - OpenVDB (optional)
  - OpenMP (optional)

- Mesher (MPM Meshing)
  - Eigen3
  - OpenMP (optional)

- Euler (aerodynamics solver)
  - OpenVDB
  - IlmBase
  - Eigen3
  - TBB
  - ZenVDB (see above)
  - OldZenBASE (see above)


Other extensions are built by default because their dependencies are
self-contained and portable to all platforms.

## Using `ccmake`

Optional: You can change some cmake configurations using `ccmake`:
```bash
cmake -B build
ccmake -B build  # will shows up a curses screen, c to save, q to exit
```

Below is the suggested Extension Setup:
![extension](images/extension1.png)
![extension](images/extension2.png)

> if you have confidence with your GPU and CUDA version, also turn ON those CUDA related stuffs, see figures below: (change mesher, gmpm, ZS_CUDA, ZFXCUDA to OFF may skip cmake and gpu dependencies issue, while disable you from using GPU computing features)
<img src="images/ccmake1.png" alt="ccmake1" style="zoom:98%;" />
<img src="images/ccmake2.png" alt="ccmake2" style="zoom:50%;" />

> Windows user may use `cmake-gui` instead of `ccmake`.
