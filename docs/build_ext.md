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
