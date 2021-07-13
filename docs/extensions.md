# ZENO Extensions

ZENO is extensible which means we may write extensions (node libraries) for it.
The source code of all our official extensions are provided in `projects/`.

## Build extensions

For now, official extensions will be built by default when running the
`ALL_BUILD` target of CMake.

### ZenVDB & FastFLIP

Note that the extensions: ZenVDB and FastFLIP are **not built by default**.
You can use
```bash
cmake -B build -DEXTENSION_zenvdb:BOOL=ON -DEXTENSION_FastFLIP:BOOL=ON
```
to enable them.

#### Known issues
```diff
- **The FastFLIP solver we know work well with OpenVDB 7.2.3, and have problem with OpenVDB 8.1.**
```

### GMPM

You need to update git submodules before building @littlemine's GPU MPM.
To do so:
```bash
git submodule update --init --recursive
```
Then:
```bash
cmake -B build -DEXTENSION_gmpm:BOOL=ON
```
to enable it.

## Major dependencies

Building them require some dependencies:

- ZFX (ZenoFX expression wrangler)
  - OpenMP (optional)

- Rigid (bullet3 rigid dynamics)
  - no dependencies!

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

Other extensions are built by default because their dependencies are
self-contained and portable to all platforms.

## Write your own extension!

See ```demo_project/``` for an example on how to write custom nodes in ZENO.

### Installing extensions

To install a node library for ZENO just copy the `.so` or `.dll` files to `zeno/lib/`. See ```demo_project/CMakeLists.txt``` for how to automate this in CMake.
