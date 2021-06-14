# Projects

## Major Dependencies

- FastFLIP
  - OpenVDB
  - IlmBase (or OpenEXR)
  - Eigen3
  - TBB

- Zenbase
  - OpenMP C++ (optional)

- Zenvdb
  - OpenVDB
  - IlmBase (or OpenEXR)
  - TBB
  - OpenMP C++ (optional)

- ZMS
  - OpenMP C++ (optional)

## Miscellaneous

### Build in Windows 
For now (up to June 13th, 2021) the following official node libraries are known to be not directly compilable under Windows 10 with Visual Studio 2019 Community Edition:

- zenvdb
- FastFLIP

If only the compilable libraries are needed, try the following before compilation starts:

In `Projects/CMakeLists.txt`, comment the following lines:

```
add_subdirectory(zenvdb)
add_subdirectory(FastFLIP)
```
