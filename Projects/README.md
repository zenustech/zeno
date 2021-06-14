# Projects

## Major Dependencies

- FastFLIP
  - todo

- Zenbase
  - todo

- Zenvdb
  - todo

- ZMS
  - todo

## Miscellaneous

### Build in Windows 
For now (up to June 13th, 2021) the following official node libraries are known to be not directly compilable under Windows 10 with Visual Studio 2019 Community Edition:

- zenvdb
- FastFLIP

If only the compilable libraries are needed, try the following before compilation starts:

In `/Projects/CMakeLists.txt`, comment the following lines:

```
add_subdirectory(zenvdb)
add_subdirectory(FastFLIP)
```