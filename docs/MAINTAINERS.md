# Maintainers' guide

This section of documentation is for Zeno maintainers.

## Install Zeno from source

For the Linux traditional `make install` style installation, please specify this argument:

```bash
cmake -DZENO_INSTALL_TARGET:BOOL=ON -DCMAKE_INSTALL_PREFIX=/usr/local
```

> Change the `/usr/local` to your preferred install path, for example `/opt/zeno-2022.4.19`.

Next, build and install via CMake:

```bash
cmake --build build
sudo cmake --build build --target install
```

This will install Zeno **globally in your system**.
