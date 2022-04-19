# Maintainers' guide

This section of documentation is for Zeno maintainers.

## Install Zeno from source (not recommended)

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

## Deploying Zeno (CI/CD)

The deployment is automated via GitHub CI, see `.github/workflows/cmake.yml`.

To trigger it, simply push a commit message containing like:

```bash
git commit -m "[release] some description"
git push
```

Push it, then GitHub CI will do the depolyment automatically for you.

It will create a release with tag, for example, `v2022.4.19` (today's date).
