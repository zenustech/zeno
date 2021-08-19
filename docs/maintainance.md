# Maintainers' manual

## Run intergrated test

### Linux

```bash
cmake -B build -DZENO_BUILD_TESTS:BOOL=ON
cmake --build build --parallel
zenqt\bin\zentest
```

### Windows

```cmd
cmake -B build -DZENO_BUILD_TESTS:BOOL=ON
cmake --build build --parallel
zenqt\bin\zentest.exe
```

## Build binary release

### CentOS 7

```bash
sudo python3 -m pip install wheel
sudo python3 -m pip install pyinstaller
./dist.py
```

> NOTE: Linux binary releases should always be done in CentOS for GLIBC version compatibilty on older systems.

### Windows 10

```bash
python -m pip install wheel
python -m pip install pyinstaller
python dist.py
```

You will get `dist/zeno-windows-2021.x.xx.zip`, now upload it to the `Release` page.

## Build binary release (old method)

### Arch Linux

```bash
scripts/dist.sh
# you will get /tmp/release/zeno-linux-20xx.x.x.tar.gz
```

### Windows

First, download `zenv-windows-prebuilt.zip` from [this page](https://github.com/zenustech/binaries/releases).
Second, extract it directly into project root.
Then run `scripts/dist.bat` in project root.
Finally, rename the `zenv` folder to `zeno-windows-20xx.x.x`, and archive it into `zeno-windows-20xx.x.x.zip`.
