# Maintainers' manual

## Run intergrated test

### Linux

```bash
cmake -B build -DZENO_BUILD_TESTS:BOOL=ON
cmake --build build --parallel
build/tests/zentest
```

### Windows

```cmd
cmake -B build -DZENO_BUILD_TESTS:BOOL=ON
cmake --build build --parallel
build\tests\zentest.exe
```

## Build binary release

- Linux
```bash
sudo python3 -m pip install wheel
sudo python3 -m pip install pyinstaller
./dist.py
```

- Windows
```bash
python -m pip install wheel
python -m pip install pyinstaller
python dist.py
```

You will get dist/launcher.zip, upload it to, for example, zeno-linux-2021.8.7.zip in the `Release` page.

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
