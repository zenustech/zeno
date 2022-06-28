# Maintainers' guide

This section of documentation is for Zeno maintainers.

## Translating Zeno editor UI

First update the translation file `zh.ts` using the following command:
```bash
lupdate -recursive ui/zenoedit/ -ts ui/zenoedit/res/languages/zh.ts
```

Then open that `zh.ts` file with Qt Linguist:
```bash
linguist ui/zenoedit/res/languages/zh.ts
```

You can now edit the translations.
Now select an entry, type the correct Chinese translation.
Then press `Ctrl-Enter` to submit your translation.

After finish, press the `Release` in Qt Linguist, you will get `ui/zenoedit/res/languages/zh.qm`.
Now rebuild Zeno, and you will see your updated translations in UI.

This process can be ran for multiple times, `lupdate` won't override the old translations, no worry.

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

To trigger it, simply push a commit message containing `[release]`, for example:

```bash
git commit -m "[release] some description"
git push
```

Push it, then GitHub CI will do the depolyment automatically for you.

It will create a release with tag, for example, `v2022.4.19` (today's date).

## Accelerate compile process

You may install `ccache` to compile faster on Linux.

```bash
sudo apt-get install -y ccache
```

```bash
sudo pacman -S ccache
```

You may install `ninja` (a faster build system than `make`).

```bash
sudo apt-get install -y ninja
```

```bash
sudo pacman -S ninja
```

```bash
pip install ninja
```

Then use `-GNinja` parameter in cmake configuration step:

```bash
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```
