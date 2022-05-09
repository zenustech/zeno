# Troubleshooting Build

You may find solutions here for known issues.

If you didn't find your problem in the following list, please consider submit a
[GitHub issue](https://github.com/zenustech/zeno/issues) so that we can help you solve it.

If you solved the problem yourself, you may create a pull Request to edit this file,
adding your own solution (according to the existing format), so that later people with same
problem can find solution easier.

# CMake problem

## Q

Almost 99% of CMake problems.

## A

CMake is a stupid state machine, the 'cached' configuration values from previous build may break the next build.

Try remove the `build` directory completely, and re-run `cmake -B build` to re-build:

```bash
rm -rf build
```

# Windows problem

# WSL problem

## Q

```
qt.qpa.xcb: could not connect to display
qt.qpa.plugin: Could not load the Qt platform plugin "xcb" in "" even though it was found.
This application failed to start because no Qt platform plugin could be initialized. Reinstalling the application may fix this problem.

Available platform plugins are: eglfs, linuxfb, minimal, minimalegl, offscreen, vnc, xcb.

Aborted (core dumped)
```

when running `build/bin/zenoedit`.

## A

This means your Linux system don't have a graphic interface (GUI).
You might be using WSL, **WSL don't have GUI by default**. They only have command line interface but no graphic interface.
So all GUI application won't work in your WSL. This is *not a Zeno bug*, but a *microsoft bug*.

My opinion: **never use WSL, Docker, or virtual machines**, they have extremely poor graphic support (no GPU).
For best performance, why not use native Windows instead? Zeno already have full-support for Windows years ago.
If you insist to use WSL, here's the solution:

1. You need to install X-Launch (or vcxsrv): https://sourceforge.net/projects/xlauncher/
2. Start X-Launch (select the multi-window mode if it asks).
3. Execute `export DISPLAY=:0` in the WSL terminal.
4. Now run Zeno again, its GUI window should shows up successfully.

> If you are WSL2, may need `export DISPLAY=192.168.xx.xx:0` instead, where `192.168.xx.xx` should be your local IP address.
> Some people says that WSL2 now have graphic interface by default, I'm not sure. I never used WSL2.

To test if your GUI is working, use the `xclock` (a small GUI application showing a clock):

```bash
sudo apt-get install -y xclock
xclock
```

* If the `xclock` failed to start: your WSL still doesn't have GUI, **all GUI application won't work**, not just Zeno.
* If the `xclock` starts successfully: configurations for a working GUI, go ahead and Zeno should work too.

> Not WSL but still meet this problem? You might be using Zeno on a headless-server, hint: `ssh -X root@yourserver.address`

# Ubuntu problem

## Q

```
/usr/include/c++/9/exception:143:10: fatal error: /mnt/e/zeno/build/ui/zenoui/zenoui_autogen/include/bits/exception_ptr.h: Invalid argument
  143 | #include <bits/exception_ptr.h>
      |          ^~~~~~~~~~~~~~~~~~~~~~
```

during `cmake --build build`.

## A

This is because your Qt5 and GCC libs version mismatched, upgrade them:

```bash
sudo apt-get update
sudo apt-get upgrade
rm -rf build           # clean cmake build directory and re-run cmake
```

## Q

```
/usr/lib/qt5/bin/uic: error while loading shared libraries: libQt5Core.so.5: cannot open shared object file: No such file or directory
```

during `cmake --build build`.

## A

This is a silly bug from Ubuntu official team, you need manually run this command to fix the stupid `libQt5Core.so.5`:

```bash
sudo strip --remove-section=.note.ABI-tag /usr/lib64/libQt5Core.so.5
```

Reference: https://askubuntu.com/questions/1034313/ubuntu-18-4-libqt5core-so-5-cannot-open-shared-object-file-no-such-file-or-dir
