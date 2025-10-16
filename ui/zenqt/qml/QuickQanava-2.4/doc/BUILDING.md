# Building

## Dependencies

- **Qt > 5.12** _is mandatory_ for Qt Quick Shapes support.
- **Google Test** is optional only to build and run tests ![Google Test GitHub](https://github.com/google/googletest).


## Building 

Get the latest QuickQanava sources:

```sh
git clone https://github.com/cneben/QuickQanava
cd QuickQanava
```

Or install as a Git submodule:

```sh
$ git submodule add https://github.com/cneben/QuickQanava
$ git submodule update
```

QuickQanava could be used with either _qmake_ or _CMake_ build configuration system.

| qmake (Qt5)          | CMake (Qt6)       | 
| :---:                | :---:             | 
| Static build, no QML module, all resources are linked statically trough QRC | Installable or embedable, QuickQanava is loaded using a QML module that need to be installed, resources can be linked statically trough QRC | 


Using qmake (**preferred and supported way of integrating QuickQanava**):

1. Open _quickqanava.pro_ in QtCreator.

2. Select a kit, build and launch samples.


:warning: CMake support is "community maintained"

or with (CMake >= 3.5) and Qt Creator:

1. Open _CMakeLists.txt_ in QtCreator.

2. In 'Projects' panel, set DQUICK_QANAVA_BUILD_SAMPLES option to true in CMake configuration panel.

3. Select a kit, build and launch samples.

Or manually in command line using CMake:

```sh
$ cd QuickQanava
$ mkdir build
$ cd build

# IF QT_DIR IS CONFIGURED AND QMAKE IN PATH
$ cmake -DCMAKE_BUILD_TYPE=Release -DQUICK_QANAVA_BUILD_SAMPLES=TRUE -DBUILD_STATIC_QRC=TRUE ..

# IF QT DIR IS NOT CONFIGURED, CONFIGURE KIT MANUALLY
$ cmake -DCMAKE_PREFIX_PATH="/home/b/Qt/5.11.0/gcc_64" -DQT_QMAKE_EXECUTABLE="/home/b/Qt/5.11.0/gcc_64/bin/qmake"  -DQUICK_QANAVA_BUILD_SAMPLES=TRUE -DBUILD_STATIC_QRC=TRUE ../QuickQanava/

$ cmake --build .
# Then run the samples in ./samples

# Eventually make install
```

Detailed instructions:  [Installation](http://cneben.github.io/QuickQanava/installation/index.html)

Note that a previously installed "QML plugin" version of QuickQanava might interfere with a fully static build using direct .pri inclusion. Typical error message looks like:

```
QQmlApplicationEngine failed to load component
qrc:/nodes.qml:33 module "QuickQanava" plugin "quickqanavaplugin" not found
```

QuickQanava and QuickContainers plugins directories could be removed manually from `$QTDIR\..\qml` to fix the problem (ex: rm -rf '~/Qt/5.11.1/gcc_64/qml/QuickQanava').

On Ubuntu system, the following dependencies must be isntalled:
```
$ sudo apt install libopengl0 -y
```
