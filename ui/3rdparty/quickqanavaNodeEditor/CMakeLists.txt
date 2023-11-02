cmake_minimum_required(VERSION 3.5.0)

project(QuickQanava VERSION 2.3.0 LANGUAGES CXX)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

add_compile_definitions(QT_DISABLE_DEPRECATED_BEFORE=0x050F00)

find_package(QT NAMES Qt6 Qt5 COMPONENTS Core Quick REQUIRED)
find_package(Qt${QT_VERSION_MAJOR} COMPONENTS Core Qml Quick QuickControls2 REQUIRED)
message("Building with Qt" ${QT_VERSION_MAJOR})

option(QUICK_QANAVA_BUILD_SAMPLES "Build the samples" TRUE)
option(QUICK_QANAVA_BUILD_STATIC_QRC "Build *.qrc resources statically" TRUE)
option(QUICK_QANAVA_WINDOWS_DEPLOY "Use windeployqt on Windows" FALSE)

add_subdirectory(QuickContainers)
add_subdirectory(src)

if (${QUICK_QANAVA_BUILD_SAMPLES})
    #add_subdirectory(samples/resizer)
    #add_subdirectory(samples/navigable)
    add_subdirectory(samples/nodeEditor)
    add_subdirectory(samples/groups)
    add_subdirectory(samples/connector)
    add_subdirectory(samples/cpp)
    add_subdirectory(samples/edges)
    add_subdirectory(samples/navigable)
    add_subdirectory(samples/nodes)
    add_subdirectory(samples/style)
    add_subdirectory(samples/topology)
    add_subdirectory(samples/dataflow)
endif()
