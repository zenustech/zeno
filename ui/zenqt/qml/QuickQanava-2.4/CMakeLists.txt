cmake_minimum_required(VERSION 3.5.0)

project(QuickQanava VERSION 2.4.0 LANGUAGES CXX)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(QT NAMES Qt6 Qt5 REQUIRED COMPONENTS Core Quick Qml Quick QuickControls2)

message("Building QuickQanava for Qt${QT_VERSION_MAJOR}")

option(QUICK_QANAVA_BUILD_SAMPLES "Build the samples" FALSE)

if (${QT_VERSION_MAJOR} EQUAL 5)
    include(CMakeLists.qt5.txt)
else()
    include(CMakeLists.qt6.txt)
endif()

add_subdirectory(src)
