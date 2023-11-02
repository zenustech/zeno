
cmake_minimum_required(VERSION 3.1.0)

project(sample_toplogy VERSION 0.11.0 LANGUAGES CXX)

# Require C++14
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(source_files
    topology.cpp
    qanFaceNode.cpp
)

set (header_files qanFaceNode.h)

# Configure Qt
find_package(QT NAMES Qt6 Qt5 COMPONENTS Core Gui QuickControls2 Widgets REQUIRED)
find_package(Qt${QT_VERSION_MAJOR} COMPONENTS Core Gui QuickControls2 Widgets REQUIRED)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set_property(DIRECTORY APPEND PROPERTY COMPILE_DEFINITIONS $<$<CONFIG:Debug>:QT_QML_DEBUG>)
set(CMAKE_INCLUDE_CURRENT_DIR ON)
include_directories( ${Qt5Quick_INCLUDE_DIRS} )
include_directories( ${CMAKE_CURRENT_SOURCE_DIR} "../../src" )

if(NOT TARGET QuickQanava)
    set(QUICK_QANAVA_BUILD_STATIC_QRC ON)
    add_subdirectory(../../ quickqanava_build)  # Use ../../src as quickqanava_build subdirectory
    # see https://stackoverflow.com/questions/50408169/cmake-error-add-subdirectory-not-given-a-binary-directory/50408555
endif()

set(CMAKE_INCLUDE_CURRENT_DIR ON)

add_executable(sample_topology ${source_files} topology.qrc)
target_include_directories(sample_topology PUBLIC QuickQanava)
target_link_libraries(sample_topology
    QuickQanava
    QuickContainers
    Qt${QT_VERSION_MAJOR}::Core
    Qt${QT_VERSION_MAJOR}::Gui
    Qt${QT_VERSION_MAJOR}::QuickControls2
    Qt${QT_VERSION_MAJOR}::Widgets
    )
