
cmake_minimum_required(VERSION 3.1.0)

# Require C++14
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set( source_files
    nodes.cpp
)

set (header_files )

# Configure Qt
find_package(QT NAMES Qt6 Qt5 COMPONENTS Core Gui QuickControls2 REQUIRED)
find_package(Qt${QT_VERSION_MAJOR} COMPONENTS Core Gui QuickControls2 REQUIRED)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set_property(DIRECTORY APPEND PROPERTY COMPILE_DEFINITIONS $<$<CONFIG:Debug>:QT_QML_DEBUG>)
set(CMAKE_INCLUDE_CURRENT_DIR ON)
include_directories( ${Qt5Quick_INCLUDE_DIRS} )
include_directories( ${CMAKE_CURRENT_SOURCE_DIR} "../../src" )
include_directories( ${CMAKE_CURRENT_SOURCE_DIR} "../../GTpo/src" )
include_directories( ${CMAKE_CURRENT_SOURCE_DIR} "../../QuickContainers/include" )

set(CMAKE_INCLUDE_CURRENT_DIR ON)
add_executable(sample_nodes ${source_files} nodes.qrc)
target_include_directories(sample_nodes PUBLIC QuickQanava Qt${QT_VERSION_MAJOR}::QuickControls2)
target_link_libraries(sample_nodes QuickQanava QuickContainers Qt${QT_VERSION_MAJOR}::Core Qt${QT_VERSION_MAJOR}::Gui Qt${QT_VERSION_MAJOR}::QuickControls2)

if(WIN32 AND DEPLOY)
    find_program(WINDEPLOYQT_EXECUTABLE NAMES windeployqt HINTS ${QTDIR} ENV QTDIR PATH_SUFFIXES bin)
    add_custom_command(TARGET sample_nodes POST_BUILD
        COMMAND ${WINDEPLOYQT_EXECUTABLE} --qmldir ${CMAKE_CURRENT_SOURCE_DIR} $<TARGET_FILE:sample_nodes> $<$<CONFIG:Debug>:--pdb>)
    add_custom_command(TARGET sample_nodes POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:QuickQanava> $<TARGET_FILE_DIR:sample_nodes>)
endif()
