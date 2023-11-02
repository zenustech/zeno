cmake_minimum_required(VERSION 3.1.0)

# Configure Qt
find_package(QT NAMES Qt6 Qt5 COMPONENTS QuickControls2 REQUIRED)
find_package(Qt${QT_VERSION_MAJOR} COMPONENTS QuickControls2 REQUIRED)

set( source_files
    cpp_sample.cpp
)

set (header_files 
    cpp_sample.h
)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set_property(DIRECTORY APPEND PROPERTY COMPILE_DEFINITIONS $<$<CONFIG:Debug>:QT_QML_DEBUG>)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

add_executable(cpp_sample ${source_files} cpp_sample.qrc)
target_link_libraries(cpp_sample PUBLIC QuickQanava Qt${QT_VERSION_MAJOR}::QuickControls2)

if(WIN32 AND DEPLOY)
    find_program(WINDEPLOYQT_EXECUTABLE NAMES windeployqt HINTS ${QTDIR} ENV QTDIR PATH_SUFFIXES bin)
    add_custom_command(TARGET sample_nodes POST_BUILD
        COMMAND ${WINDEPLOYQT_EXECUTABLE} --qmldir ${CMAKE_CURRENT_SOURCE_DIR} $<TARGET_FILE:sample_nodes> $<$<CONFIG:Debug>:--pdb>)
    add_custom_command(TARGET sample_nodes POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:QuickQanava> $<TARGET_FILE_DIR:sample_nodes>)
endif()

