cmake_minimum_required(VERSION 3.8.2)

# Find includes in the build directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)

# Turn on automatic invocation of the MOC, UIC & RCC
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)

# Make this a GUI application on Windows
if(WIN32)
  set(CMAKE_WIN32_EXECUTABLE ON)
endif()

if (WIN32)
    find_package(Qt5 REQUIRED COMPONENTS Widgets Svg)
else()
    find_package(X11)
    find_package(Qt5 REQUIRED COMPONENTS Widgets X11Extras Svg)
endif (WIN32)

file(GLOB_RECURSE source CONFIGURE_DEPENDS *.h *.cpp *.ui *.qrc)

add_executable(zenodesign ${source})
target_include_directories(zenodesign PRIVATE ../3rdparty ../zenoui ../editor)
target_link_libraries(zenodesign PRIVATE Qt5::Widgets Qt5::Svg zenoui zenocore)
if (NOT WIN32)
    target_link_libraries(zenodesign PRIVATE Qt5::X11Extras dl X11)
endif(NOT WIN32)
