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

IF (WIN32)
    SET(SHORTCUT_SRCS "./shortcut/gui/qxtwindowsystem.cpp" "./shortcut/gui/qxtwindowsystem.h" "./shortcut/gui/qxtglobalshortcut_win.cpp" "./shortcut/gui/qxtwindowsystem_win.cpp")
    find_package(Qt5 REQUIRED COMPONENTS Widgets Svg)
ENDIF (WIN32)

IF (UNIX)
    find_package(X11)
    find_package(Qt5 REQUIRED COMPONENTS Widgets X11Extras Svg)
    IF (APPLE)
        SET(SHORTCUT_SRCS "./shortcut/gui/qxtglobalshortcut_mac.cpp" "./shortcut/gui/qxtwindowsystem_mac.cpp" "./shortcut/gui/qxtwindowsystem_mac.h") 
    ELSE (APPLE)
        SET(SHORTCUT_SRCS "./shortcut/gui/qxtwindowsystem_x11.cpp" "./shortcut/gui/qxtglobalshortcut_x11.cpp" "./shortcut/gui/x11info.cpp" "./shortcut/gui/x11info.h")
    ENDIF(APPLE)
ENDIF (UNIX)

file(GLOB_RECURSE source CONFIGURE_DEPENDS *.h *.cpp *.ui *.qrc)

include_directories(
    ../3rdparty
)

add_executable(designer
   ${source}
)

if (WIN32)
    target_link_libraries(designer Qt5::Widgets Qt5::Svg rapidjson)
else()
    target_link_libraries(designer Qt5::Widgets Qt5::Svg Qt5::X11Extras dl X11)
endif(WIN32)