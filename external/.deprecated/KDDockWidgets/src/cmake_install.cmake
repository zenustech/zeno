# Install script for directory: /home/bate/zeno/3rdparty/qtmod/KDDockWidgets/src

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Debug")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "0")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "/home/bate/zeno/3rdparty/qtmod/KDDockWidgets/src/libkddockwidgets.a")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/kddockwidgets" TYPE FILE FILES
    "/home/bate/zeno/3rdparty/qtmod/KDDockWidgets/src/Config"
    "/home/bate/zeno/3rdparty/qtmod/KDDockWidgets/src/DockWidget"
    "/home/bate/zeno/3rdparty/qtmod/KDDockWidgets/src/DockWidgetBase"
    "/home/bate/zeno/3rdparty/qtmod/KDDockWidgets/src/FocusScope"
    "/home/bate/zeno/3rdparty/qtmod/KDDockWidgets/src/FrameworkWidgetFactory"
    "/home/bate/zeno/3rdparty/qtmod/KDDockWidgets/src/DefaultWidgetFactory"
    "/home/bate/zeno/3rdparty/qtmod/KDDockWidgets/src/LayoutSaver"
    "/home/bate/zeno/3rdparty/qtmod/KDDockWidgets/src/MainWindow"
    "/home/bate/zeno/3rdparty/qtmod/KDDockWidgets/src/MainWindowBase"
    "/home/bate/zeno/3rdparty/qtmod/KDDockWidgets/src/MainWindowMDI"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/kddockwidgets" TYPE FILE FILES
    "/home/bate/zeno/3rdparty/qtmod/KDDockWidgets/src/docks_export.h"
    "/home/bate/zeno/3rdparty/qtmod/KDDockWidgets/src/Config.h"
    "/home/bate/zeno/3rdparty/qtmod/KDDockWidgets/src/FrameworkWidgetFactory.h"
    "/home/bate/zeno/3rdparty/qtmod/KDDockWidgets/src/DockWidgetBase.h"
    "/home/bate/zeno/3rdparty/qtmod/KDDockWidgets/src/KDDockWidgets.h"
    "/home/bate/zeno/3rdparty/qtmod/KDDockWidgets/src/Qt5Qt6Compat_p.h"
    "/home/bate/zeno/3rdparty/qtmod/KDDockWidgets/src/FocusScope.h"
    "/home/bate/zeno/3rdparty/qtmod/KDDockWidgets/src/QWidgetAdapter.h"
    "/home/bate/zeno/3rdparty/qtmod/KDDockWidgets/src/LayoutSaver.h"
    "/home/bate/zeno/3rdparty/qtmod/KDDockWidgets/src/MainWindowMDI.h"
    "/home/bate/zeno/3rdparty/qtmod/KDDockWidgets/src/MainWindowBase.h"
    "/home/bate/zeno/3rdparty/qtmod/KDDockWidgets/src/MainWindow.h"
    "/home/bate/zeno/3rdparty/qtmod/KDDockWidgets/src/DockWidget.h"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/kddockwidgets/private" TYPE FILE FILES
    "/home/bate/zeno/3rdparty/qtmod/KDDockWidgets/src/private/DragController_p.h"
    "/home/bate/zeno/3rdparty/qtmod/KDDockWidgets/src/private/Draggable_p.h"
    "/home/bate/zeno/3rdparty/qtmod/KDDockWidgets/src/private/DropArea_p.h"
    "/home/bate/zeno/3rdparty/qtmod/KDDockWidgets/src/private/DropIndicatorOverlayInterface_p.h"
    "/home/bate/zeno/3rdparty/qtmod/KDDockWidgets/src/private/FloatingWindow_p.h"
    "/home/bate/zeno/3rdparty/qtmod/KDDockWidgets/src/private/Frame_p.h"
    "/home/bate/zeno/3rdparty/qtmod/KDDockWidgets/src/private/LayoutSaver_p.h"
    "/home/bate/zeno/3rdparty/qtmod/KDDockWidgets/src/private/MultiSplitter_p.h"
    "/home/bate/zeno/3rdparty/qtmod/KDDockWidgets/src/private/LayoutWidget_p.h"
    "/home/bate/zeno/3rdparty/qtmod/KDDockWidgets/src/private/SideBar_p.h"
    "/home/bate/zeno/3rdparty/qtmod/KDDockWidgets/src/private/TitleBar_p.h"
    "/home/bate/zeno/3rdparty/qtmod/KDDockWidgets/src/private/WindowBeingDragged_p.h"
    "/home/bate/zeno/3rdparty/qtmod/KDDockWidgets/src/private/WidgetResizeHandler_p.h"
    "/home/bate/zeno/3rdparty/qtmod/KDDockWidgets/src/private/DockRegistry_p.h"
    "/home/bate/zeno/3rdparty/qtmod/KDDockWidgets/src/private/TabWidget_p.h"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/kddockwidgets/private/multisplitter" TYPE FILE FILES "/home/bate/zeno/3rdparty/qtmod/KDDockWidgets/src/private/multisplitter/Item_p.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/kddockwidgets/private/multisplitter" TYPE FILE FILES "/home/bate/zeno/3rdparty/qtmod/KDDockWidgets/src/private/multisplitter/Widget.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/kddockwidgets/private/multisplitter" TYPE FILE FILES "/home/bate/zeno/3rdparty/qtmod/KDDockWidgets/src/private/multisplitter/Separator_p.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/kddockwidgets/private/indicators" TYPE FILE FILES "/home/bate/zeno/3rdparty/qtmod/KDDockWidgets/src/private/indicators/ClassicIndicators_p.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/kddockwidgets/private/indicators" TYPE FILE FILES "/home/bate/zeno/3rdparty/qtmod/KDDockWidgets/src/private/indicators/SegmentedIndicators_p.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/kddockwidgets/private/widgets" TYPE FILE FILES
    "/home/bate/zeno/3rdparty/qtmod/KDDockWidgets/src/private/widgets/QWidgetAdapter_widgets_p.h"
    "/home/bate/zeno/3rdparty/qtmod/KDDockWidgets/src/private/widgets/TitleBarWidget_p.h"
    "/home/bate/zeno/3rdparty/qtmod/KDDockWidgets/src/private/widgets/SideBarWidget_p.h"
    "/home/bate/zeno/3rdparty/qtmod/KDDockWidgets/src/private/widgets/FloatingWindowWidget_p.h"
    "/home/bate/zeno/3rdparty/qtmod/KDDockWidgets/src/private/widgets/FrameWidget_p.h"
    "/home/bate/zeno/3rdparty/qtmod/KDDockWidgets/src/private/widgets/TabBarWidget_p.h"
    "/home/bate/zeno/3rdparty/qtmod/KDDockWidgets/src/private/widgets/TabWidgetWidget_p.h"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/kddockwidgets/private/multisplitter" TYPE FILE FILES "/home/bate/zeno/3rdparty/qtmod/KDDockWidgets/src/private/multisplitter/Separator_qwidget.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/kddockwidgets/private/multisplitter" TYPE FILE FILES "/home/bate/zeno/3rdparty/qtmod/KDDockWidgets/src/private/multisplitter/Widget_qwidget.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/kddockwidgets" TYPE FILE FILES "/home/bate/zeno/3rdparty/qtmod/KDDockWidgets/src/kddockwidgets_version.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/KDDockWidgets/KDDockWidgetsTargets.cmake")
    file(DIFFERENT EXPORT_FILE_CHANGED FILES
         "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/KDDockWidgets/KDDockWidgetsTargets.cmake"
         "/home/bate/zeno/3rdparty/qtmod/KDDockWidgets/src/CMakeFiles/Export/lib/cmake/KDDockWidgets/KDDockWidgetsTargets.cmake")
    if(EXPORT_FILE_CHANGED)
      file(GLOB OLD_CONFIG_FILES "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/KDDockWidgets/KDDockWidgetsTargets-*.cmake")
      if(OLD_CONFIG_FILES)
        message(STATUS "Old export file \"$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/KDDockWidgets/KDDockWidgetsTargets.cmake\" will be replaced.  Removing files [${OLD_CONFIG_FILES}].")
        file(REMOVE ${OLD_CONFIG_FILES})
      endif()
    endif()
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/KDDockWidgets" TYPE FILE FILES "/home/bate/zeno/3rdparty/qtmod/KDDockWidgets/src/CMakeFiles/Export/lib/cmake/KDDockWidgets/KDDockWidgetsTargets.cmake")
  if("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^([Dd][Ee][Bb][Uu][Gg])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/KDDockWidgets" TYPE FILE FILES "/home/bate/zeno/3rdparty/qtmod/KDDockWidgets/src/CMakeFiles/Export/lib/cmake/KDDockWidgets/KDDockWidgetsTargets-debug.cmake")
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/KDDockWidgets" TYPE FILE FILES
    "/home/bate/zeno/3rdparty/qtmod/KDDockWidgets/src/KDDockWidgetsConfig.cmake"
    "/home/bate/zeno/3rdparty/qtmod/KDDockWidgets/src/KDDockWidgetsConfigVersion.cmake"
    )
endif()

