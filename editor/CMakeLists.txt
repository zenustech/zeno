set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

find_package(QT NAMES Qt6 Qt5 COMPONENTS Widgets REQUIRED)
find_package(Qt${QT_VERSION_MAJOR} COMPONENTS Widgets REQUIRED)

#file(GLOB_RECURSE sources CONFIGURE_DEPENDS *.h *.cpp *.ui)
set(PROJECT_SOURCES
        main.cpp
        mainwindow.h mainwindow.cpp mainwindow.ui
        qdmgraphicsscene.h qdmgraphicsscene.cpp
        qdmgraphicsview.h qdmgraphicsview.cpp
        qdmgraphicsnode.h qdmgraphicsnode.cpp
        qdmgraphicssocket.h qdmgraphicssocket.cpp
        qdmgraphicssocketin.h qdmgraphicssocketin.cpp
        qdmgraphicssocketout.h qdmgraphicssocketout.cpp
        qdmgraphicslink.h qdmgraphicslink.cpp
        qdmgraphicslinkhalf.h qdmgraphicslinkhalf.cpp
        qdmgraphicslinkfull.h qdmgraphicslinkfull.cpp
        qdmlistviewnodemenu.h qdmlistviewnodemenu.cpp
)

if(${QT_VERSION_MAJOR} GREATER_EQUAL 6)
    qt_add_executable(zs_editor MANUAL_FINALIZATION ${PROJECT_SOURCES})
else()
    if(ANDROID)
        add_library(zs_editor SHARED ${PROJECT_SOURCES})
    else()
        add_executable(zs_editor ${PROJECT_SOURCES})
    endif()
endif()

target_link_libraries(zs_editor PRIVATE Qt${QT_VERSION_MAJOR}::Widgets)

set_target_properties(zs_editor PROPERTIES
    MACOSX_BUNDLE_GUI_IDENTIFIER zenustech.com
    MACOSX_BUNDLE_BUNDLE_VERSION ${PROJECT_VERSION}
    MACOSX_BUNDLE_SHORT_VERSION_STRING ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}
)

if(QT_VERSION_MAJOR EQUAL 6)
    qt_finalize_executable(zs_editor)
endif()
