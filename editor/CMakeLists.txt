set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

find_package(QT NAMES Qt6 Qt5 COMPONENTS Widgets REQUIRED)
find_package(Qt${QT_VERSION_MAJOR} COMPONENTS Widgets REQUIRED)

message("-- Found Qt${QT_VERSION_MAJOR} version: ${Qt${QT_VERSION_MAJOR}_VERSION}")

file(GLOB_RECURSE source CONFIGURE_DEPENDS *.h *.cpp *.ui)

#if(${QT_VERSION_MAJOR} GREATER_EQUAL 6)
    #qt_add_executable(zeno MANUAL_FINALIZATION ${source})
#else()
    add_executable(zeno ${source})
#endif()

if (ZENO_WITH_BACKWARD)
    target_sources(zeno PRIVATE ${BACKWARD_SOURCES})
    add_backward(zeno)
endif()

target_link_libraries(zeno PRIVATE Qt${QT_VERSION_MAJOR}::Widgets)

if (APPLE)
    set_target_properties(zeno PROPERTIES
        MACOSX_BUNDLE_GUI_IDENTIFIER zenustech.com
        MACOSX_BUNDLE_BUNDLE_VERSION ${PROJECT_VERSION}
        MACOSX_BUNDLE_SHORT_VERSION_STRING ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}
    )
endif()

#if(QT_VERSION_MAJOR EQUAL 6)
    #qt_finalize_executable(zeno)
#endif()
