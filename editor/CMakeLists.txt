find_package(QT NAMES Qt5 COMPONENTS Widgets REQUIRED)
find_package(Qt${QT_VERSION_MAJOR} COMPONENTS Widgets WebEngineWidgets REQUIRED)
message(STATUS "Found Qt${QT_VERSION_MAJOR} version ${Qt${QT_VERSION_MAJOR}_VERSION}: ${Qt${QT_VERSION_MAJOR}_DIR}")

set(WEB_ENGINE "")
set(WEB_ENGINE_LINK "")

file(GLOB_RECURSE source CONFIGURE_DEPENDS *.h *.cpp *.ui *.qrc)
add_executable(zeno_editor ${source})

set_property(TARGET zeno_editor PROPERTY AUTOUIC ON)
set_property(TARGET zeno_editor PROPERTY AUTOMOC ON)
set_property(TARGET zeno_editor PROPERTY AUTORCC ON)
set_property(TARGET zeno_editor PROPERTY WIN32_EXECUTABLE ON)

if (ZENO_WITH_BACKWARD)
    target_sources(zeno_editor PRIVATE ${BACKWARD_ENABLE})
    add_backward(zeno_editor)
endif()

if (APPLE)
    set_target_properties(zeno PROPERTIES
        MACOSX_BUNDLE_GUI_IDENTIFIER zenustech.com
        MACOSX_BUNDLE_BUNDLE_VERSION ${PROJECT_VERSION}
        MACOSX_BUNDLE_SHORT_VERSION_STRING ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}
    )
endif()

target_link_libraries(zeno_editor PRIVATE Qt${QT_VERSION_MAJOR}::Widgets)
target_link_libraries(zeno_editor PRIVATE Qt${QT_VERSION_MAJOR}::WebEngineWidgets)
target_link_libraries(zeno_editor PRIVATE KDAB::kddockwidgets)
target_include_directories(zeno_editor PRIVATE .)
target_link_libraries(zeno_editor PRIVATE zeno_core)
