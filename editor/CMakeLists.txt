find_package(QT NAMES Qt5 COMPONENTS Widgets REQUIRED)
find_package(Qt${QT_VERSION_MAJOR} COMPONENTS Widgets REQUIRED)
message(STATUS "Found Qt${QT_VERSION_MAJOR} version ${Qt${QT_VERSION_MAJOR}_VERSION}: ${Qt${QT_VERSION_MAJOR}_DIR}")

file(GLOB_RECURSE source CONFIGURE_DEPENDS *.h *.cpp *.ui *.qrc)
add_executable(zeno_editor ${source})

set_property(TARGET zeno_editor PROPERTY AUTOUIC ON)
set_property(TARGET zeno_editor PROPERTY AUTOMOC ON)
set_property(TARGET zeno_editor PROPERTY AUTORCC ON)

target_link_libraries(zeno_editor PRIVATE Qt${QT_VERSION_MAJOR}::Widgets)
target_include_directories(zeno_editor PRIVATE .)
target_link_libraries(zeno_editor PRIVATE zeno_core)
