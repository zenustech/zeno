find_package(QT NAMES Qt5 COMPONENTS Widgets REQUIRED)
find_package(Qt${QT_VERSION_MAJOR} COMPONENTS Widgets REQUIRED)
message(STATUS "Found Qt${QT_VERSION_MAJOR} version ${Qt${QT_VERSION_MAJOR}_VERSION}: ${Qt${QT_VERSION_MAJOR}_DIR}")

zeno_glob_recurse(source editor *.h *.cpp *.ui *.qrc)
add_executable(zeno ${source})

set_property(TARGET zeno PROPERTY AUTOUIC ON)
set_property(TARGET zeno PROPERTY AUTOMOC ON)
set_property(TARGET zeno PROPERTY AUTORCC ON)

target_link_libraries(zeno PRIVATE Qt${QT_VERSION_MAJOR}::Widgets)
target_include_directories(zeno PRIVATE editor)

if (ZENO_WITH_DESIGNER)
    message(STATUS "Building Zeno Designer")
    add_subdirectory(designer)
endif()
