set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

# Qt6 seems doesn't have <QOpenGLWidget>... let's stick to Qt5 for now
find_package(QT NAMES Qt5 COMPONENTS Widgets REQUIRED)
find_package(Qt${QT_VERSION_MAJOR} COMPONENTS Widgets REQUIRED)
message("-- Found Qt${QT_VERSION_MAJOR} version: ${Qt${QT_VERSION_MAJOR}_VERSION}")

file(GLOB_RECURSE source RELATIVE editor CONFIGURE_DEPENDS *.h *.cpp *.ui *.qrc)

add_executable(zeno ${source})

target_link_libraries(zeno PRIVATE Qt${QT_VERSION_MAJOR}::Widgets)
