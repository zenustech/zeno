set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

find_package(QT NAMES Qt5 COMPONENTS Widgets REQUIRED)
find_package(Qt${QT_VERSION_MAJOR} COMPONENTS Widgets REQUIRED)
message("-- Found Qt${QT_VERSION_MAJOR} version: ${Qt${QT_VERSION_MAJOR}_VERSION}")

zeno_glob_recurse(source editor *.h *.cpp *.ui *.qrc)
add_executable(zeno ${source})

target_link_libraries(zeno PRIVATE Qt${QT_VERSION_MAJOR}::Widgets)
target_include_directories(zeno PRIVATE editor)
