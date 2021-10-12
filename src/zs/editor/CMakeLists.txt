cmake_minimum_required(VERSION 3.18)

project(ZenoEditor)

find_package(Qt5 COMPONENTS Widgets Qml Quick REQUIRED)

qt5_add_resources(QT_RESOURCES resources.qrc)
file(GLOB_RECURSE HEADERS CONFIGURE_DEPENDS *.h)
file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS *.cpp)

add_executable(ZenoEditor ${HEADERS} ${SOURCES} ${QT_RESOURCES})
target_link_libraries(ZenoEditor
    Qt5::Widgets
    Qt5::Qml
    Qt5::Quick
    )
