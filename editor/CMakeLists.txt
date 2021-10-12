
add_executable(zs_editor)
target_link_libraries(zs_editor PRIVATE
    Qt5::Widgets
    Qt5::Qml
    Qt5::Quick
    )

target_sources(zs_editor PRIVATE ${BACKWARD_ENABLE})
add_backward(zs_editor)

target_link_libraries(zs_editor PRIVATE zs)

file(GLOB_RECURSE source CONFIGURE_DEPENDS *.h *.cpp)
target_sources(zs_editor PRIVATE ${source})

find_package(Qt5 COMPONENTS Widgets Qml Quick REQUIRED)
qt5_add_resources(qt_resources resources.qrc)
target_sources(zs_editor PRIVATE ${qt_resources})
target_link_libraries(zs PRIVATE
    Qt5::Widgets
    Qt5::Qml
    Qt5::Quick
    )
