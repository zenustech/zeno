add_executable(zs_editor)

target_sources(zs_editor PRIVATE ${BACKWARD_ENABLE})
add_backward(zs_editor)

target_link_libraries(zs_editor PRIVATE zs)

file(GLOB_RECURSE sources CONFIGURE_DEPENDS *.h *.cpp)
target_sources(zs_editor PRIVATE ${sources})

find_package(Qt5 COMPONENTS Widgets Qml Quick REQUIRED)
qt5_add_resources(resources resources.qrc)
target_sources(zs_editor PRIVATE ${resources})
set_property(TARGET zs_editor PROPERTY AUTOMOC ON)
target_link_libraries(zs_editor PRIVATE
    Qt5::Widgets
    Qt5::Qml
    Qt5::Quick
    )
