find_package(Qt5 REQUIRED COMPONENTS Widgets)

file(GLOB_RECURSE source CONFIGURE_DEPENDS *.h *.cpp *.ui)
file(GLOB_RECURSE resource CONFIGURE_DEPENDS *.qrc)
qt5_add_big_resources(source ${resource})  # cihou msvc: fatal error C1060: compiler is out of heap space

add_library(zenoio OBJECT ${source})

set_property(TARGET zenoio PROPERTY AUTOUIC ON)
set_property(TARGET zenoio PROPERTY AUTOMOC ON)
set_property(TARGET zenoio PROPERTY AUTORCC ON)

target_link_libraries(zenoio PUBLIC Qt5::Widgets)
target_link_libraries(zenoio PUBLIC zeno)
target_include_directories(zenoio PRIVATE ${Qt5Widgets_PRIVATE_INCLUDE_DIRS})
target_include_directories(zenoio PRIVATE ..)
