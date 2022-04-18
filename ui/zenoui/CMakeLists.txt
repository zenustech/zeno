find_package(Qt5 REQUIRED COMPONENTS Widgets Svg OpenGL)

file(GLOB_RECURSE source CONFIGURE_DEPENDS *.h *.cpp *.ui *.qrc)

add_library(zenoui OBJECT ${source})

set_property(TARGET zenoui PROPERTY AUTOUIC ON)
set_property(TARGET zenoui PROPERTY AUTOMOC ON)
set_property(TARGET zenoui PROPERTY AUTORCC ON)

target_link_libraries(zenoui PUBLIC Qt5::Widgets Qt5::Svg)
target_link_libraries(zenoui PUBLIC zeno zenoinclude)

target_include_directories(zenoui PRIVATE ${Qt5Widgets_PRIVATE_INCLUDE_DIRS})
target_include_directories(zenoui PRIVATE . ..)
