find_package(Qt5 REQUIRED COMPONENTS Widgets Svg OpenGL)

file(GLOB_RECURSE source CONFIGURE_DEPENDS *.h *.cpp)

add_library(zenomodel OBJECT ${source})

set_property(TARGET zenomodel PROPERTY AUTOUIC ON)
set_property(TARGET zenomodel PROPERTY AUTOMOC ON)
set_property(TARGET zenomodel PROPERTY AUTORCC ON)

target_link_libraries(zenomodel PUBLIC Qt5::Widgets)
target_link_libraries(zenomodel PUBLIC zenoui zeno)
target_include_directories(zenomodel PRIVATE . .. ../include)