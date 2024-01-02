find_package(Qt5 REQUIRED COMPONENTS Widgets Svg OpenGL)

file(GLOB_RECURSE source CONFIGURE_DEPENDS *.h *.cpp)

add_library(zenomodel OBJECT ${source})

set_property(TARGET zenomodel PROPERTY AUTOUIC ON)
set_property(TARGET zenomodel PROPERTY AUTOMOC ON)
set_property(TARGET zenomodel PROPERTY AUTORCC ON)

if (ZENO_DEBUG_MSVC)
    zeno_dbg_msvc(zenomodel)
endif()

target_link_libraries(zenomodel PUBLIC Qt5::Widgets)
target_link_libraries(zenomodel PUBLIC zenocore)
target_include_directories(zenomodel PRIVATE . .. ../include include)