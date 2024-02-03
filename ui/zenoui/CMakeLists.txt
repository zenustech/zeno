find_package(Qt5 REQUIRED COMPONENTS Widgets Svg OpenGL Network)

file(GLOB_RECURSE source CONFIGURE_DEPENDS *.h *.cpp *.ui)
file(GLOB_RECURSE resource CONFIGURE_DEPENDS *.qrc)
qt5_add_big_resources(source ${resource})  # cihou msvc: fatal error C1060: compiler is out of heap space

add_library(zenoui OBJECT ${source})

set_property(TARGET zenoui PROPERTY AUTOUIC ON)
set_property(TARGET zenoui PROPERTY AUTOMOC ON)
set_property(TARGET zenoui PROPERTY AUTORCC ON)

if (ZENO_DEBUG_MSVC)
    zeno_dbg_msvc(zenoui)
endif()

target_link_libraries(zenoui PUBLIC Qt5::Widgets Qt5::Svg Qt5::Network)
target_link_libraries(zenoui PUBLIC zeno)

target_include_directories(zenoui PRIVATE ${Qt5Widgets_PRIVATE_INCLUDE_DIRS})
target_include_directories(zenoui PRIVATE . .. ../include)

if (WIN32)
    if (CMAKE_BUILD_TYPE STREQUAL "Debug")
        target_link_libraries(zenoui PRIVATE  ${CMAKE_CURRENT_SOURCE_DIR}/./lib/qwtd.lib)
    else()
        target_link_libraries(zenoui PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/./lib/qwt.lib)
    endif()
endif()
