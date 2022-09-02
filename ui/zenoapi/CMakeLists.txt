#find_package(Qt5 REQUIRED COMPONENTS Widgets Svg OpenGL)

file(GLOB_RECURSE source CONFIGURE_DEPENDS *.h *.cpp)

add_library(zenoapi OBJECT ${source})

#set_property(TARGET zenoapi PROPERTY AUTOUIC ON)
#set_property(TARGET zenoapi PROPERTY AUTOMOC ON)
#set_property(TARGET zenoapi PROPERTY AUTORCC ON)

target_link_libraries(zenoapi PUBLIC zenomodel)
target_include_directories(zenoapi PRIVATE . .. ../include include)