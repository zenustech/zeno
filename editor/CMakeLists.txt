add_library(ZenoEditor OBJECT)
target_include_directories(ZenoEditor PUBLIC .)
target_link_libraries(ZenoEditor PUBLIC fmt spdlog)
target_link_libraries(ZenoEditor PRIVATE glad glfw ftgl glm)

file(GLOB_RECURSE source CONFIGURE_DEPENDS *.h *.cpp)
target_sources(ZenoEditor PRIVATE ${source})
