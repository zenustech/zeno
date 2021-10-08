add_library(ZenoCore OBJECT)
target_include_directories(ZenoCore PUBLIC .)
target_link_libraries(ZenoCore PUBLIC fmt spdlog)
target_link_libraries(ZenoCore PRIVATE glad glfw ftgl)

file(GLOB_RECURSE source CONFIGURE_DEPENDS *.h *.cpp)
target_sources(ZenoCore PRIVATE ${source})
