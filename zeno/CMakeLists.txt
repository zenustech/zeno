add_executable(ZenoEditor)

target_include_directories(ZenoEditor PRIVATE .)
target_link_libraries(ZenoEditor PRIVATE spdlog fmt)
target_link_libraries(ZenoEditor PRIVATE glad glfw ftgl)
target_sources(ZenoEditor PRIVATE ${BACKWARD_ENABLE})
add_backward(ZenoEditor)

file(GLOB_RECURSE source FOLLOW_SYMLINKS LIST_DIRECTORIES false CONFIGURE_DEPENDS *.h *.cpp)
#message("* found zeno sources:\n${source}")
target_sources(ZenoEditor PRIVATE ${source})
