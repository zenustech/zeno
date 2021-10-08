add_executable(ZenoEditor)

target_include_directories(ZenoEditor PRIVATE .)
target_link_libraries(ZenoEditor PRIVATE fmt spdlog)
target_link_libraries(ZenoEditor PRIVATE glad glfw ftgl)
target_sources(ZenoEditor PRIVATE ${BACKWARD_ENABLE})
add_backward(ZenoEditor)

file(GLOB_RECURSE source FOLLOW_SYMLINKS LIST_DIRECTORIES FALSE CONFIGURE_DEPENDS zeno2/*.h zeno2/*.cpp)
target_sources(ZenoEditor PRIVATE ${source})
