add_library(zs_blender SHARED)

target_sources(zs_blender PRIVATE ${BACKWARD_ENABLE})
add_backward(zs_blender)

add_subdirectory(pybind11)

target_link_libraries(zs_blender PRIVATE zs pybind11)

file(GLOB_RECURSE sources CONFIGURE_DEPENDS *.h *.cpp)
target_sources(zs_blender PRIVATE ${sources})
