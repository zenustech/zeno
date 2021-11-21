add_subdirectory(external/pybind11)

pybind11_add_module(zs_blender)

target_sources(zs_blender PRIVATE ${BACKWARD_ENABLE})
add_backward(zs_blender)

target_link_libraries(zs_blender PRIVATE zs)
target_include_directories(zs_blender PRIVATE include)

file(GLOB_RECURSE sources CONFIGURE_DEPENDS *.h *.cpp)
target_sources(zs_blender PRIVATE ${sources})
