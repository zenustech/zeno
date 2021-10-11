add_executable(zs_main)
target_sources(zs_main PRIVATE ${BACKWARD_ENABLE})
add_backward(zs_main)

target_link_libraries(zs_main PRIVATE zs)

file(GLOB_RECURSE source CONFIGURE_DEPENDS *.h *.cpp)
target_sources(zs_main PRIVATE ${source})
