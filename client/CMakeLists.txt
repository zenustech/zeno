file(GLOB_RECURSE source CONFIGURE_DEPENDS *.h *.cpp)

add_executable(zeno)
target_sycl_sources(zeno PRIVATE ${source})

if (ZENO_WITH_BACKWARD)
    target_sources(zeno PRIVATE ${BACKWARD_ENABLE})
    add_backward(zeno)
endif()
