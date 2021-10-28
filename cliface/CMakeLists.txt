file(GLOB_RECURSE source CONFIGURE_DEPENDS *.h *.cpp *.ui)

add_executable(zeno_cliface ${source})

if (ZENO_WITH_BACKWARD)
    target_sources(zeno_cliface PRIVATE ${BACKWARD_SOURCES})
    add_backward(zeno_cliface)
endif()

target_link_libraries(zeno_cliface PRIVATE zeno)

