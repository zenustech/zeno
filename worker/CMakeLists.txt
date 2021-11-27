file(GLOB_RECURSE source CONFIGURE_DEPENDS *.h *.cpp *.ui *.qrc)
add_executable(zeno_worker ${source}) 

if (ZENO_WITH_BACKWARD)
    target_sources(zeno_worker PRIVATE ${BACKWARD_ENABLE})
    add_backward(zeno_worker)
endif()

target_include_directories(zeno_worker PRIVATE .)
target_link_libraries(zeno_worker PRIVATE zeno_core)
