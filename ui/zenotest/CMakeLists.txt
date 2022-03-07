file(GLOB_RECURSE source CONFIGURE_DEPENDS *.h *.cpp)
add_executable(zenotest ${source}) 

if (ZENO_FAULTHANDLER)
    target_sources(zenotest PRIVATE ${BACKWARD_ENABLE})
    add_backward(zenotest)
endif()

target_link_libraries(zenotest PRIVATE zenodes)
