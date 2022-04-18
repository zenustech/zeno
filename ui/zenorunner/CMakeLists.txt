file(GLOB_RECURSE sources CONFIGURE_DEPENDS *.h *.cpp)

add_executable(zenorunner ${sources})
target_link_libraries(zenorunner PUBLIC zeno zenoinclude)

if (ZENO_FAULTHANDLER)
    target_sources(zenorunner PRIVATE ${BACKWARD_ENABLE})
    add_backward(zenorunner)
endif()
target_link_libraries(zenorunner PRIVATE zeno)

if (ZENO_INSTALL_TARGET)
    install(TARGETS zenorunner EXPORT ZenoTargets)
endif()
