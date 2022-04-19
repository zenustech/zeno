file(GLOB_RECURSE sources CONFIGURE_DEPENDS *.h *.cpp)

add_executable(zenorunner ${sources})
target_link_libraries(zenorunner PRIVATE zeno zenoSharedExtension)

if (ZENO_ENABLE_BACKWARD)
    target_sources(zenorunner PRIVATE ${BACKWARD_ENABLE})
    add_backward(zenorunner)
endif()

if (ZENO_INSTALL_TARGET)
    install(TARGETS zenorunner EXPORT ZenoTargets)
endif()
