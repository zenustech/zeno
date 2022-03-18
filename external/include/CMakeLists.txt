add_library(zenoinclude INTERFACE)
target_include_directories(zenoinclude INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include/Zeno-${PROJECT_VERSION}/zenoinclude>)
if (ZENO_INSTALL_TARGET)
    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/ DESTINATION include/Zeno-${PROJECT_VERSION}/zenoinclude)
endif()
