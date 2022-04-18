add_library(zenoinclude INTERFACE)

target_include_directories(zenoinclude INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include/Zeno/zenoinclude>)

if (ZENO_INSTALL_TARGET)
    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include DESTINATION include/Zeno/zenoinclude)
    install(TARGETS zenoinclude EXPORT ZenoTargets)
endif()
