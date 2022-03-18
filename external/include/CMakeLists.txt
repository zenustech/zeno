add_library(zenoinclude INTERFACE)
target_include_directories(zenoinclude INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:${INSTALL_CMAKE_DIR}>)
