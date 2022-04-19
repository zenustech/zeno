# ZXX = Modern Fancy GL Boy
# LZH = Modern Fancy Qt Boy
# WXL = Modern Fancy Cpp Boy
# WJJ = Modern Fancy Mesa Boy
# PYB = Modern Fancy CMake Boy

add_library(zenoSharedExtension OBJECT)
target_link_libraries(zenoSharedExtension PUBLIC zeno)
target_sources(zenoSharedExtension PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/dummy_export.cpp)
target_sources(zenoSharedExtension INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/dummy_import.cpp)
set_property(TARGET zenoSharedExtension PROPERTY LINK_WHAT_YOU_USE ON)
# if (ZENO_INSTALL_TARGET)
#     install(TARGETS zenoSharedExtension EXPORT ZenoTargets)
# endif()
