add_library(zenostbi OBJECT stbi.cpp)
target_include_directories(zenostbi PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include/Zeno/zenostbi>)
if (ZENO_INSTALL_TARGET)
    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include DESTINATION include/Zeno/zenostbi)
    install(TARGETS zenostbi EXPORT ZenoTargets)
endif()
