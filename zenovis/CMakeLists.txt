file(GLOB_RECURSE source CONFIGURE_DEPENDS include/*.h src/*.cpp glad/include/*.h glad/src/*.c)

add_library(zenovis OBJECT ${source})
target_link_libraries(zenovis PUBLIC zeno zenoinclude zenostbi)
target_include_directories(zenovis PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/glad/include>
    $<INSTALL_INTERFACE:include/Zeno/zenovis>)

if (ZENO_INSTALL_TARGET)
    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include DESTINATION include/Zeno/zenovis)
    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/glad/include DESTINATION include/Zeno/zenovis)
    install(TARGETS zenovis EXPORT ZenoTargets)
endif()
