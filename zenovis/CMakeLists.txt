file(GLOB_RECURSE source CONFIGURE_DEPENDS include/*.h src/*.cpp)
file(GLOB_RECURSE glad_source CONFIGURE_DEPENDS glad/include/*.h glad/src/*.c)
file(GLOB_RECURSE stbi_source CONFIGURE_DEPENDS stbi/include/*.h stbi/src/*.cpp)

add_library(zenovis OBJECT ${source} ${glad_source} ${stbi_source})
target_link_libraries(zenovis PRIVATE ${CMAKE_DL_LIBS})
target_link_libraries(zenovis PUBLIC zeno)
target_include_directories(zenovis PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/glad/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/stbi/include>)

add_subdirectory(zhxxvis)
target_link_libraries(zenovis PRIVATE zhxxvis)

#if (ZENO_INSTALL_TARGET)
    #install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include DESTINATION include/Zeno/zenovis)
    #install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/glad/include DESTINATION include/Zeno/zenovis)
    #install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/stbi/include DESTINATION include/Zeno/zenovis)
    #install(TARGETS zenovis EXPORT ZenoTargets)
#endif()
