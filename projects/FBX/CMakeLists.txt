file(GLOB FBX_SOURCE *.cpp *.h)

target_sources(zeno PRIVATE ${FBX_SOURCE})
target_include_directories(zeno PRIVATE .)
target_include_directories(zeno PRIVATE ../../zenovis/stbi/include)

add_subdirectory(assimp)

target_link_libraries(zeno PRIVATE assimp)

set(DEM_PATH ${CMAKE_CURRENT_SOURCE_DIR}/dem-bones-bin/${CMAKE_SYSTEM_NAME})
configure_file(${CMAKE_CURRENT_BINARY_DIR}/zeno_FBX_config.h @ONLY)
