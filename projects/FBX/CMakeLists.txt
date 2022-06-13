file(GLOB FBX_SOURCE *.cpp *.h)

target_sources(zeno PRIVATE ${FBX_SOURCE})
target_include_directories(zeno PRIVATE .)
target_include_directories(zeno PRIVATE ../../zenovis/stbi/include)

add_subdirectory(assimp)

target_link_libraries(zeno PRIVATE assimp)

set(DEM_DIR ${CMAKE_CURRENT_SOURCE_DIR}/dem-bones-bin/bin/${CMAKE_SYSTEM_NAME})
list(APPEND ZENO_ASSET_PATHS ${DEM_DIR})
set(DEM_PATH ${DEM_DIR}/DemBones)
configure_file(zeno_FBX_config.h.in ${CMAKE_CURRENT_BINARY_DIR}/confinc/zeno_FBX_config.h @ONLY)
target_include_directories(zeno PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/confinc)
