file(GLOB FBX_SOURCE *.cpp *.h)

option(USE_OFFICIAL_ASSIMP "Use Official Assimp Library" OFF)

target_sources(zeno PRIVATE ${FBX_SOURCE})
target_include_directories(zeno PRIVATE .)
target_include_directories(zeno PRIVATE ${CMAKE_SOURCE_DIR}/zenovis/stbi/include)

if(USE_OFFICIAL_ASSIMP)
    message(STATUS "FBX Extension: USE_OFFICIAL_ASSIMP ON")
    target_compile_definitions(zeno PRIVATE "USE_OFFICIAL_ASSIMP=1")
    add_subdirectory(lib/assimp)
else()
    add_subdirectory(assimp)
endif()

target_link_libraries(zeno PRIVATE assimp)