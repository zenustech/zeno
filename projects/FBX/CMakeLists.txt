file(GLOB FBX_SOURCE *.cpp *.h)

target_sources(zeno PRIVATE ${FBX_SOURCE})
zeno_disable_warning(${FBX_SOURCE})
target_include_directories(zeno PRIVATE .)
target_include_directories(zeno PRIVATE ../../zenovis/stbi/include)
target_include_directories(zeno PRIVATE ../../ui/include)

set(ASSIMP_BUILD_ALL_EXPORTERS_BY_DEFAULT FALSE CACHE BOOL "GUN TMD ASS" FORCE)
set(ASSIMP_BUILD_ALL_IMPORTERS_BY_DEFAULT FALSE CACHE BOOL "GUN TMD ASS" FORCE)
set(ASSIMP_BUILD_FBX_IMPORTER TRUE CACHE BOOL "CIHOU FBX" FORCE)
set(ASSIMP_BUILD_FBX_EXPORTER TRUE CACHE BOOL "CIHOU FBX" FORCE)
add_subdirectory(assimp)
target_link_libraries(zeno PRIVATE assimp)

#find_package(assimp CONFIG REQUIRED)
#target_link_libraries(zeno PRIVATE assimp::assimp)

if(ZENO_FBXSDK)
    # 1. Set `-DZENO_FBXSDK=ON`
    # 2. On Windows: Download FBXSDK from `https://www.autodesk.com/developer-network/platform-technologies/fbx-sdk-archives`
    # 3. Install FBXSDK to `projects/FBX/FBXSDK/windows/2020.3.1`
    # 4. Set `-DFBX_SDK_DIR="C:\src\zeno2\projects\FBX\FBXSDK"`

    message(STATUS "FBX_SDK_DIR: ${FBX_SDK_DIR}")
    target_compile_definitions(zeno PRIVATE -DZENO_FBXSDK)
    if(WIN32)
        target_include_directories(zeno PRIVATE ${FBX_SDK_DIR}/2020.3.1/include)
        if(CMAKE_BUILD_TYPE STREQUAL "Debug")
            target_link_libraries(zeno PRIVATE ${FBX_SDK_DIR}/2020.3.1/lib/vs2019/x64/debug/libfbxsdk-md.lib)
            target_link_libraries(zeno PRIVATE ${FBX_SDK_DIR}/2020.3.1/lib/vs2019/x64/debug/libxml2-md.lib)
        else()
            target_link_libraries(zeno PRIVATE ${FBX_SDK_DIR}/2020.3.1/lib/vs2019/x64/release/libfbxsdk-md.lib)
            target_link_libraries(zeno PRIVATE ${FBX_SDK_DIR}/2020.3.1/lib/vs2019/x64/release/libxml2-md.lib)
        endif()
    else()
        target_include_directories(zeno PRIVATE ${FBX_SDK_DIR}/linux/include)

        target_link_libraries(zeno PRIVATE ${FBX_SDK_DIR}/linux/lib/gcc/x64/release/libfbxsdk.a)
endif()
endif()

set(DEM_DIR ${CMAKE_CURRENT_SOURCE_DIR}/dem-bones-bin/bin/${CMAKE_SYSTEM_NAME})
if (NOT EXISTS ${DEM_DIR})
    message(FATAL_ERROR "Please run: git submodule update --init --recursive")
endif()
zeno_add_asset_dir(${DEM_DIR})
configure_file(zeno_FBX_config.h.in ${CMAKE_CURRENT_BINARY_DIR}/confinc/zeno_FBX_config.h @ONLY)
target_include_directories(zeno PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/confinc)

execute_process(COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/dem-bones-bin/models ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/models)