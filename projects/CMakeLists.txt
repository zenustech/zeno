cmake_minimum_required(VERSION 3.10)

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH})

set(EXTENSIONS_LIST ZenoFX oldzenbase zenvdb FastFLIP FLIPtools Rigid BulletTools cgmesh gmpm mesher Euler OldZMS OldZFX EasyGL)
set(EXTENSIONS_ENABLE_LIST ZenoFX oldzenbase)

add_library(zeno_nodep INTERFACE)
if (ZENO_HIDE_SYMBOLS AND UNIX)
    message("Hidding symbols from ZENO libraries...")
    target_link_libraries(zeno_nodep INTERFACE -Wl,--version-script,${CMAKE_CURRENT_SOURCE_DIR}/../scripts/linker.map)
endif()

foreach (name IN ITEMS ${EXTENSIONS_LIST})
    if (${name} IN_LIST EXTENSIONS_ENABLE_LIST)
        set(val ON)
    else()
        set(val OFF)
    endif()
    option(EXTENSION_${name} "Build ZENO with extension: ${name}" ${val})
endforeach()

foreach (name IN ITEMS ${EXTENSIONS_LIST})
    if (${EXTENSION_${name}})
        message("*** Building ZENO with extension: ${name}")
        add_subdirectory(${name})
    endif()
endforeach()
