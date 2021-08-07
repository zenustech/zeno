cmake_minimum_required(VERSION 3.10)

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH})

set(EXTENSIONS_LIST ZenoFX Rigid oldzenbase zenvdb FastFLIP gmpm Euler mesher ZMS OldZFX EasyGL)
set(EXTENSIONS_ENABLE_LIST ZenoFX Rigid oldzenbase)

foreach (name IN ITEMS ${EXTENSIONS_LIST})
    if (${name} IN_LIST EXTENSIONS_ENABLE_LIST)
        set(val ON)
    else()
        set(val OFF)
    endif()
    option(EXTENSION_${name} "Build ZENO with extension: ${name}" ${val})
endforeach()

function(add_zeno_autoload target)
    if (ZENO_ENABLE_PYTHON)
        set(OUTPUT_DIR ${PROJECT_SOURCE_DIR}/zenqt)
        set_target_properties(${target} PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY ${OUTPUT_DIR}
            RUNTIME_OUTPUT_DIRECTORY ${OUTPUT_DIR}
            ARCHIVE_OUTPUT_DIRECTORY ${OUTPUT_DIR}
            LIBRARY_OUTPUT_DIRECTORY_DEBUG ${OUTPUT_DIR}
            RUNTIME_OUTPUT_DIRECTORY_DEBUG ${OUTPUT_DIR}
            ARCHIVE_OUTPUT_DIRECTORY_DEBUG ${OUTPUT_DIR}
            LIBRARY_OUTPUT_DIRECTORY_RELEASE ${OUTPUT_DIR}
            RUNTIME_OUTPUT_DIRECTORY_RELEASE ${OUTPUT_DIR}
            ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${OUTPUT_DIR}
            )
    endif()
endfunction()

add_library(zeno_projects INTERFACE)
foreach (name IN ITEMS ${EXTENSIONS_LIST})
    if (${EXTENSION_${name}})
        message("*** Building ZENO with extension: ${name}")
        add_subdirectory(${name})
        target_link_libraries(zeno_projects INTERFACE zeno_${name})
        add_zeno_autoload(zeno_${name})
    endif()
endforeach()
