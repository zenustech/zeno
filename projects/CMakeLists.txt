cmake_minimum_required(VERSION 3.10)

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH})

set(EXTENSIONS_LIST ZenoFX oldzenbase zenvdb FastFLIP gmpm mesher Euler Rigid ZMS OldZFX EasyGL)
set(EXTENSIONS_ENABLE_LIST ZenoFX oldzenbase)

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

        if (ZENO_ENABLE_PYTHON)
            get_property(OUTPUT_DIR TARGET zeno
                PROPERTY LIBRARY_OUTPUT_DIRECTORY)
            set_target_properties(zeno_${name} PROPERTIES
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
    endif()
endforeach()
