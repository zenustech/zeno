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

# todo: to be deprecated:
add_library(zeno_projects INTERFACE)
if (NOT MSVC)
    target_link_options(zeno_projects INTERFACE LINKER:--no-as-needed)
endif()

foreach (name IN ITEMS ${EXTENSIONS_LIST})
    if (${EXTENSION_${name}})
        message("*** Building ZENO with extension: ${name}")
        add_subdirectory(${name})
        add_zeno_autoload(zeno_${name})
        # todo: to be deprecated:
        target_link_libraries(zeno_projects INTERFACE zeno_${name})
    endif()
endforeach()
