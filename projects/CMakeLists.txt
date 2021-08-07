cmake_minimum_required(VERSION 3.10)

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake ${CMAKE_CURRENT_SOURCE_DIR}/../zeno/cmake ${CMAKE_MODULE_PATH})
set(ENV{PYTHONPATH} ${CMAKE_CURRENT_SOURCE_DIR}/.. $ENV{PYTHONPATH})

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

foreach (name IN ITEMS ${EXTENSIONS_LIST})
    if (${EXTENSION_${name}})
        message("*** Building ZENO with extension: ${name}")
        add_subdirectory(${name})
    endif()
endforeach()

add_library(zeno_projects INTERFACE)
foreach (name IN ITEMS ${EXTENSIONS_LIST})
    target_link_libraries(zeno_projects INTERFACE zeno_${name})
endforeach()
