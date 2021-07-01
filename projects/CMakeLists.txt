cmake_minimum_required(VERSION 3.10)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/../zeno/cmake ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
set(ENV{PYTHONPATH} ${CMAKE_CURRENT_SOURCE_DIR}/..)

set(EXTENSIONS_LIST Rigid ZMS zenbase zenvdb FastFLIP gmpm Euler)
set(EXTENSIONS_ENABLE_LIST Rigid ZMS zenbase)

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
