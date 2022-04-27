#set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} /usr/lib/cmake/OpenVDB /usr/local/lib/cmake/OpenVDB ${CMAKE_CURRENT_SOURCE_DIR}/cmake)  # cihou zhxx
# ^^^NONO^^^ Please add `export OpenVDB_DIR=/usr/local/lib/cmake/OpenVDB` in your .bashrc to YI-LAO-YONG-YI!!!!!!

set(ZENO_EXTENSIONS
    ZenoFX
    oldzenbase
    zenvdb
    FastFLIP
    Rigid
    cgmesh
    gmpm
    mesher
    Euler
    FEM
    Alembic
    LSystem
    Skinning
    Functional
    TreeSketch
    )
add_subdirectory(SharedExtension)

if (ZENO_NO_WARNING)
    set(CMAKE_CXX_FLAGS "${ZENO_OLD_CXX_FLAGS}")
    if (CMAKE_CXX_COMPILER_ID MATCHES "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -w")
    endif()
    if (CMAKE_C_COMPILER_ID MATCHES "GNU" OR CMAKE_C_COMPILER_ID MATCHES "Clang" OR CMAKE_C_COMPILER_ID MATCHES "MSVC")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -w")
    endif()
endif()

foreach (name IN ITEMS ${ZENO_EXTENSIONS})
    option(ZENO_WITH_${name} "Build ZENO with extension: ${name}" OFF)
    if (ZENO_WITH_${name})
        message(STATUS "Building ZENO with extension: ${name}")
        add_subdirectory(${name})
    endif()
endforeach()

set(ZENO_EXTENSION_TOOLS
    FLIPtools
    BulletTools
    HerculesTools
    )
foreach (name IN ITEMS ${ZENO_EXTENSION_TOOLS})
    option(ZENO_WITH_TOOL_${name} "Build ZENO with extension tool: ${name}" OFF)
    if (ZENO_WITH_TOOL_${name})
        message(STATUS "Building ZENO with extension tool: ${name}")
        add_subdirectory(tools/${name})
    endif()
endforeach()
