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

if (ZENO_NO_WARNING)
    add_compile_options(-w)
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
    cgmeshTools
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
