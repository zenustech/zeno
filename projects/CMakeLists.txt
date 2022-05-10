#set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} /usr/lib/cmake/OpenVDB /usr/local/lib/cmake/OpenVDB ${CMAKE_CURRENT_SOURCE_DIR}/cmake)  # cihou zhxx
# ^^^NONO^^^ Please add `export OpenVDB_DIR=/usr/local/lib/cmake/OpenVDB` in your .bashrc to YI-LAO-YONG-YI!!!!!!

<<<<<<< HEAD
if (ZENO_NO_WARNING)
    set(CMAKE_CXX_FLAGS "${ZENO_OLD_CXX_FLAGS} -w")
    add_compile_options(-w)
=======
set(CMAKE_MODULE_PATH /usr/lib/cmake/OpenVDB ${CMAKE_CURRENT_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH})

set(EXTENSIONS_LIST ZenoFX oldzenbase zenvdb FastFLIP FLIPtools Rigid BulletTools cgmesh gmpm mesher Euler OldZMS OldZFX EasyGL FEM Alembic FBX LSystem Skinning Functional TreeSketch HerculesTools)
set(EXTENSIONS_ENABLE_LIST ZenoFX oldzenbase)

add_library(zeno_nodep INTERFACE)
if (ZENO_HIDE_SYMBOLS AND UNIX)
    message("Hidding symbols from ZENO libraries...")
    target_link_libraries(zeno_nodep INTERFACE -Wl,--version-script,${CMAKE_CURRENT_SOURCE_DIR}/../scripts/linker.map)
>>>>>>> 630499906 (FBX Extension # Indices and Vertices Read)
endif()

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
    FBX
    Alembic
    LSystem
    Skinning
    Functional
    TreeSketch
    )

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
