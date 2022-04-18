#set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} /usr/lib/cmake/OpenVDB /usr/local/lib/cmake/OpenVDB ${CMAKE_CURRENT_SOURCE_DIR}/cmake)  # cihou zhxx

set(ZENO_EXTENSIONS ZenoFX oldzenbase zenvdb FastFLIP FLIPtools Rigid BulletTools cgmesh gmpm mesher Euler OldZMS OldZFX EasyGL FEM Alembic LSystem Skinning Functional TreeSketch HerculesTools)
foreach (name IN ITEMS ${ZENO_EXTENSIONS})
    option(ZENO_WITH_${name} "Build ZENO with extension: ${name}" OFF)

    if (ZENO_WITH_${name})
        message(STATUS "Building ZENO with extension: ${name}")
        add_subdirectory(${name})
    endif()
endforeach()
