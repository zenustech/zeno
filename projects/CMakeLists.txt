set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} /usr/lib/cmake/OpenVDB /usr/local/lib/cmake/OpenVDB ${CMAKE_CURRENT_SOURCE_DIR}/cmake)  # cihou zhxx
add_library(zeno_nodep INTERFACE)  # deprecated

add_library(zenodes SHARED dummy.cpp)
target_link_libraries(zenodes PUBLIC zenocore)

set(extensions ZenoFX oldzenbase zenvdb FastFLIP FLIPtools Rigid BulletTools cgmesh gmpm mesher Euler OldZMS OldZFX EasyGL FEM Alembic LSystem Skinning)
foreach (name IN ITEMS ${extensions})
    option(ZENO_WITH_${name} "Build ZENO with extension: ${name}" OFF)

    if (ZENO_WITH_${name})
        message(STATUS "Building ZENO with extension: ${name}")
        add_subdirectory(${name})
        if (NOT TARGET zeno_${name})
            message(WARNING "Please use 'zeno_${name}' as the target name of ZENO extension, otherwise it won't be linked")
        else()
            target_link_libraries(zenodes PRIVATE zeno_${name})
        endif()
    endif()
endforeach()
