add_subdirectory(bullet3)
add_subdirectory(HACD)

file (GLOB ZEN_RIGID_SOURCE *.cpp)
add_library(zenRigidLib SHARED ${ZEN_RIGID_SOURCE})
target_include_directories(zenRigidLib PRIVATE .)
target_include_directories(zenRigidLib PRIVATE bullet3/src)

target_link_libraries(zenRigidLib PRIVATE LinearMath)
target_link_libraries(zenRigidLib PRIVATE Bullet3Common)
target_link_libraries(zenRigidLib PRIVATE BulletInverseDynamics)
target_link_libraries(zenRigidLib PRIVATE BulletCollision)
target_link_libraries(zenRigidLib PRIVATE BulletDynamics)
target_link_libraries(zenRigidLib PRIVATE BulletSoftBody)
target_link_libraries(zenRigidLib PRIVATE HACD)

target_link_libraries(zenRigidLib PRIVATE zeno)

get_property(zeno_AUTOLOAD_DIR TARGET zeno PROPERTY LIBRARY_OUTPUT_DIRECTORY)
set_target_properties(zenRigidLib PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${zeno_AUTOLOAD_DIR}
    RUNTIME_OUTPUT_DIRECTORY ${zeno_AUTOLOAD_DIR}
    ARCHIVE_OUTPUT_DIRECTORY ${zeno_AUTOLOAD_DIR}
    LIBRARY_OUTPUT_DIRECTORY_DEBUG ${zeno_AUTOLOAD_DIR}
    RUNTIME_OUTPUT_DIRECTORY_DEBUG ${zeno_AUTOLOAD_DIR}
    ARCHIVE_OUTPUT_DIRECTORY_DEBUG ${zeno_AUTOLOAD_DIR}
    LIBRARY_OUTPUT_DIRECTORY_RELEASE ${zeno_AUTOLOAD_DIR}
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${zeno_AUTOLOAD_DIR}
    ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${zeno_AUTOLOAD_DIR}
    )
