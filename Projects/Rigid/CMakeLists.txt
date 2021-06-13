cmake_minimum_required(VERSION 3.19)
set(CMAKE_CXX_STANDARD 17)
project(zenRigidLib)

file (GLOB ZEN_RIGID_SOURCE *.cpp)
#add_library(zenRigidLib SHARED ${ZEN_RIGID_SOURCE})
add_executable(zenRigidLib ${ZEN_RIGID_SOURCE})
add_subdirectory(bullet3)
target_include_directories(zenRigidLib PRIVATE .)
target_include_directories(zenRigidLib PRIVATE bullet3/src)

#find_package(IlmBase REQUIRED COMPONENTS Half REQUIRED)
#target_link_libraries(zenRigidLib PRIVATE openvdb IlmBase::Half)
target_link_libraries(zenRigidLib PRIVATE pthread openblas lapack cblas)

target_link_libraries(zenRigidLib PRIVATE LinearMath)
target_link_libraries(zenRigidLib PRIVATE Bullet3Common)
target_link_libraries(zenRigidLib PRIVATE BulletInverseDynamics)
target_link_libraries(zenRigidLib PRIVATE BulletCollision)
target_link_libraries(zenRigidLib PRIVATE BulletDynamics)
target_link_libraries(zenRigidLib PRIVATE BulletSoftBody)


if (OFF)
    find_package(zen REQUIRED)
    target_link_libraries(zenRigidLib PRIVATE zen)
    set_target_properties(zenRigidLib PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${zen_AUTOLOAD_DIR}
        RUNTIME_OUTPUT_DIRECTORY ${zen_AUTOLOAD_DIR}
        ARCHIVE_OUTPUT_DIRECTORY ${zen_AUTOLOAD_DIR}
        LIBRARY_OUTPUT_DIRECTORY_DEBUG ${zen_AUTOLOAD_DIR}
        RUNTIME_OUTPUT_DIRECTORY_DEBUG ${zen_AUTOLOAD_DIR}
        ARCHIVE_OUTPUT_DIRECTORY_DEBUG ${zen_AUTOLOAD_DIR}
        LIBRARY_OUTPUT_DIRECTORY_RELEASE ${zen_AUTOLOAD_DIR}
        RUNTIME_OUTPUT_DIRECTORY_RELEASE ${zen_AUTOLOAD_DIR}
        ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${zen_AUTOLOAD_DIR}
        )
    target_include_directories(zenRigidLib PRIVATE ../zenbase/include)
    target_include_directories(zenRigidLib PRIVATE ../zenvdb/include)
endif()
