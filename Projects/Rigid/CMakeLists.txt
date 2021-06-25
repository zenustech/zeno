####################################################
# FUN FACT: DID U KNOW THAT MICROSOFT LOVES LINUX? #
####################################################
if (MSVC)  # we must use this to fix bullet3 on win
    set(BUILD_SHARED_LIBS OFF)
    add_compile_options(
            $<$<CONFIG:>:/MT> #---------|
            $<$<CONFIG:Debug>:/MTd> #---|-- Statically link the runtime libraries
            $<$<CONFIG:Release>:/MT> #--|
    )  # cl: 命令行 warning D9025 :正在重写“/MD”(用“/MT”)
else()
    set(BUILD_SHARED_LIBS ON)
endif()
####################################################
# FUN FACT: DID U KNOW THAT MICROSOFT BUYS GITHUB? #
####################################################
#set(CMAKE_INSTALL_PREFIX ${zen_INSTALL_DIR})

add_subdirectory(bullet3)
add_subdirectory(HACD)

file (GLOB ZEN_RIGID_SOURCE *.cpp)
add_library(zenRigidLib SHARED ${ZEN_RIGID_SOURCE})
target_include_directories(zenRigidLib PRIVATE .)
target_include_directories(zenRigidLib PRIVATE bullet3/src)

#find_package(IlmBase REQUIRED COMPONENTS Half REQUIRED)
#target_link_libraries(zenRigidLib PRIVATE openvdb IlmBase::Half)
#target_link_libraries(zenRigidLib PRIVATE pthread openblas lapack cblas)

target_link_libraries(zenRigidLib PRIVATE LinearMath)
target_link_libraries(zenRigidLib PRIVATE Bullet3Common)
target_link_libraries(zenRigidLib PRIVATE BulletInverseDynamics)
target_link_libraries(zenRigidLib PRIVATE BulletCollision)
target_link_libraries(zenRigidLib PRIVATE BulletDynamics)
target_link_libraries(zenRigidLib PRIVATE BulletSoftBody)
target_link_libraries(zenRigidLib PRIVATE HACD)

find_package(zen REQUIRED)
target_link_libraries(zenRigidLib PRIVATE zen)
target_include_directories(zenRigidLib PRIVATE ../zenbase/include)
target_include_directories(zenRigidLib PRIVATE ../zenvdb/include)
####################################################
# WELL: SO DID LINUS TORVALDS LOVES MICROSOFT TOO? #
####################################################
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
