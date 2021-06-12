cmake_minimum_required(VERSION 3.19)
set(CMAKE_CXX_STANDARD 17)
project(zenRigidLib)
add_definitions(-D__TBB_CPP20_COMPARISONS_PRESENT=0)  
find_package(IlmBase REQUIRED COMPONENTS Half REQUIRED)

# add directory with macros
set (CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake)
file (GLOB ZEN_RIGID_SOURCE RigidNode/*.cpp )
add_library(zenRigidLib SHARED ${ZEN_RIGID_SOURCE})
# add cmake file
include (common)
add_subdirectory(zenbullet)
target_include_directories(zenRigidLib PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
target_include_directories(zenRigidLib PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/zenbullet/src/")
if (WIN32)
  target_compile_options(zenRigidLib PRIVATE -DNOMINMAX)
endif()

target_link_directories(zenRigidLib PUBLIC /usr/lib/x86_64-linux-gnu:/usr/local/lib)
target_link_libraries(zenRigidLib PRIVATE openvdb IlmBase::Half)
target_link_libraries(zenRigidLib PRIVATE pthread openblas lapack cblas)

target_link_libraries(zenRigidLib PRIVATE LinearMath)
target_link_libraries(zenRigidLib PRIVATE Bullet3Common)
target_link_libraries(zenRigidLib PRIVATE BulletInverseDynamics)
target_link_libraries(zenRigidLib PRIVATE BulletCollision)
target_link_libraries(zenRigidLib PRIVATE BulletDynamics)
target_link_libraries(zenRigidLib PRIVATE BulletSoftBody)


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
