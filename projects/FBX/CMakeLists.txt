cmake_minimum_required(VERSION 3.16)
<<<<<<< HEAD

file(GLOB FBX_SOURCE *.cpp *.h)

target_sources(zeno PRIVATE ${FBX_SOURCE})

target_include_directories(zeno PRIVATE .)

add_subdirectory(assimp)

target_link_libraries(zeno PRIVATE assimp)
=======
set(CMAKE_CXX_STANDARD 17)
project(zeno_FBX)

file(GLOB FBX_SOURCE *.cpp *.h)

add_library(zeno_FBX SHARED ${FBX_SOURCE})

target_include_directories(zeno_FBX PRIVATE .)

find_package(assimp CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)

target_link_libraries(zeno_FBX PRIVATE assimp::assimp)
target_link_libraries(zeno_FBX PRIVATE zeno)
<<<<<<< HEAD
>>>>>>> 630499906 (FBX Extension # Indices and Vertices Read)
=======
target_link_libraries(zeno_FBX PRIVATE glm::glm)
>>>>>>> 6b309d70b (FBX Extension # Bone Animation)
