cmake_minimum_required(VERSION 3.18)
set(CMAKE_CXX_STANDARD 20)
project(zenvis)

file(GLOB PROJECT_SOURCE "*.cpp" "*.hpp" "*.h")

add_library(zenvis SHARED)
target_sources(zenvis PRIVATE ${PROJECT_SOURCE})
target_include_directories(zenvis PRIVATE .)
target_include_directories(zenvis PRIVATE ../include)
target_link_libraries(zenvis PRIVATE OpenGL GLEW glfw)

# zen link flags
target_include_directories(zenvis PRIVATE ../zen/include)
target_link_directories(zenvis PRIVATE ../zen/build)
target_link_libraries(zenvis PRIVATE zen)

# zenbase link flags
target_include_directories(zenvis PRIVATE ../zenbase/include)
target_link_directories(zenvis PRIVATE ../zenbase/build)
target_link_libraries(zenvis PRIVATE zenbase)
