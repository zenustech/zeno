cmake_minimum_required(VERSION 3.10)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

# GCC 11, Clang 12, MSVC 19.28
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
#set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR})

project(ZenoEditor LANGUAGES C CXX)

add_subdirectory(external/glad)
add_subdirectory(external/glfw)
add_subdirectory(external/ftgl)
add_subdirectory(external/fmt)
add_subdirectory(external/spdlog)
add_subdirectory(external/backward-cpp)

file(GLOB PROJECT_SOURCE z2/*/*.cpp z2/*/*.h)# zeno/*.cpp zeno/*.h zeno/*/*.cpp zeno/*/*.h zeno/*/*/*.cpp zeno/*/*/*.h)
add_executable(ZenoEditor ${PROJECT_SOURCE})

target_include_directories(ZenoEditor PRIVATE .)
target_link_libraries(ZenoEditor PRIVATE spdlog fmt)
target_link_libraries(ZenoEditor PRIVATE glad glfw ftgl)
target_sources(ZenoEditor PRIVATE ${BACKWARD_ENABLE})
add_backward(ZenoEditor)
