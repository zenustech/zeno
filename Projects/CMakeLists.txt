cmake_minimum_required(VERSION 3.12)
project(zeno_projects LANGUAGES C CXX)
#cmake_policy(SET CMP0079 NEW)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/../share/cmake ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

add_subdirectory(zenbase)
add_subdirectory(zenvdb)
add_subdirectory(FastFLIP)
add_subdirectory(ZMS)
add_subdirectory(QuickOCT)
