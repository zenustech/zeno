cmake_minimum_required(VERSION 2.6)
project(libjmeshext)
include_directories(JMeshLib-1.2/include)
if (WIN32)
include_directories(winhappy)
endif()

SET(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-strict-aliasing")

add_definitions(-DIS64BITPLATFORM)

include_directories("${PROJECT_SOURCE_DIR}/JMeshLib-1.2/include")
include_directories("${PROJECT_SOURCE_DIR}/include")
include_directories("${PROJECT_SOURCE_DIR}/OpenNL3.2.1/src")
include_directories("${PROJECT_SOURCE_DIR}/OpenNL3.2.1/src/NL")

add_subdirectory("${PROJECT_SOURCE_DIR}/JMeshLib-1.2")
add_subdirectory("${PROJECT_SOURCE_DIR}/OpenNL3.2.1")

file(GLOB SOURCES_JRS_Predicates "${PROJECT_SOURCE_DIR}/src/JRS_Predicates/*.c")
file(GLOB SOURCES_SRC "${PROJECT_SOURCE_DIR}/src/*.cpp")
add_library(jmeshext STATIC 
  ${SOURCES_JRS_Predicates} 
  ${SOURCES_SRC})
