cmake_minimum_required(VERSION 3.5)
set(CMAKE_CXX_STANDARD 17)
project(zeno_CalcGeometryUV)

find_package (Threads)
find_package(Eigen3 REQUIRED NO_MODULE)

include_directories(./calcUVCore)
include_directories(${EIGEN3_INCLUDE_DIR})

file(GLOB SRC_LIST *.cpp *.h ./calcUVCore/*.cpp ./calcUVCore/*.h)

target_include_directories(zeno PRIVATE . ./calcUVCore)
target_sources(zeno PRIVATE ${SRC_LIST})
target_link_libraries(zeno PRIVATE ${CMAKE_THREAD_LIBS_INIT})