cmake_minimum_required(VERSION 3.18)

include(cmake/ZenoHelpers.cmake)
include(cmake/ZenoOptions.cmake)

project(zeno VERSION 0.1 LANGUAGES CXX DESCRIPTION "Zeno node system for simulations")

add_subdirectory(designer)

message(STATUS "Zeno Target: [${ZENO_TARGET}]")
include(cmake/ZenoTarget${ZENO_TARGET}.cmake)
include(cmake/ZenoSource.cmake)
