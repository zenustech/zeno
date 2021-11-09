cmake_minimum_required(VERSION 3.18)

include(cmake/ZenoOptions.cmake)

project(zeno VERSION 0.1 LANGUAGES C CXX)

add_subdirectory(tpls)

message("-- Zeno Target: [${ZENO_TARGET}]")
include(cmake/ZenoBuild${ZENO_TARGET}.cmake)

include(cmake/ZenoSource.cmake)
