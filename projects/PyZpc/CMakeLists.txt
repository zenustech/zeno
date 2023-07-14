cmake_minimum_required(VERSION 3.19)

if(NOT ZENO_WITH_zenvdb OR NOT ZENO_WITH_ZenoFX OR NOT ZENO_WITH_CUDA)
  message(FATAL_ERROR "zenvdb, ZenoFX and CUDA must be ON when PyZpc is ON! "
    "Please specify: -DZENO_WITH_zenvdb:BOOL=ON -DZENO_WITH_ZenoFX:BOOL=ON  -DZENO_WITH_CUDA:BOOL=ON")
endif()

# assume cuda
project(ZENO_PyZpc CXX CUDA)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

target_link_libraries(zeno PRIVATE zshelper ${Python3_LIBRARIES})
target_include_directories(zeno PRIVATE ${Python3_INCLUDE_DIRS})

message("Python3_EXECUTABLE: ${Python3_EXECUTABLE}")
message("Python3_INTERPRETER_ID: ${Python3_INTERPRETER_ID}")
message("Python3_STDLIB: ${Python3_STDLIB}")
message("Python3_COMPILER: ${Python3_COMPILER}")
message("Python3_INCLUDE_DIR: ${Python3_INCLUDE_DIR}")
message("Python3_LIBRARIES: ${Python3_LIBRARIES}")

#
target_sources(zeno PRIVATE
  pyzfx.cpp
)
