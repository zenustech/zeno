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

if (WIN32)
    set(ZENO_PYZPC_DLL_FILE zeno.dll)
elseif (APPLE)
    set(ZENO_PYZPC_DLL_FILE libzeno.dylib)
else()
    set(ZENO_PYZPC_DLL_FILE libzeno.so)
endif()
target_include_directories(zeno PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

if (DEFINED CMAKE_LIBRARY_OUTPUT_DIRECTORY)
  set(RESOURCE_BASE_DIR ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
elseif (DEFINED CMAKE_RUNTIME_OUTPUT_DIRECTORY)
  set(RESOURCE_BASE_DIR ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
else()
  set(RESOURCE_BASE_DIR ${CMAKE_CURRENT_BINARY_DIR})
endif()

add_custom_target(copy_zpy)
add_dependencies(zeno copy_zpy)
set(PY_LIBS_DIR ${RESOURCE_BASE_DIR}/resource/py_libs)
add_custom_command(
  TARGET copy_zpy
  POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory ${PY_LIBS_DIR}/zpy
  COMMENT "creating resource Python module directory at ${PY_LIBS_DIR}/zpy"
)
file(GLOB ZPY_SRC ${CMAKE_CURRENT_SOURCE_DIR}/Lib/zpy/*.py)
foreach(ZPY_FILE IN LISTS ZPY_SRC)
    add_custom_command(
                TARGET copy_zpy POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different ${ZPY_FILE} ${RESOURCE_BASE_DIR}/resource/py_libs/zpy
                COMMENT "Copying zpy src file: ${ZPY_FILE}")
endforeach()

configure_file(zeno_PyZpc_config.h.in ${CMAKE_CURRENT_BINARY_DIR}/zeno_PyZpc_config.h @ONLY)