cmake_minimum_required(VERSION 3.19)

if(NOT ZENO_WITH_zenvdb OR NOT ZENO_WITH_ZenoFX OR NOT ZENO_WITH_CUDA)
  message(FATAL_ERROR "zenvdb, ZenoFX and CUDA must be ON when PyZpc is ON! "
    "Please specify: -DZENO_WITH_zenvdb:BOOL=ON -DZENO_WITH_ZenoFX:BOOL=ON  -DZENO_WITH_CUDA:BOOL=ON")
endif()

# assume cuda
target_link_libraries(zeno PRIVATE zpc_jit_py)
target_link_libraries(zeno PRIVATE zshelper)

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
message("ZPY_SRC : ${ZPY_SRC}")
foreach(ZPY_FILE IN LISTS ZPY_SRC)
    add_custom_command(
                TARGET copy_zpy POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different ${ZPY_FILE} ${RESOURCE_BASE_DIR}/resource/py_libs/zpy
                COMMENT "Copying zpy src file: ${ZPY_FILE}")
endforeach()

configure_file(zeno_PyZpc_config.h.in ${CMAKE_CURRENT_BINARY_DIR}/zeno_PyZpc_config.h @ONLY)

add_subdirectory(interop)
