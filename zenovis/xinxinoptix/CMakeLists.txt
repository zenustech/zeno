cmake_minimum_required(VERSION 3.16)

if (NOT DEFINED OPTIX_PATH)
    message(FATAL_ERROR "OPTIX_PATH not defined, please specify from command line, for example: cmake -B build -DOPTIX_PATH=\"/home/bate/Downloads/NVIDIA-OptiX-SDK-7.4.0-linux64-x86_64\"")
endif()

target_sources(zenovis PRIVATE
  optixPathTracer.cpp
  optixPathTracer.h
  OptiXStuff.h
  SDK/sutil/Aabb.h              SDK/sutil/Quaternion.h
  SDK/sutil/Camera.cpp          SDK/sutil/Record.h
  SDK/sutil/Camera.h            SDK/sutil/Scene.cpp
  SDK/sutil/CUDAOutputBuffer.h  SDK/sutil/Scene.h
  SDK/sutil/Exception.h         SDK/sutil/sutilapi.h
  SDK/sutil/GLDisplay.cpp       SDK/sutil/sutil.cpp
  SDK/sutil/GLDisplay.h         SDK/sutil/sutil.h
  SDK/sutil/Matrix.h            SDK/sutil/Trackball.cpp
  SDK/sutil/PPMLoader.cpp       SDK/sutil/Trackball.h
  SDK/sutil/PPMLoader.h         SDK/sutil/vec_math.h
  SDK/sutil/Preprocessor.h      SDK/sutil/WorkDistribution.h
  )

find_package(CUDAToolkit REQUIRED COMPONENTS cudart nvrtc REQUIRED)
target_link_libraries(zenovis PRIVATE CUDA::cudart CUDA::nvrtc)
target_include_directories(zenovis PRIVATE ${OPTIX_PATH}/include)
target_include_directories(zenovis PRIVATE SDK)
target_include_directories(zenovis PRIVATE SDK/sutil)
target_include_directories(zenovis PRIVATE SDK/sdk_cuda)
target_include_directories(zenovis PRIVATE SDK/support)
target_include_directories(zenovis PRIVATE SDK/support/GLFW/include)
target_include_directories(zenovis PRIVATE SDK/support/imgui)
target_include_directories(zenovis PRIVATE .)
configure_file(sampleConfig.h.in ${CMAKE_CURRENT_BINARY_DIR}/sampleConfig.h @ONLY)
target_include_directories(zenovis PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

if (NOT DEFINED CUDAToolkit_BIN_DIR)
    message(FATAL_ERROR "CUDAToolkit_BIN_DIR not defined")
endif()
set(CUDA_PATH ${CUDAToolkit_BIN_DIR}/..)

set(FILE_LIST
    ${OPTIX_PATH}/include/@optix.h
    ${OPTIX_PATH}/include/@optix_device.h
    ${OPTIX_PATH}/include/@optix_7_device.h
    ${OPTIX_PATH}/include/@optix_7_types.h
    ${OPTIX_PATH}/include/@internal/optix_7_device_impl.h
    ${OPTIX_PATH}/include/@internal/optix_7_device_impl_exception.h
    ${OPTIX_PATH}/include/@internal/optix_7_device_impl_transformations.h
    ${CUDA_PATH}/include/@vector_types.h
    ${CUDA_PATH}/include/@vector_functions.h
    ${CMAKE_CURRENT_SOURCE_DIR}/SDK/@cuda/random.h
    ${CMAKE_CURRENT_SOURCE_DIR}/SDK/@sutil/vec_math.h
    ${CMAKE_CURRENT_SOURCE_DIR}/SDK/@cuda/helpers.h
    ${CMAKE_CURRENT_SOURCE_DIR}/SDK/@sutil/Preprocessor.h
    ${CMAKE_CURRENT_SOURCE_DIR}/@optixPathTracer.h
    ${CMAKE_CURRENT_SOURCE_DIR}/@TraceStuff.h
    ${CMAKE_CURRENT_SOURCE_DIR}/@PTKernel.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/@DisneyBRDF.h
    ${CMAKE_CURRENT_SOURCE_DIR}/@zxxglslvec.h
    ${CMAKE_CURRENT_SOURCE_DIR}/@IOMat.h
    ${CMAKE_CURRENT_SOURCE_DIR}/@IOMat.h
    )
foreach (FILE_INFO IN ITEMS ${FILE_LIST})
    #string(REPLACE "/" "_" ESCAPED_FILE_NAME "${FILE_NAME}")
    string(MD5 ENCODED_FILE "${FILE_INFO}")
    string(REPLACE "@" ";" SEXY_LIST "${FILE_INFO}")
    list(GET SEXY_LIST 0 FILE_PATH)
    list(GET SEXY_LIST 1 FILE_NAME)
    set(FILE_PATH "${FILE_PATH}${FILE_NAME}")
    set(DUMMY_SOURCE_FILE ${CMAKE_CURRENT_BINARY_DIR}/${ENCODED_FILE}_bundle.cpp)
    if (NOT EXISTS ${DUMMY_SOURCE_FILE} OR ${FILE_PATH} IS_NEWER_THAN ${DUMMY_SOURCE_FILE})

        message(STATUS "Encoding NVRTC header: ${FILE_PATH}")
        file(READ ${FILE_PATH} HEX_CONTENTS HEX)
        string(REGEX MATCHALL "([A-Za-z0-9][A-Za-z0-9])" SEPARATED_HEX "${HEX_CONTENTS}")
        list(JOIN SEPARATED_HEX ",\n0x" FORMATTED_HEX)
        string(PREPEND FORMATTED_HEX "0x")

        file(WRITE ${DUMMY_SOURCE_FILE} "/* generated from: ${FILE_PATH} */\n#include <vector>\n#include <sutil/sutil.h>\n\nnamespace sutil {\nstatic const unsigned char mydata[] = {\n${FORMATTED_HEX},\n0};\nstatic int helper = (getIncPathTab().push_back(\"${FILE_NAME}\"), getIncFileTab().push_back((const char *)mydata), 0);\n}")

    endif()
    target_sources(zenovis PRIVATE ${DUMMY_SOURCE_FILE})
endforeach()
