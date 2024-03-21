cmake_minimum_required(VERSION 3.16)

set(OPTIX_PATH ${CMAKE_CURRENT_SOURCE_DIR})
target_sources(zenovis PRIVATE
  optixPathTracer.cpp
  optixPathTracer.h
  OptiXStuff.h
  Host.h

  volume/optixVolume.cpp
  volume/optixVolume.h
  optixSphere.cpp
  optixSphere.h
  Shape.h
  XAS.h

  LightTree.cpp
  LightTree.h
  LightBounds.cpp
  LightBounds.h
  ies/ies.cpp
  ies/ies.h
  TypeCaster.cpp
  TypeCaster.h
  
  SDK/sutil/Aabb.h              SDK/sutil/Quaternion.h
  SDK/sutil/Camera.cpp          SDK/sutil/Record.h
  SDK/sutil/Camera.h            SDK/cuda/climits.h
  SDK/sutil/CUDAOutputBuffer.h  SDK/cuda/cstdint.h
  SDK/sutil/Exception.h         SDK/sutil/sutilapi.h
  SDK/sutil/GLDisplay.cpp       SDK/sutil/sutil.cpp
  SDK/sutil/GLDisplay.h         SDK/sutil/sutil.h
  SDK/sutil/Matrix.h            SDK/sutil/Trackball.cpp
  SDK/sutil/PPMLoader.cpp       SDK/sutil/Trackball.h
  SDK/sutil/PPMLoader.h         SDK/sutil/vec_math.h
  SDK/sutil/Preprocessor.h      SDK/sutil/WorkDistribution.h
  )

#if (OPTIX_USE_20XX)
    #target_compile_definitions(zenovis PRIVATE -DUSING_20XX)
#endif()

find_package(CUDAToolkit REQUIRED COMPONENTS cudart nvrtc REQUIRED)
target_link_libraries(zenovis PRIVATE CUDA::cudart CUDA::nvrtc)

if (ZENO_SYSTEM_OPENVDB)
    message(FATAL_ERROR "Zeno Optix requires OpenVDB/nanoVDB as submodule, system installed OpenVDB is not supported. Please set ZENO_SYSTEM_OPENVDB=OFF, then run: git submodule update --init --recursive")
else() # ../../projects/zenvdb/openvdb/nanovdb
    get_filename_component(NVDB_PATH ${OPTIX_PATH}/../../projects/zenvdb/openvdb/nanovdb ABSOLUTE)
    message(${NVDB_PATH})
endif()

target_include_directories(zenovis PRIVATE volume)
target_include_directories(zenovis PRIVATE ${NVDB_PATH})

find_package(TBB CONFIG REQUIRED COMPONENTS tbb tbbmalloc)
target_link_libraries(zenovis PRIVATE TBB::tbb TBB::tbbmalloc)

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

## The Chief Designer's Excellent Work BEG
if (NOT ZENO_WITH_Alembic)
    message(FATAL_ERROR "(Choice of the LK-99, Choice of the artists) The Chief Designer Peng requires you to enable -DZENO_WITH_Alembic=ON for building embed OpenEXR")
else()
    target_link_libraries(zenovis PRIVATE Alembic)
endif()
## The Chief Designer's Excellent Work END

if (NOT DEFINED CUDAToolkit_BIN_DIR)
    message(FATAL_ERROR "CUDAToolkit_BIN_DIR not defined")
endif()
set(CUDA_PATH ${CUDAToolkit_BIN_DIR}/..)

set(FILE_LIST
    ${OPTIX_PATH}/include/@optix.h
    ${OPTIX_PATH}/include/@optix_types.h
    ${OPTIX_PATH}/include/@optix_host.h
    ${OPTIX_PATH}/include/@optix_device.h
    ${OPTIX_PATH}/include/@internal/optix_device_impl.h
    ${OPTIX_PATH}/include/@internal/optix_device_impl_exception.h
    ${OPTIX_PATH}/include/@internal/optix_device_impl_transformations.h

    ${CUDA_PATH}/include/@cuda_fp16.h
    ${CUDA_PATH}/include/@cuda_fp16.hpp  
    # ${CUDA_PATH}/include/@nvfunctional 
    ${CUDA_PATH}/include/@math_constants.h
    ${CUDA_PATH}/include/@vector_types.h
    ${CUDA_PATH}/include/@vector_functions.h 
    
    ${CMAKE_CURRENT_SOURCE_DIR}/@OptiXToolkit/ShaderUtil/Preprocessor.h
    ${CMAKE_CURRENT_SOURCE_DIR}/@OptiXToolkit/ShaderUtil/SelfIntersectionAvoidance.h
    ${CMAKE_CURRENT_SOURCE_DIR}/@OptiXToolkit/ShaderUtil/CudaSelfIntersectionAvoidance.h
    ${CMAKE_CURRENT_SOURCE_DIR}/@OptiXToolkit/ShaderUtil/OptixSelfIntersectionAvoidance.h

    ${CMAKE_CURRENT_SOURCE_DIR}/OptiXToolkit/ShaderUtil/@SelfIntersectionAvoidanceTypes.h
    ${CMAKE_CURRENT_SOURCE_DIR}/OptiXToolkit/ShaderUtil/SelfIntersectionAvoidance/@SelfIntersectionAvoidanceImpl.h
    ${CMAKE_CURRENT_SOURCE_DIR}/OptiXToolkit/ShaderUtil/@SelfIntersectionAvoidance/CudaSelfIntersectionAvoidanceImpl.h
    ${CMAKE_CURRENT_SOURCE_DIR}/OptiXToolkit/ShaderUtil/@SelfIntersectionAvoidance/OptixSelfIntersectionAvoidanceImpl.h

    ${CMAKE_CURRENT_SOURCE_DIR}/SDK/@cuda/random.h
    ${CMAKE_CURRENT_SOURCE_DIR}/SDK/@cuda/helpers.h
    ${CMAKE_CURRENT_SOURCE_DIR}/SDK/@cuda/climits.h
    ${CMAKE_CURRENT_SOURCE_DIR}/SDK/@cuda/cstdint.h
    
    ${CMAKE_CURRENT_SOURCE_DIR}/SDK/@sutil/vec_math.h
    ${CMAKE_CURRENT_SOURCE_DIR}/SDK/@sutil/Preprocessor.h
    #include <zeno/types/LightObject.h>
    ${CMAKE_CURRENT_SOURCE_DIR}/../../zeno/include/@zeno/types/LightObject.h

    ${CMAKE_CURRENT_SOURCE_DIR}/@optixPathTracer.h
    ${CMAKE_CURRENT_SOURCE_DIR}/@TraceStuff.h
    ${CMAKE_CURRENT_SOURCE_DIR}/@PTKernel.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/@CallableDefault.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/@CallableVolume.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/@DeflMatShader.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/@DisneyBRDF.h
    ${CMAKE_CURRENT_SOURCE_DIR}/@DisneyBSDF.h
    ${CMAKE_CURRENT_SOURCE_DIR}/@HairBSDF.h
    ${CMAKE_CURRENT_SOURCE_DIR}/@zxxglslvec.h
    ${CMAKE_CURRENT_SOURCE_DIR}/@IOMat.h
    ${CMAKE_CURRENT_SOURCE_DIR}/@proceduralSky.h

    ${CMAKE_CURRENT_SOURCE_DIR}/@LightBounds.h
    ${CMAKE_CURRENT_SOURCE_DIR}/@LightTree.h

    ${CMAKE_CURRENT_SOURCE_DIR}/@Sampling.h    
    ${CMAKE_CURRENT_SOURCE_DIR}/@Shape.h
    ${CMAKE_CURRENT_SOURCE_DIR}/@Light.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/@Light.h
    ${CMAKE_CURRENT_SOURCE_DIR}/@volume.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/@volume.h
    
    ${NVDB_PATH}/@nanovdb/NanoVDB.h
    ${NVDB_PATH}/@nanovdb/util/Ray.h
    ${NVDB_PATH}/@nanovdb/util/HDDA.h
    # ${NVDB_PATH}/@nanovdb/util/Stencils.h
    ${NVDB_PATH}/@nanovdb/util/SampleFromVoxels.h
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
