find_package(CUDA REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")

# set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-11.6)
message("The cuda toolkit directory is : " ${CUDA_TOOLKIT_ROOT_DIR}  "\n")

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; 
                    -use_fast_math; 
                    -gencode arch=compute_60,code=sm_60; 
                    -gencode arch=compute_70,code=sm_70;)

set(CUDA_LINK_LIBRARIES_KEYWORD PUBLIC)

cuda_add_library(gpu_lib2 utilities.cu ImmersedBoundaryMethod.cu ImmersedBoudnaryMethod.cpp GpuVector.cu PiecewisePolynomial.cu test_gpu_vector.cu)

target_include_directories(gpu_lib2 ${CUDA_LINK_LIBRARIES_KEYWORD} ${CUDA_INCLUDE_DIRS})
target_link_libraries(gpu_lib2 ${CUDA_LINK_LIBRARIES_KEYWORD} ${CUDA_TOOLKIT_ROOT_DIR}/lib64/libcudart_static.a)
target_include_directories(gpu_lib2 ${CUDA_LINK_LIBRARIES_KEYWORD} ${CUDA_TOOLKIT_ROOT_DIR}/samples/common/inc)
target_include_directories(gpu_lib2 PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/cuda-samples/Common")

# Expose the interface
target_include_directories(gpu_lib2 PUBLIC ./include)

set_target_properties(gpu_lib2 PROPERTIES CXX_STANDARD_REQUIRED ON)

set_property(TARGET gpu_lib2 PROPERTY
        CMAKE_CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
        CUDA_STANDARD 17
        CUDA_ARCHITECTURES "60;72")


