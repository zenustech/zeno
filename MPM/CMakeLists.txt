cmake_minimum_required(VERSION 3.19)
project(MPM)

if (${ZS_ENABLE_CUDA})
    enable_language(CUDA)
endif()
# add_library(mpm SHARED main.cpp)
add_executable(mpm main.cpp)

#simd compiler flags
if (WIN32)
#target_compile_options(FLIPlib PRIVATE "/arch:AVX")
else()
target_compile_options(mpm PRIVATE "-march=native")
endif()

target_link_libraries(mpm PRIVATE zen)

#disable default malloc options
if (WIN32)
  target_compile_options(mpm PRIVATE -DNOMINMAX)
endif()

set_target_properties(mpm
  PROPERTIES CUDA_ARCHITECTURES OFF
)
#[=[
target_compile_options(mpm 
      PUBLIC        $<$<COMPILE_LANGUAGE:CUDA>:${CMAKE_CUDA_FLAGS} -fopenmp --expt-extended-lambda --expt-relaxed-constexpr --default-stream=per-thread --use_fast_math -lineinfo --ptxas-options=-allow-expensive-optimizations=true>
    )
    target_link_options(mpm 
      PUBLIC       $<$<LINK_LANGUAGE:CUDA>:-dlto>
    )
    set_target_properties(mpm
      PROPERTIES  CUDA_EXTENSIONS ON
                  CUDA_SEPARABLE_COMPILATION ON
                  LINKER_LANGUAGE CUDA
                  #CUDA_RESOLVE_DEVICE_SYMBOLS OFF
                  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                  POSITION_INDEPENDENT_CODE ON
                  CUDA_ARCHITECTURES OFF
    )
    target_compile_features(mpm PRIVATE cuda_std_17)
    target_compile_definitions(mpm 
      PUBLIC        CMAKE_GENERATOR_PLATFORM=x64
    )
#]=]

# target_link_directories(mpm PUBLIC /usr/lib/x86_64-linux-gnu:/usr/local/lib)
target_link_libraries(mpm PRIVATE zensim)
