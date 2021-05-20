cmake_minimum_required(VERSION 3.19)
project(MPM)

# if (POLICY CMP0104)
#   cmake_policy(SET CMP0104 OLD)
# endif (POLICY CMP0104)
enable_language(CXX)

add_library(mpm SHARED nosys/ToZensimParticles.cpp nosys/OutputZensimParticles.cpp nosys/ConstitutiveModels.cpp nosys/ToZensimLevelSet.cpp nosys/ZensimBoundary.cpp nosys/PartitionForParticles.cu nosys/GridFromPartition.cu nosys/TransferNodes.cu nosys/GridOps.cu nosys/ToParticleObject.cpp nosys/AppendZensimObject.cpp)

if (${ZS_ENABLE_CUDA})
    enable_language(CUDA)
    # set_target_properties(mpm PROPERTIES CUDA_ARCHITECTURES OFF)
    target_compile_features(mpm PRIVATE cuda_std_17)
endif()

#[=[
set_target_properties(mpm
  PROPERTIES  COMPILE_LANGUAGE CXX
              LINKER_LANGUAGE CXX)
#]=]
target_compile_features(mpm PRIVATE cxx_std_17)
# target_link_libraries(mpm PRIVATE culib)

#simd compiler flags
if (WIN32)
#target_compile_options(FLIPlib PRIVATE "/arch:AVX")
else()
target_compile_options(mpm PRIVATE "-march=native")
endif()
target_compile_options(mpm 
      PRIVATE        $<$<COMPILE_LANGUAGE:CUDA>:${CMAKE_CUDA_FLAGS} -dlto -fopenmp --expt-extended-lambda --expt-relaxed-constexpr --default-stream=per-thread --use_fast_math -lineinfo --ptxas-options=-allow-expensive-optimizations=true>
    )
target_link_options(mpm 
      PUBLIC       $<$<LINK_LANGUAGE:CUDA>:-dlto>
    )
set_target_properties(mpm
      PROPERTIES  CUDA_EXTENSIONS ON
                  CUDA_SEPARABLE_COMPILATION ON
                  LINKER_LANGUAGE CUDA
                  # CUDA_RESOLVE_DEVICE_SYMBOLS OFF
                  # LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                  POSITION_INDEPENDENT_CODE ON
                  CUDA_ARCHITECTURES OFF
    )
target_compile_definitions(mpm 
      PUBLIC        CMAKE_GENERATOR_PLATFORM=x64
    )

target_link_libraries(mpm PRIVATE zen zenbase zenvdb)
target_link_libraries(mpm PRIVATE zensim)

#disable default malloc options
if (WIN32)
  target_compile_options(mpm PRIVATE -DNOMINMAX)
endif()
