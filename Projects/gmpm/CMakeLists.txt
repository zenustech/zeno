cmake_minimum_required(VERSION 3.19)
project(MPM CXX)

include(CheckLanguage)
check_language(CUDA)
if(${CMAKE_CUDA_COMPILER})
    enable_language(CUDA)
    set(ZS_ENABLE_CUDA ON)
    set(ZS_ENABLE_OPENVDB OFF)
    set(ENABLE_GMPM_BUILD ON)
    add_subdirectory(zecomp)
else()
    set(ENABLE_GMPM_BUILD OFF)
endif()

#if (EXISTS zecomp AND EXISTS zecomp/include/zensim/tpls)

#message("enable gmpm:${ENABLE_GMPM_BUILD}, cuda compiler: ${CMAKE_CUDA_COMPILER}")

# if (POLICY CMP0104)
#   cmake_policy(SET CMP0104 OLD)
# endif (POLICY CMP0104)

if (ENABLE_GMPM_BUILD)

add_library(gmpm SHARED 
  nosys/ToZensimParticles.cpp 
  nosys/OutputZensimParticles.cpp 
  nosys/ConstitutiveModels.cpp 
  nosys/ToZensimLevelSet.cpp 
  nosys/ZensimBoundary.cpp 
  #nosys/Test.cu
  nosys/PartitionForParticles.cu 
  nosys/GridFromPartition.cu 
  nosys/TransferNodes.cu 
  nosys/GridOps.cu 
  nosys/ToParticleObject.cpp 
  nosys/AppendZensimObject.cpp)

if (${ZS_ENABLE_CUDA})
    enable_language(CUDA)
    target_compile_features(gmpm PRIVATE cuda_std_17)
endif()
target_compile_features(gmpm PRIVATE cuda_std_17)

target_compile_features(gmpm PRIVATE cxx_std_17)

#simd compiler flags
if (WIN32)
else()
target_compile_options(gmpm PRIVATE "-march=native")
endif()
target_compile_options(gmpm 
      PRIVATE        $<$<COMPILE_LANGUAGE:CUDA>:${CMAKE_CUDA_FLAGS} -dlto -fopenmp --expt-extended-lambda --expt-relaxed-constexpr --default-stream=per-thread --use_fast_math -lineinfo --ptxas-options=-allow-expensive-optimizations=true>
    )
target_link_options(gmpm 
      PUBLIC       $<$<LINK_LANGUAGE:CUDA>:-dlto>
    )
set_target_properties(gmpm
      PROPERTIES  CUDA_EXTENSIONS ON
                  CUDA_SEPARABLE_COMPILATION ON
                  LINKER_LANGUAGE CUDA
                  POSITION_INDEPENDENT_CODE ON
                  CUDA_ARCHITECTURES OFF
    )

find_package(zen REQUIRED)
target_link_libraries(gmpm PRIVATE zen zenbase zenvdb)
target_link_libraries(gmpm PRIVATE zecomp)
target_include_directories(gmpm PRIVATE ../zenbase/include)
target_include_directories(gmpm PRIVATE ../zenvdb/include)

endif(ENABLE_GMPM_BUILD)