cmake_minimum_required(VERSION 3.19)

#set(ENABLE_GMPM_BUILD ON)
set(ENABLE_GMPM_BUILD ON)

if (ENABLE_GMPM_BUILD)

project(MPM CXX CUDA)
set(ZS_ENABLE_CUDA ON)
set(ZS_ENABLE_OPENVDB ON)
set(ZENSIM_INSTALL OFF)
add_subdirectory(zecomp)

#if (EXISTS zecomp AND EXISTS zecomp/include/zensim/tpls)

#message("enable zeno_gmpm:${ENABLE_GMPM_BUILD}, cuda compiler: ${CMAKE_CUDA_COMPILER}")

# if (POLICY CMP0104)
#   cmake_policy(SET CMP0104 OLD)
# endif (POLICY CMP0104)

add_library(zeno_gmpm SHARED)

target_sources(zeno_gmpm PRIVATE
  nosys/ToZensimParticles.cpp 
  nosys/OutputZensimParticles.cpp 
  nosys/ConstitutiveModels.cpp 
  nosys/ToZensimLevelSet.cpp 
  nosys/ZensimBoundary.cpp 
  nosys/MakePartitionAndGrid.cpp 
  nosys/MakeIndexBuckets.cpp 

  #nosys/Test.cu
  nosys/PartitionForParticles.cu 
  nosys/GridFromPartition.cu 
  nosys/TransferNodes.cu 
  nosys/GridOps.cu 
  nosys/ToParticleObject.cpp 
  nosys/AppendZensimObject.cpp)

if (NOT MSVC)
target_sources(zeno_gmpm PRIVATE 
  nosys/ZensimParticleWrangler.cu
  nosys/ZensimParticleNeighborWrangler.cu
)
endif()

if (${ZS_ENABLE_CUDA})
    enable_language(CUDA)
    target_compile_features(zeno_gmpm PRIVATE cuda_std_17)
endif()
target_compile_features(zeno_gmpm PRIVATE cuda_std_17)

target_compile_features(zeno_gmpm PRIVATE cxx_std_17)

#simd compiler flags
if (WIN32)
else()
target_compile_options(zeno_gmpm PRIVATE "-march=native")
endif()
target_compile_options(zeno_gmpm 
      PRIVATE        $<$<COMPILE_LANGUAGE:CUDA>:${CMAKE_CUDA_FLAGS} --expt-extended-lambda --expt-relaxed-constexpr --default-stream=per-thread --use_fast_math -lineinfo --ptxas-options=-allow-expensive-optimizations=true>
    )
target_link_options(zeno_gmpm 
      PUBLIC       $<$<LINK_LANGUAGE:CUDA>:-dlto>
    )
set_target_properties(zeno_gmpm
      PROPERTIES  CUDA_EXTENSIONS ON
                  CUDA_SEPARABLE_COMPILATION ON
                  LINKER_LANGUAGE CUDA
                  POSITION_INDEPENDENT_CODE ON
                  CUDA_ARCHITECTURES OFF
    )

target_link_libraries(zeno_gmpm PRIVATE zensim)
target_link_libraries(zeno_gmpm PUBLIC zeno)
target_link_libraries(zeno_gmpm PRIVATE zeno_nodep)
target_link_libraries(zeno_gmpm PUBLIC ZFX)
target_include_directories(zeno_gmpm PRIVATE ../oldzenbase/include)
target_include_directories(zeno_gmpm PRIVATE ../zenvdb/include)

endif(ENABLE_GMPM_BUILD)
