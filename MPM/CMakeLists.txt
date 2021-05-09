cmake_minimum_required(VERSION 3.19)
project(MPM)

# if (POLICY CMP0104)
#   cmake_policy(SET CMP0104 OLD)
# endif (POLICY CMP0104)
enable_language(CXX)

add_library(mpm SHARED main.cpp nosys/ToZensimParticles.cpp nosys/OutputZensimParticles.cpp)

if (${ZS_ENABLE_CUDA})
    enable_language(CUDA)
    set_target_properties(mpm PROPERTIES CUDA_ARCHITECTURES OFF)
endif()

set_target_properties(mpm
PROPERTIES  COMPILE_LANGUAGE CXX
            LINKER_LANGUAGE CXX)
target_compile_features(mpm PRIVATE cxx_std_17)
# target_link_libraries(mpm PRIVATE culib)

#simd compiler flags
if (WIN32)
#target_compile_options(FLIPlib PRIVATE "/arch:AVX")
else()
target_compile_options(mpm PRIVATE "-march=native")
endif()

target_link_libraries(mpm PRIVATE zen)
target_link_libraries(mpm PRIVATE zensim)

#disable default malloc options
if (WIN32)
  target_compile_options(mpm PRIVATE -DNOMINMAX)
endif()
