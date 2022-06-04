cmake_minimum_required(VERSION 3.19)

# assume cuda
project(MPM CXX CUDA)
set(ZS_ENABLE_CUDA ON)
set(ZS_ENABLE_OPENVDB ON)
set(ZENSIM_INSTALL OFF)

add_subdirectory(zecomp)

find_package(CUDAToolkit)

add_library(zeno_zstest SHARED test.cpp)
target_compile_features(zeno_zstest PRIVATE cxx_std_17)

if (WIN32)
else()
target_compile_options(zeno_zstest PRIVATE "-march=native")
endif()
target_link_libraries(zeno_zstest PRIVATE zensim)
target_link_libraries(zeno_zstest PUBLIC zeno)
target_link_libraries(zeno_zstest PRIVATE zeno_nodep)
target_link_libraries(zeno_zstest PUBLIC ZFX)
target_include_directories(zeno_zstest PRIVATE ../oldzenbase/include)
target_include_directories(zeno_zstest PRIVATE ../zenvdb/include)

add_library(zeno_gpufem SHARED)
target_sources(zeno_gpufem PRIVATE
    femtool.cu
)

target_link_libraries(zeno_gpufem PRIVATE zensim)
target_link_libraries(zeno_gpufem PRIVATE zeno_nodep)
target_link_libraries(zeno_gpufem PUBLIC zeno ZFX)
target_include_directories(zeno_gpufem PRIVATE ../oldzenbase/include)
target_include_directories(zeno_gpufem PRIVATE ../zenvdb/include)
target_compile_options(zeno_mpm 
  PRIVATE        $<$<COMPILE_LANGUAGE:CUDA>:-Xcudafe --display_error_number,--diag_suppress=implicit_return_from_non_void_function,--diag_suppress=2642,--diag_suppress=554>
)

