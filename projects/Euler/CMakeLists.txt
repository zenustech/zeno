cmake_minimum_required(VERSION 3.19)
add_definitions(-D__TBB_CPP20_COMPARISONS_PRESENT=0)  

if (NOT ZENO_WITH_zenvdb)
    message(FATAL_ERROR "zenvdb must be ON when Euler is ON! Please specify -DZENO_WITH_zenvdb:BOOL=ON")
endif()

# add cmake file
# include (common)

find_package(TBB CONFIG REQUIRED COMPONENTS tbb tbbmalloc)
find_package(Eigen3 REQUIRED)
target_sources(zeno PRIVATE nodesys.cpp)

#simd compiler flags
if (WIN32)
target_compile_options(zeno PRIVATE /arch:AVX)
else()
target_compile_options(zeno PRIVATE "-march=native")
endif()


target_include_directories(zeno PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
#if (UNIX)
    #target_link_directories(zeno PRIVATE /usr/lib/x86_64-linux-gnu:/usr/local/lib)
#endif()

target_link_libraries(zeno PRIVATE TBB::tbb TBB::tbbmalloc)
# target_link_libraries(zeno PRIVATE bow)

#disable default malloc options
if (WIN32)
  target_compile_options(zeno PRIVATE -DNOMINMAX)
endif()

#target_link_libraries(zeno PRIVATE OpenVDB::openvdb )
target_link_libraries(zeno PRIVATE Eigen3::Eigen)

target_include_directories(zeno PRIVATE ../oldzenbase/include)
target_include_directories(zeno PRIVATE ../zenvdb/include)
