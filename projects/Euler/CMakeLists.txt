cmake_minimum_required(VERSION 3.19)
set(CMAKE_CXX_STANDARD 17)
project(zeno_Euler)
add_definitions(-D__TBB_CPP20_COMPARISONS_PRESENT=0)  

# add cmake file
include (common)

find_package(TBB CONFIG REQUIRED COMPONENTS tbb tbbmalloc)
find_package(IlmBase REQUIRED COMPONENTS Half REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenVDB REQUIRED)
add_library(zeno_Euler OBJECT nodesys.cpp)

#simd compiler flags
if (WIN32)
target_compile_options(zeno_Euler PRIVATE /arch:AVX)
else()
target_compile_options(zeno_Euler PRIVATE "-march=native")
endif()


target_include_directories(zeno_Euler PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
if (UNIX)
    target_link_directories(zeno_Euler PUBLIC /usr/lib/x86_64-linux-gnu:/usr/local/lib)
endif()

target_link_libraries(zeno_Euler PRIVATE TBB::tbb TBB::tbbmalloc)
# target_link_libraries(zeno_Euler PRIVATE bow)

#disable default malloc options
if (WIN32)
  target_compile_options(zeno_Euler PRIVATE -DNOMINMAX)
endif()

target_link_libraries(zeno_Euler PRIVATE OpenVDB::openvdb )
target_link_libraries(zeno_Euler PRIVATE Eigen3::Eigen)

target_link_libraries(zeno_Euler PUBLIC zenocore)
target_include_directories(zeno_Euler PRIVATE ../oldzenbase/include)
target_include_directories(zeno_Euler PRIVATE ../zenvdb/include)
