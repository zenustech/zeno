
cmake_minimum_required(VERSION 3.19)
set(CMAKE_CXX_STANDARD 17)
project(ZenEulerLib)
add_definitions(-D__TBB_CPP20_COMPARISONS_PRESENT=0)  

# add cmake file
include (common)

find_package(TBB CONFIG REQUIRED COMPONENTS tbb tbbmalloc)
find_package(IlmBase REQUIRED COMPONENTS Half REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenVDB REQUIRED)
add_subdirectory(Hades)
add_library(ZenEulerLib EulerTest.cpp)

#simd compiler flags
if (WIN32)
else()
target_compile_options(ZenEulerLib PRIVATE "-march=native")
endif()


target_include_directories(ZenEulerLib PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_directories(ZenEulerLib PUBLIC /usr/lib/x86_64-linux-gnu:/usr/local/lib)

target_link_libraries(ZenEulerLib PRIVATE TBB::tbb TBB::tbbmalloc)
target_link_libraries(ZenEulerLib PRIVATE bow)

#disable default malloc options
if (WIN32)
  target_compile_options(ZenEulerLib PRIVATE -DNOMINMAX)
endif()

target_link_libraries(ZenEulerLib PRIVATE OpenVDB::openvdb )
target_link_libraries(ZenEulerLib PRIVATE Eigen3::Eigen)


find_package(zeno REQUIRED)
target_link_libraries(ZenEulerLib PRIVATE zeno)
set_target_properties(ZenEulerLib PROPERTIES
	LIBRARY_OUTPUT_DIRECTORY ${zeno_AUTOLOAD_DIR}
	RUNTIME_OUTPUT_DIRECTORY ${zeno_AUTOLOAD_DIR}
	ARCHIVE_OUTPUT_DIRECTORY ${zeno_AUTOLOAD_DIR}
	LIBRARY_OUTPUT_DIRECTORY_DEBUG ${zeno_AUTOLOAD_DIR}
	RUNTIME_OUTPUT_DIRECTORY_DEBUG ${zeno_AUTOLOAD_DIR}
	ARCHIVE_OUTPUT_DIRECTORY_DEBUG ${zeno_AUTOLOAD_DIR}
	LIBRARY_OUTPUT_DIRECTORY_RELEASE ${zeno_AUTOLOAD_DIR}
	RUNTIME_OUTPUT_DIRECTORY_RELEASE ${zeno_AUTOLOAD_DIR}
	ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${zeno_AUTOLOAD_DIR}
)
target_include_directories(ZenEulerLib PRIVATE ../oldzenbase/include)
target_include_directories(ZenEulerLib PRIVATE ../zenvdb/include)

