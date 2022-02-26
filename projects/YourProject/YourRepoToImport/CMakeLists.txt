cmake_minimum_required(VERSION 3.18)
project(your_repo_to_import LANGUAGES CXX)

add_library(your_repo_to_import)

target_sources(your_repo_to_import 
  PRIVATE	func.cpp
)
target_include_directories(your_repo_to_import 
  PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_compile_features(your_repo_to_import PUBLIC cxx_std_17)
set_target_properties(your_repo_to_import
	PROPERTIES	POSITION_INDEPENDENT_CODE ON
)

# optional setup
find_package(OpenMP)

if (TARGET OpenMP::OpenMP_CXX)
target_link_libraries(your_repo_to_import 
  PUBLIC OpenMP::OpenMP_CXX)
target_compile_options(your_repo_to_import 
  PUBLIC     $<$<COMPILE_LANGUAGE:CXX>: 
  $<IF:$<CXX_COMPILER_ID:MSVC>,/O2  /openmp:llvm /openmp:experimental, 
  	-O3 -fopenmp $<IF:$<CXX_COMPILER_ID:Clang>, 
	  	, > >>
)
endif(TARGET OpenMP::OpenMP_CXX)
