#find_package(Eigen3 REQUIRED)
#find_package(TBB REQUIRED)
#find_package(OpenVDB REQUIRED)
#find_package(Half REQUIRED)
#find_package(CUDA REQUIRED)
#find_package(OpenVDB REQUIRED)
#
#add_library(eigen INTERFACE IMPORTED)
#include_directories(${OPENVDB_INCLUDE_DIRS}/openvdb/math/)
#
#include_directories(${CUDA_INCLUDE_DIRS})
#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -std=c++11 -ccbin=g++-4.8 -DFMMTL_KERNEL -arch=sm_70 --compiler-options "-fopenmp -fPIC -funroll-loops -O3 -Wall -Wextra -Wno-unused-local-typedefs" -Xcompiler -Wno-unused-parameter)
#
#add_library(laplace_lib STATIC ${CMAKE_CURRENT_SOURCE_DIR}/fmmtl/kernel/Laplace.kern.cu)
#target_compile_options(laplace_lib PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_NVCC_FLAGS}>)
#
#add_executable(FLIP main.cpp fluidsim.cpp GeometricLevelGen.cpp
#        levelset_util.cpp octree3.cpp predicates.cxx
#        tetgen.cxx BEMBasic.h BiCGSTAB.h eigenheaders.h FMMWrapper.h LaplaceBEM.h)
#
#target_compile_options(FLIP PRIVATE -std=c++11 -fopenmp -fPIC -funroll-loops -O3 -Wall -Wextra -Wno-unused-local-typedefs -DFMMTL_WITH_CUDA -lcudart)
#target_include_directories(FLIP SYSTEM PUBLIC ${EIGEN3_INCLUDE_DIR})
#target_link_libraries(FLIP ${TBB_LIBRARIES})
#target_link_libraries(FLIP ${OPENVDB_LIBRARIES})
#target_link_libraries(FLIP Half)
#target_link_libraries(FLIP partio)
#target_link_libraries(FLIP eigen)
#target_link_libraries(FLIP laplace_lib)
#target_link_libraries(FLIP ${CUDA_LIBRARIES})

find_package(TBB REQUIRED)
find_package(OpenVDB REQUIRED)
find_package(ZLIB)
if (WIN32)
find_package(IlmBase REQUIRED COMPONENTS Half REQUIRED)
else()
find_package(Half REQUIRED)
endif()


add_executable(FLIP main.cpp fluidsim.cpp GeometricLevelGen.cpp
        levelset_util.cpp octree3.cpp predicates.cxx
        tetgen.cxx fluidsim_fusion_p2g.cpp)
        
#target_include_directories(FLIP PUBLIC ${OPENVDB_INCLUDE_DIRS}/openvdb/math/)  

if(WIN32)
target_include_directories(FLIP PUBLIC ${OPENVDB_INCLUDE_DIRS})  
endif()

target_link_libraries(FLIP PUBLIC ${TBB_LIBRARIES})
target_link_libraries(FLIP PUBLIC ${TBB_MALLOC_PROXY} ${TBB_MALLOC})
target_link_libraries(FLIP PUBLIC ${OPENVDB_LIBRARIES})
target_link_libraries(FLIP PUBLIC reactphysics3d)
if (WIN32)
target_link_libraries(FLIP PUBLIC IlmBase::Half)
else()
target_link_libraries(FLIP PUBLIC Half)
endif()
target_link_libraries(FLIP PUBLIC partio)
target_link_libraries(FLIP PUBLIC BEM)
target_link_libraries(FLIP PUBLIC LosTopos)

target_link_libraries(FLIP PUBLIC project_cxx_dependencies)
