#ifndef  __DEVICE_FEM_MESHES_CUH__
#define __DEVICE_FEM_MESHES_CUH__

//#include <cuda_runtime.h>
#include "../Structures.hpp"
#include "gpu_eigen_libs.cuh"
#include "zensim/container/TileVector.hpp"
#include <cstdint>
#include "mesh.h"

using tiles_t = zs::TileVector<float, 32>;
using dtiles_t = zs::TileVector<double, 32>;

class device_TetraData{
public:
	tiles_t verts, eles;
	dtiles_t vtemp, etemp;
	std::size_t numVerts, numEles;

  	zeno::ZenoConstitutiveModel model{};

	double3* vertexes;
	double3* o_vertexes;
	double3* rest_vertexes;
	double3* temp_double3Mem;
	double3* velocities;
	double3* xTilta;
	double3* fb;
	uint4* tetrahedras;
	uint4* tempTetrahedras;
	double* volum;
	//double* tempV;
	double* masses;
	double* tempDouble;
	uint64_t* MChash;
	uint32_t* sortIndex;
	uint32_t* sortMapVertIndex;
	//uint32_t* sortTetIndex;
	__GEIGEN__::Matrix3x3d* DmInverses;
	__GEIGEN__::Matrix3x3d* Constraints;
	int* BoundaryType;
	int* tempBoundaryType;
	//__GEIGEN__::Matrix3x3d* tempDmInverses;
	__GEIGEN__::Matrix3x3d* tempMat3x3;

public:
	device_TetraData() {}
	~device_TetraData();
	void init(const tetrahedra_obj&);
	void retrieve();
	void commit();
	void Malloc_DEVICE_MEM(const int& vertex_num, const int& tetradedra_num);
	void FREE_DEVICE_MEM();
};

#endif // ! __DEVICE_FEM_MESHES_CUH__
