#pragma once
#ifndef _PCG_SOLVER_CUH_
#define _PCG_SOLVER_CUH_
#include <cuda_runtime.h>
#include "device_fem_data.cuh"
#include <cstdint>
class BHessian {
public:
	uint32_t* D1Index;//pIndex, DpeIndex, DptIndex;
	uint3* D3Index;
	uint4* D4Index;
	uint2* D2Index;
	__GEIGEN__::Matrix12x12d* H12x12;
	__GEIGEN__::Matrix3x3d* H3x3;
	__GEIGEN__::Matrix6x6d* H6x6;
	__GEIGEN__::Matrix9x9d* H9x9;

	uint32_t DNum[4];

public:
	BHessian() {}
	~BHessian() {};
	void updateDNum(const int& tet_number, const uint32_t* cpNums, const uint32_t* last_cpNums);
	void MALLOC_DEVICE_MEM_O(const int& tet_number, const int& surfvert_number, const int& surface_number, const int& edge_number);
	void FREE_DEVICE_MEM();
	//void init(const int& edgeNum, const int& faceNum, const int& vertNum);
};

class PCG_Data {
public:
	double* squeue;
	double3* b;
	__GEIGEN__::Matrix3x3d* P;
	double3* r;
	double3* c;
	double3* q;
	double3* s;
	double3* z;
	double3* dx;
	double3* tempDx;
public:
	void Malloc_DEVICE_MEM(const int& vertex_num, const int& tetradedra_num);
	void FREE_DEVICE_MEM();
};

bool PCG_Process(device_TetraData* mesh, PCG_Data* pcg_data, const BHessian& BH, double3* _mvDir, int vertexNum, int tetrahedraNum, double IPC_dt, double meanVolumn);

#endif // ! _PCG_SOLVER_CUH_
