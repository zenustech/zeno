#pragma once

#include <vector>
#include <string>
#include <cstdlib>
#include "eigen_data.h"
#include <cuda_runtime_api.h>

class tetrahedra_obj {
public:
	double maxVolum;
	std::vector<bool> isNBC;
	std::vector<bool> isCollide;
	std::vector<double> volum;
	std::vector<double> masses;

	double meanMass;
	double meanVolum;
	std::vector<double3> vertexes;
	std::vector<int> boundaryTypies;
	std::vector<double3> d_positions;
	std::vector<uint4> tetrahedras;
	std::vector<int4> tetrahedrasV;
	std::vector<double3> forces;
	std::vector<double3> velocities;
	std::vector<double3> d_velocities;
	std::vector<__GEIGEN__::Matrix3x3d> DM_inverse;
	std::vector<__GEIGEN__::Matrix3x3d> constraints;

	std::vector<double3> targetPos;
	std::vector<double3> tetra_fiberDir;

	std::vector<uint32_t> surfId2TetId;
	std::vector<uint3> surface;

	std::vector<uint32_t> surfVerts;
	std::vector<uint2> surfEdges;

	std::vector<double3> xTilta, dx_Elastic, acceleration;
	std::vector<double3> rest_V, V_prev;
	int D12x12Num;
	int D9x9Num;
	int D6x6Num;
	int D3x3Num;
	int vertexNum;
	int tetrahedraNum;

	int tetraheraOffset;

	double3 minTConer;
	double3 maxTConer;

	double3 minConer;
	double3 maxConer;
	tetrahedra_obj();
	//void InitMesh(int type, double scale);
	bool load_tetrahedraVtk(const std::string& filename, double scale, double3 position_offset);
	bool load_tetrahedraMesh(const std::string& filename, double scale, double3 position_offset);

	bool load_tetrahedraMesh_IPC_TetMesh(const std::string& filename, double scale, double3 position_offset);
	//void load_test(double scale, int num = 1);
	void getSurface();
	void zsGetSurface();
	bool output_tetrahedraMesh(const std::string& filename);
};