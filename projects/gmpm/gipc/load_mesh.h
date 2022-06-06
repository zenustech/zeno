#pragma once
//#include "Eigen/Eigen"
//#include "mIPC.h"
#include <cuda_runtime.h>
#include <sstream>
#include "mesh.h"

using namespace std;

class mesh_obj {
public:
	vector<double3> vertexes;
	vector<double3> normals;
	vector<uint3> facenormals;
	vector<uint3> faces;
	vector<uint2> edges;
	int vertexNum;
	int faceNum;
	int edgeNum;
	//void InitMesh(int type, double scale);
	bool load_mesh(const std::string& filename, double scale, double3 transform);
};

