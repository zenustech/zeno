#include <iostream>
#include <cstring>
#include "load_mesh.h"
#include <unordered_map>
#include <fstream>
#include <set>
#include <queue>
#include<iostream>
#include "gpu_eigen_libs.cuh"
#include "zensim/io/MeshIO.hpp"
#include "zensim/omp/execution/ExecutionPolicy.hpp"
#include "zensim/container/HashTable.hpp"
#include "zensim/container/Vector.hpp"

using namespace std;

class Triangle {

public:
	uint64_t key[3];

	Triangle(const uint64_t* p_key)
	{
		key[0] = p_key[0];
		key[1] = p_key[1];
		key[2] = p_key[2];
	}
	Triangle(uint64_t key0, uint64_t key1, uint64_t key2)
	{
		key[0] = key0;
		key[1] = key1;
		key[2] = key2;
	}

	uint64_t operator[](int i) const
	{
		//assert(0 <= i && i <= 2);
		return key[i];
	}

	bool operator<(const Triangle& right) const
	{
		if (key[0] < right.key[0]) {
			return true;
		}
		else if (key[0] == right.key[0]) {
			if (key[1] < right.key[1]) {
				return true;
			}
			else if (key[1] == right.key[1]) {
				if (key[2] < right.key[2]) {
					return true;
				}
			}
		}
		return false;
	}

	bool operator==(const Triangle& right) const
	{
		return key[0] == right[0] && key[1] == right[1] && key[2] == right[2];
	}
};

void split(string str, vector<string>& v, string spacer)
{
	int pos1, pos2;
	int len = spacer.length();
	pos1 = 0;
	pos2 = str.find(spacer);
	while (pos2 != string::npos)
	{
		v.push_back(str.substr(pos1, pos2 - pos1));
		pos1 = pos2 + len;
		pos2 = str.find(spacer, pos1);
	}
	if (pos1 != str.length())
		v.push_back(str.substr(pos1));
}

bool mesh_obj::load_mesh(const std::string& filename, double scale, double3 transform) {
	ifstream ifs(filename);
	if (!ifs) {

		fprintf(stderr, "unable to read file %s\n", filename.c_str());
		ifs.close();
		exit(-1);
		return false;
	}
	char buffer[1024];
	string line = "";
	int nodeNumber = 0;
	int elementNumber = 0;
	double x, y, z;

	while (getline(ifs, line)) {
		string key = line.substr(0, 2);
		if (key.length() <= 1) continue;
		stringstream ss(line.substr(2));
		if (key == "v ") {
			ss >> x >> y >> z;
			double3 vertex = make_double3(scale * x + transform.x, scale * y + transform.y, scale * z + transform.z);
			vertexes.push_back(vertex);
		}
		else if (key == "vn") {
			ss >> x >> y >> z;
			double3 normal = make_double3(x, y, z);
			normals.push_back(normal);
		}
		else if (key == "f ") {
			if (line.length() >= 1024) {
				printf("[WARN]: skip line due to exceed max buffer length (1024).\n");
				continue;
			}

			std::vector<string> fs;

			{
				string buf;
				stringstream ss(line);
				vector<string> tokens;
				while (ss >> buf)
					tokens.push_back(buf);

				for (size_t index = 3; index < tokens.size(); index += 1) {
					fs.push_back("f " + tokens[1] + " " + tokens[index - 1] + " " + tokens[index]);
				}
			}

			int uv0, uv1, uv2;

			for (const auto& f : fs) {
				memset(buffer, 0, sizeof(char) * 1024);
				std::copy(f.begin(), f.end(), buffer);

				uint3 faceVertIndex;
				uint3 faceNormalIndex;

				if (sscanf(buffer, "f %d/%d/%d %d/%d/%d %d/%d/%d", &faceVertIndex.x, &uv0, &faceNormalIndex.x,
					&faceVertIndex.y, &uv1, &faceNormalIndex.y,
					&faceVertIndex.z, &uv2, &faceNormalIndex.z) == 9) {

					faceVertIndex.x -= 1;
					faceVertIndex.y -= 1;
					faceVertIndex.z -= 1;
					faces.push_back(faceVertIndex);
					facenormals.push_back(faceNormalIndex);
				}
				else if (sscanf(buffer, "f %d %d %d", &faceVertIndex.x,
					&faceVertIndex.y,
					&faceVertIndex.z) == 3) {
					faceVertIndex.x -= 1;
					faceVertIndex.y -= 1;
					faceVertIndex.z -= 1;
					faces.push_back(faceVertIndex);
				}
				else if (sscanf(buffer, "f %d/%d %d/%d %d/%d", &faceVertIndex.x, &uv0,
					&faceVertIndex.y, &uv1,
					&faceVertIndex.z, &uv2) == 6) {
					faceVertIndex.x -= 1;
					faceVertIndex.y -= 1;
					faceVertIndex.z -= 1;
					faces.push_back(faceVertIndex);
				}

			}
		}
	}

	vertexNum = vertexes.size();
	faceNum = faces.size();
	edgeNum = 0;
	set<pair<uint32_t, uint32_t>> SFEdges_set;
	for (const auto& cTri : faces) {
		for (int i = 0;i < 3;i++) {
			if (SFEdges_set.find(pair<uint32_t, uint32_t>(cTri.y, cTri.x)) == SFEdges_set.end() && SFEdges_set.find(pair<uint32_t, uint32_t>(cTri.x, cTri.y)) == SFEdges_set.end()) {
				SFEdges_set.insert(pair<uint32_t, uint32_t>(cTri.x, cTri.y));
				edgeNum++;
			}
			if (SFEdges_set.find(pair<uint32_t, uint32_t>(cTri.z, cTri.y)) == SFEdges_set.end() && SFEdges_set.find(pair<uint32_t, uint32_t>(cTri.y, cTri.z)) == SFEdges_set.end()) {
				SFEdges_set.insert(pair<uint32_t, uint32_t>(cTri.y, cTri.z));
				edgeNum++;
			}
			if (SFEdges_set.find(pair<uint32_t, uint32_t>(cTri.x, cTri.z)) == SFEdges_set.end() && SFEdges_set.find(pair<uint32_t, uint32_t>(cTri.z, cTri.x)) == SFEdges_set.end()) {
				SFEdges_set.insert(pair<uint32_t, uint32_t>(cTri.z, cTri.x));
				edgeNum++;
			}
		}
	}

	vector<pair<uint32_t, uint32_t>> temp_edges = vector<pair<uint32_t, uint32_t>>(SFEdges_set.begin(), SFEdges_set.end());

	for (int i = 0;i < edgeNum;i++) {
		edges.push_back(make_uint2(temp_edges[i].first, temp_edges[i].second));
	}
	return true;
}

tetrahedra_obj::tetrahedra_obj() {
	tetraheraOffset = 0;
	vertexNum = 0;
	tetrahedraNum = 0;
	minConer = make_double3(0, 0, 0);
	maxConer = make_double3(0, 0, 0);
}

bool tetrahedra_obj::load_tetrahedraVtk(const std::string& filename, double scale, double3 position_offset) {
	zs::Mesh<float, 3, int, 4> tet;
	read_tet_mesh_vtk(filename, tet);
	const auto numVerts = tet.nodes.size();
	const auto numEles = tet.elems.size();
	// auto ompExec = zs::omp_exec();

	vertexNum += numVerts;
	/// verts
	double xmin = std::numeric_limits<double>::max(), ymin = std::numeric_limits<double>::max(), zmin = std::numeric_limits<double>::max();
	double xmax = std::numeric_limits<double>::lowest(), ymax = std::numeric_limits<double>::lowest(), zmax = std::numeric_limits<double>::lowest();
	for (int vi = 0; vi != numVerts; ++vi) {
		auto [x, y, z] = tet.nodes[vi];
		boundaryTypies.push_back(0);
		auto vertex = make_double3(scale * x - position_offset.x, scale * y - position_offset.y, scale * z - position_offset.z);
		vertexes.push_back(vertex);
		velocities.push_back(make_double3(0, 0, 0));
		d_velocities.push_back(make_double3(0, 0, 0));
		masses.push_back(0);
		d_positions.push_back(make_double3(0, 0, 0));

		__GEIGEN__::Matrix3x3d constraint;
		__GEIGEN__::__set_Mat_val(constraint, 1, 0, 0, 0, 1, 0, 0, 0, 1);

		constraints.push_back(constraint);

		if (xmin > vertex.x) xmin = vertex.x;
		if (ymin > vertex.y) ymin = vertex.y;
		if (zmin > vertex.z) zmin = vertex.z;
		if (xmax < vertex.x) xmax = vertex.x;
		if (ymax < vertex.y) ymax = vertex.y;
		if (zmax < vertex.z) zmax = vertex.z;
	};
	minTConer = make_double3(xmin, ymin, zmin);
	maxTConer = make_double3(xmax, ymax, zmax);

	tetrahedraNum += numEles;
	for (int ei = 0; ei != numEles; ++ei) {
		auto quad = tet.elems[ei];
		uint4 tetrahedra;
		tetrahedra.x = quad[0] + tetraheraOffset;
		tetrahedra.y = quad[1] + tetraheraOffset;
		tetrahedra.z = quad[2] + tetraheraOffset;
		tetrahedra.w = quad[3] + tetraheraOffset;
		tetrahedras.push_back(tetrahedra);
		tetra_fiberDir.push_back(make_double3(0, 0, 0));;
	}

	double boxTVolum = (maxTConer.x - minTConer.x) * (maxTConer.y - minTConer.y) * (maxTConer.z - minTConer.z);
	double boxVolum = (maxConer.x - minConer.x) * (maxConer.y - minConer.y) * (maxConer.z - minConer.z);

	if (boxTVolum > boxVolum) {
		maxConer = maxTConer;
		minConer = minTConer;
	}
	//V_prev = vertexes;
	tetraheraOffset = vertexNum;
	D12x12Num = 0;
	D9x9Num = 0;
	D6x6Num = 0;
	D3x3Num = 0;


	return true;
}
bool tetrahedra_obj::load_tetrahedraMesh(const std::string& filename, double scale, double3 position_offset) {

	ifstream ifs(filename);
	if (!ifs) {

		fprintf(stderr, "unable to read file %s\n", filename.c_str());
		ifs.close();
		exit(-1);
		return false;
	}

	double x, y, z;
	int index0, index1, index2, index3;
	string line = "";
	int nodeNumber = 0;
	int elementNumber = 0;
	while (getline(ifs, line)) {
		if (line.length() <= 1) continue;
		if (line == "$Nodes") {
			getline(ifs, line);
			nodeNumber = atoi(line.c_str());
			vertexNum += nodeNumber;

			double xmin = std::numeric_limits<double>::max(), ymin = std::numeric_limits<double>::max(), zmin = std::numeric_limits<double>::max();
			double xmax = std::numeric_limits<double>::lowest(), ymax = std::numeric_limits<double>::lowest(), zmax = std::numeric_limits<double>::lowest();
			for (int i = 0; i < nodeNumber; i++) {
				getline(ifs, line);
				vector<std::string> nodePos;
				std::string spacer = " ";
				split(line, nodePos, spacer);
				x = atof(nodePos[1].c_str());
				y = atof(nodePos[2].c_str());
				z = atof(nodePos[3].c_str());
				double3 d_velocity = make_double3(0, 0, 0);
				double3 vertex = make_double3(scale * x - position_offset.x, scale * y - position_offset.y, scale * z - position_offset.z);
				//Matrix3d Constraint; Constraint.setIdentity();
				//Vector3d force = Vector3d(0, 0, 0);
				double3 velocity = make_double3(0, 0, 0);
				double3 d_pos = make_double3(0, 0, 0);
				double mass = 0;
				int boundaryType = 0;
				boundaryTypies.push_back(boundaryType);
				vertexes.push_back(vertex);
				//forces.push_back(force);
				velocities.push_back(velocity);
				//Constraints.push_back(Constraint);
				//isNBC.push_back(false);
				d_velocities.push_back(d_velocity);
				masses.push_back(mass);
				//isDelete.push_back(false);
				d_positions.push_back(d_pos);
				//externalForce.push_back(Vector3d(0, 0, 0));


				__GEIGEN__::Matrix3x3d constraint;
				__GEIGEN__::__set_Mat_val(constraint, 1, 0, 0, 0, 1, 0, 0, 0, 1);

				constraints.push_back(constraint);

				if (xmin > vertex.x) xmin = vertex.x;
				if (ymin > vertex.y) ymin = vertex.y;
				if (zmin > vertex.z) zmin = vertex.z;
				if (xmax < vertex.x) xmax = vertex.x;
				if (ymax < vertex.y) ymax = vertex.y;
				if (zmax < vertex.z) zmax = vertex.z;
			}
			minTConer = make_double3(xmin, ymin, zmin);
			maxTConer = make_double3(xmax, ymax, zmax);
		}
		
		if (line == "$Elements") {
			getline(ifs, line);
			elementNumber = atoi(line.c_str());
			tetrahedraNum += elementNumber;
			for (int i = 0; i < elementNumber; i++) {
				getline(ifs, line);

				vector<std::string> elementIndexex;
				std::string spacer = " ";
				split(line, elementIndexex, spacer);
				index0 = atoi(elementIndexex[3].c_str()) - 1;
				index1 = atoi(elementIndexex[4].c_str()) - 1;
				index2 = atoi(elementIndexex[5].c_str()) - 1;
				index3 = atoi(elementIndexex[6].c_str()) - 1;

				uint4 tetrahedra;
				tetrahedra.x = index0 + tetraheraOffset;
				tetrahedra.y = index1 + tetraheraOffset;
				tetrahedra.z = index2 + tetraheraOffset;
				tetrahedra.w = index3 + tetraheraOffset;
				tetrahedras.push_back(tetrahedra);
				tetra_fiberDir.push_back(make_double3(0, 0, 0));

				
			}
			break;
		}
	}
	ifs.close();

	double boxTVolum = (maxTConer.x - minTConer.x) * (maxTConer.y - minTConer.y) * (maxTConer.z - minTConer.z);
	double boxVolum = (maxConer.x - minConer.x) * (maxConer.y - minConer.y) * (maxConer.z - minConer.z);

	if (boxTVolum > boxVolum) {
		maxConer = maxTConer;
		minConer = minTConer;
	}
	//V_prev = vertexes;
	tetraheraOffset = vertexNum;
	D12x12Num = 0;
	D9x9Num = 0;
	D6x6Num = 0;
	D3x3Num = 0;


	return true;
}

bool tetrahedra_obj::load_tetrahedraMesh_IPC_TetMesh(const std::string& filename, double scale, double3 position_offset) {

	ifstream ifs(filename);
	if (!ifs) {

		fprintf(stderr, "unable to read file %s\n", filename.c_str());
		ifs.close();
		exit(-1);
		return false;
	}

	double x, y, z;
	int index0, index1, index2, index3;
	string line = "";
	int nodeNumber = 0;
	int elementNumber = 0;
	while (getline(ifs, line)) {
		if (line.length() <= 1) continue;
		if (line == "$Nodes") {
			getline(ifs, line);
			nodeNumber = atoi(line.c_str());
			vertexNum += nodeNumber;

			double xmin = std::numeric_limits<double>::max(), ymin = std::numeric_limits<double>::max(), zmin = std::numeric_limits<double>::max();
			double xmax = std::numeric_limits<double>::lowest(), ymax = std::numeric_limits<double>::lowest(), zmax = std::numeric_limits<double>::lowest();
			for (int i = 0; i < nodeNumber; i++) {
				getline(ifs, line);
				vector<std::string> nodePos;
				std::string spacer = " ";
				split(line, nodePos, spacer);
				x = atof(nodePos[0].c_str());
				y = atof(nodePos[1].c_str());
				z = atof(nodePos[2].c_str());
				double3 d_velocity = make_double3(0, 0, 0);
				double3 vertex = make_double3(scale * x - position_offset.x, scale * y - position_offset.y, scale * z - position_offset.z);
				//Matrix3d Constraint; Constraint.setIdentity();
				//Vector3d force = Vector3d(0, 0, 0);
				double3 velocity = make_double3(0, 0, 0);
				double3 d_pos = make_double3(0, 0, 0);
				double mass = 0;
				int boundaryType = 0;
				boundaryTypies.push_back(boundaryType);
				vertexes.push_back(vertex);
				//forces.push_back(force);
				velocities.push_back(velocity);
				//Constraints.push_back(Constraint);
				//isNBC.push_back(false);
				d_velocities.push_back(d_velocity);
				masses.push_back(mass);
				//isDelete.push_back(false);
				d_positions.push_back(d_pos);
				//externalForce.push_back(Vector3d(0, 0, 0));


				__GEIGEN__::Matrix3x3d constraint;
				__GEIGEN__::__set_Mat_val(constraint, 1, 0, 0, 0, 1, 0, 0, 0, 1);

				constraints.push_back(constraint);

				if (xmin > vertex.x) xmin = vertex.x;
				if (ymin > vertex.y) ymin = vertex.y;
				if (zmin > vertex.z) zmin = vertex.z;
				if (xmax < vertex.x) xmax = vertex.x;
				if (ymax < vertex.y) ymax = vertex.y;
				if (zmax < vertex.z) zmax = vertex.z;
			}
			minTConer = make_double3(xmin, ymin, zmin);
			maxTConer = make_double3(xmax, ymax, zmax);
		}

		if (line == "$Elements") {
			getline(ifs, line);
			elementNumber = atoi(line.c_str());
			tetrahedraNum += elementNumber;
			for (int i = 0; i < elementNumber; i++) {
				getline(ifs, line);

				vector<std::string> elementIndexex;
				std::string spacer = " ";
				split(line, elementIndexex, spacer);
				index0 = atoi(elementIndexex[1].c_str()) - 1;
				index1 = atoi(elementIndexex[2].c_str()) - 1;
				index2 = atoi(elementIndexex[3].c_str()) - 1;
				index3 = atoi(elementIndexex[4].c_str()) - 1;

				uint4 tetrahedra;
				tetrahedra.x = index0 + tetraheraOffset;
				tetrahedra.y = index1 + tetraheraOffset;
				tetrahedra.z = index2 + tetraheraOffset;
				tetrahedra.w = index3 + tetraheraOffset;
				tetrahedras.push_back(tetrahedra);
				tetra_fiberDir.push_back(make_double3(0, 0, 0));


			}
			break;
		}
	}
	ifs.close();

	double boxTVolum = (maxTConer.x - minTConer.x) * (maxTConer.y - minTConer.y) * (maxTConer.z - minTConer.z);
	double boxVolum = (maxConer.x - minConer.x) * (maxConer.y - minConer.y) * (maxConer.z - minConer.z);

	if (boxTVolum > boxVolum) {
		maxConer = maxTConer;
		minConer = minTConer;
	}
	//V_prev = vertexes;
	tetraheraOffset = vertexNum;
	D12x12Num = 0;
	D9x9Num = 0;
	D6x6Num = 0;
	D3x3Num = 0;


	return true;
}


void tetrahedra_obj::getSurface() {
	uint64_t length = vertexNum;
	auto triangle_hash = [&](const Triangle& tri) {
		return length * (length * tri[0] + tri[1]) + tri[2];
	};
	//vector<Vector4i> surface;
	std::unordered_map<Triangle, uint64_t, decltype(triangle_hash)> tri2Tet(4 * tetrahedraNum, triangle_hash);
	for (int i = 0;i < tetrahedraNum;i++) {

		const auto& triI4 = tetrahedras[i];
		uint64_t triI[4] = { triI4.x,  triI4.y ,triI4.z ,triI4.w };
		for (int j = 0;j < 4;j++) {
			const Triangle& triVInd = Triangle(triI[j % 4], triI[(1 + j) % 4], triI[(2 + j) % 4]);
			if (tri2Tet.find(Triangle(triVInd[0], triVInd[1], triVInd[2])) != tri2Tet.end()) {
				tri2Tet[Triangle(triVInd[0], triVInd[1], triVInd[2])] = tetrahedraNum + 1;
			}
			else if (tri2Tet.find(Triangle(triVInd[0], triVInd[2], triVInd[1])) != tri2Tet.end()) {
				tri2Tet[Triangle(triVInd[0], triVInd[2], triVInd[1])] = tetrahedraNum + 1;
			}
			else if (tri2Tet.find(Triangle(triVInd[1], triVInd[0], triVInd[2])) != tri2Tet.end()) {
				tri2Tet[Triangle(triVInd[1], triVInd[0], triVInd[2])] = tetrahedraNum + 1;
			}
			else if (tri2Tet.find(Triangle(triVInd[1], triVInd[2], triVInd[0])) != tri2Tet.end()) {
				tri2Tet[Triangle(triVInd[1], triVInd[2], triVInd[0])] = tetrahedraNum + 1;
			}
			else if (tri2Tet.find(Triangle(triVInd[2], triVInd[0], triVInd[1])) != tri2Tet.end()) {
				tri2Tet[Triangle(triVInd[2], triVInd[0], triVInd[1])] = tetrahedraNum + 1;
			}
			else if (tri2Tet.find(Triangle(triVInd[2], triVInd[1], triVInd[0])) != tri2Tet.end()) {
				tri2Tet[Triangle(triVInd[2], triVInd[1], triVInd[0])] = tetrahedraNum + 1;
			}
			else {
				tri2Tet[Triangle(triVInd[0], triVInd[1], triVInd[2])] = i;
			}
		}
	}

	for (const auto& triI : tri2Tet) {
		const uint64_t& tetId = triI.second;
		const Triangle& triVInd = triI.first;
		if (tetId < tetrahedraNum) {
			double3 vec1 = __GEIGEN__::__minus(vertexes[triVInd[1]], vertexes[triVInd[0]]);
			double3 vec2 = __GEIGEN__::__minus(vertexes[triVInd[2]], vertexes[triVInd[0]]);
			int id3 = 0;

			if (tetrahedras[tetId].x != triVInd[0]
				&& tetrahedras[tetId].x != triVInd[1]
				&& tetrahedras[tetId].x != triVInd[2]) {
				id3 = tetrahedras[tetId].x;
			}
			else if (tetrahedras[tetId].y != triVInd[0]
				&& tetrahedras[tetId].y != triVInd[1]
				&& tetrahedras[tetId].y != triVInd[2]) {
				id3 = tetrahedras[tetId].y;
			}
			else if (tetrahedras[tetId].z != triVInd[0]
				&& tetrahedras[tetId].z != triVInd[1]
				&& tetrahedras[tetId].z != triVInd[2]) {
				id3 = tetrahedras[tetId].z;
			}
			else if (tetrahedras[tetId].w != triVInd[0]
				&& tetrahedras[tetId].w != triVInd[1]
				&& tetrahedras[tetId].w != triVInd[2]) {
				id3 = tetrahedras[tetId].w;
			}


			double3 vec3 = __GEIGEN__::__minus(vertexes[id3], vertexes[triVInd[0]]);
			double3 n = __GEIGEN__::__v_vec_cross(vec1, vec2);
			if (__GEIGEN__::__v_vec_dot(n, vec3) > 0) {
				surface.push_back(make_uint3(triVInd[0], triVInd[2], triVInd[1]));
			}
			else {
				surface.push_back(make_uint3(triVInd[0], triVInd[1], triVInd[2]));
			}
		}
	}

	vector<bool> flag(vertexNum, false);
	for (const auto& cTri : surface) {

		if (!flag[cTri.x]) {
			surfVerts.push_back(cTri.x);
			flag[cTri.x] = true;
		}
		if (!flag[cTri.y]) {
			surfVerts.push_back(cTri.y);
			flag[cTri.y] = true;
		}
		if (!flag[cTri.z]) {
			surfVerts.push_back(cTri.z);
			flag[cTri.z] = true;
		}

	}

	set<pair<uint64_t, uint64_t>> SFEdges_set;
	for (const auto& cTri : surface) {
		for (int i = 0;i < 3;i++) {
			if (SFEdges_set.find(pair<uint64_t, uint64_t>(cTri.y, cTri.x)) == SFEdges_set.end()) {
				SFEdges_set.insert(pair<uint64_t, uint64_t>(cTri.x, cTri.y));
			}
			if (SFEdges_set.find(pair<uint64_t, uint64_t>(cTri.z, cTri.y)) == SFEdges_set.end()) {
				SFEdges_set.insert(pair<uint64_t, uint64_t>(cTri.y, cTri.z));
			}
			if (SFEdges_set.find(pair<uint64_t, uint64_t>(cTri.x, cTri.z)) == SFEdges_set.end()) {
				SFEdges_set.insert(pair<uint64_t, uint64_t>(cTri.z, cTri.x));
			}
		}
	}
	vector<pair<uint64_t, uint64_t>> tempEdge = vector<pair<uint64_t, uint64_t>>(SFEdges_set.begin(), SFEdges_set.end());
	for (int i = 0;i < tempEdge.size();i++) {
		surfEdges.push_back(make_uint2(tempEdge[i].first, tempEdge[i].second));
	}
}

void tetrahedra_obj::zsGetSurface() {
	using namespace zs;
	using vec2i = zs::vec<int, 2>;
	using vec3i = zs::vec<int, 3>;
	using vec4i = zs::vec<int, 4>;
	const auto numEles = tetrahedraNum;
      zs::HashTable<int, 3, int> surfTable{0, memsrc_e::host, -1};
      constexpr auto space = zs::execspace_e::openmp;
      auto ompExec = omp_exec();

      surfTable.resize(ompExec, 4 * numEles);
      surfTable.reset(ompExec, true);

      // compute getsurface
      std::vector<int> tri2tet(4 * numEles);
      ompExec(range(numEles), [table = proxy<space>(surfTable), &tri2tet,
                               &quads = this->tetrahedras](int ei) mutable {
        using table_t = RM_CVREF_T(table);
        using vec3i = zs::vec<int, 3>;
        auto record = [&table, &tri2tet, ei](const vec3i &triInds) mutable {
          if (auto sno = table.insert(triInds); sno != table_t::sentinel_v)
            tri2tet[sno] = ei;
          else
            printf("ridiculous, more than one tet share the same surface!");
        };
        auto inds = quads[ei];
        record(vec3i{inds.x, inds.z, inds.y});
        record(vec3i{inds.x, inds.w, inds.z});
        record(vec3i{inds.x, inds.y, inds.w});
        record(vec3i{inds.y, inds.z, inds.w});
      });
      //
      surface.resize(numEles * 4);
      int surfCnt = 0;
      ompExec(range(surfTable.size()),
              [table = proxy<space>(surfTable), &surfCnt, &tri2tet,
               &quads = this->tetrahedras, this](int i) mutable {
                using vec3i = zs::vec<int, 3>;
                auto triInds = table._activeKeys[i];
        	auto ei = tri2tet[i];
		auto inds = vec4i{quads[ei].x, quads[ei].y, quads[ei].z, quads[ei].w};
                using table_t = RM_CVREF_T(table);
                if (table.query(vec3i{triInds[2], triInds[1], triInds[0]}) ==
                        table_t::sentinel_v &&
                    table.query(vec3i{triInds[1], triInds[0], triInds[2]}) ==
                        table_t::sentinel_v &&
                    table.query(vec3i{triInds[0], triInds[2], triInds[1]}) ==
                        table_t::sentinel_v) {
		  auto no = atomic_add(exec_omp, &surfCnt, 1);
		  auto id3 = -1;
		  for (int d = 0; d != 4; ++d)
		  	if (inds[d] != triInds[0] && inds[d] != triInds[1] && inds[d] != triInds[2]) {
		  		id3 = d;
				break;
			}
			double3 vec1 = __GEIGEN__::__minus(vertexes[triInds[1]], vertexes[triInds[0]]);
			double3 vec2 = __GEIGEN__::__minus(vertexes[triInds[2]], vertexes[triInds[0]]);
			double3 vec3 = __GEIGEN__::__minus(vertexes[id3], vertexes[triInds[0]]);
			double3 n = __GEIGEN__::__v_vec_cross(vec1, vec2);
			if (__GEIGEN__::__v_vec_dot(n, vec3) > 0) {
				surface[no] = make_uint3(triInds[0], triInds[2], triInds[1]);
			}
			else {
				surface[no] = make_uint3(triInds[0], triInds[1], triInds[2]);
			}
		}
              });
      auto scnt = surfCnt;
      surface.resize(scnt);
      fmt::print("\t{} surfaces\n", scnt);

      // surface points
      HashTable<int, 1, int> vertTable{3 * (std::size_t)scnt , memsrc_e::host, -1};
      HashTable<int, 2, int> edgeTable{3 * (std::size_t)scnt, memsrc_e::host, -1};
      vertTable.reset(ompExec, true);
      edgeTable.reset(ompExec, true);
      ompExec(range(surface.size()),
              [vertTable = proxy<space>(vertTable),
               edgeTable = proxy<space>(edgeTable), this](int i) mutable {
		       auto tri = surface[i];
		vec3i triInds{tri.x, tri.y, tri.z};
                using vec1i = zs::vec<int, 1>;
                using vec2i = zs::vec<int, 2>;
                for (int d = 0; d != 3; ++d) {
                  vertTable.insert(vec1i{triInds[d]});
                  edgeTable.insert(vec2i{triInds[d], triInds[(d + 1) % 3]});
                }
              });
      auto svcnt = vertTable.size();
      surfVerts.resize(svcnt);
      ompExec(range(svcnt), [this, &vertTable](int vi) {
	      surfVerts[vi] = vertTable._activeKeys[vi][0];
	});
      fmt::print("\t{} surface verts\n", svcnt);

      // surface edges
      int surfEdgeCnt = 0;
      auto dupEdgeCnt = edgeTable.size();
      surfEdges.resize(dupEdgeCnt);
      ompExec(range(dupEdgeCnt), [edgeTable = proxy<space>(edgeTable), this,
                                  &surfEdgeCnt](int edgeNo) mutable {
        using vec2i = zs::vec<int, 2>;
        vec2i edge = edgeTable._activeKeys[edgeNo];
        using table_t = RM_CVREF_T(edgeTable);
        if (auto eno = edgeTable.query(vec2i{edge[1], edge[0]});
            eno == table_t::sentinel_v || // opposite edge not exists
            (eno != table_t::sentinel_v &&
             edge[0] < edge[1])) { // opposite edge does exist
          auto no = atomic_add(exec_omp, &surfEdgeCnt, 1);
          surfEdges[no] = make_uint2(edge[0], edge[1]);
        }
      });
      auto secnt = surfEdgeCnt;
      surfEdges.resize(secnt);
      fmt::print("\t{} surface edges\n", secnt);
}

bool tetrahedra_obj::output_tetrahedraMesh(const std::string& filename) {
	std::ofstream outmsh1(filename);
	outmsh1 << "$Nodes\n";
	outmsh1 << vertexNum << endl;
	for (int i = 0; i < vertexNum; i++) {
		outmsh1 << i + 1 << " " << vertexes[i].x << " " <<
			vertexes[i].y << " " <<
			vertexes[i].z << endl;
	}
	outmsh1 << "$Elements\n";
	outmsh1 << tetrahedraNum << endl;
	for (int i = 0; i < tetrahedraNum; i++) {
		outmsh1 << i + 1 << " 4 0 " << tetrahedras[i].x + 1 << " " <<
			tetrahedras[i].y + 1 << " " <<
			tetrahedras[i].z + 1 << " " <<
			tetrahedras[i].w + 1 << endl;
	}
	outmsh1.close();
	return true;
}

