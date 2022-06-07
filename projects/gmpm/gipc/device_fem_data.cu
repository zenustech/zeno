#include "device_fem_data.cuh"
#include "zensim/cuda/execution/ExecutionPolicy.cuh"
#include "cuda_tools.h"


void device_TetraData::Malloc_DEVICE_MEM(const int& vertex_num, const int& tetradedra_num){
	int maxNumbers = vertex_num > tetradedra_num ? vertex_num : tetradedra_num;
	CUDA_SAFE_CALL(cudaMalloc((void**)&vertexes, vertex_num*sizeof(double3)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&o_vertexes, vertex_num * sizeof(double3)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&velocities, vertex_num * sizeof(double3)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&rest_vertexes, vertex_num * sizeof(double3)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&temp_double3Mem, vertex_num * sizeof(double3)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&xTilta, vertex_num * sizeof(double3)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&fb, vertex_num * sizeof(double3)));
	
	CUDA_SAFE_CALL(cudaMalloc((void**)&tetrahedras, tetradedra_num * sizeof(uint4)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&tempTetrahedras, tetradedra_num * sizeof(uint4)));

	CUDA_SAFE_CALL(cudaMalloc((void**)&volum, tetradedra_num * sizeof(double)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&masses, vertex_num * sizeof(double)));

	CUDA_SAFE_CALL(cudaMalloc((void**)&tempDouble, maxNumbers * sizeof(double)));
	//CUDA_SAFE_CALL(cudaMalloc((void**)&tempM, vertex_num * sizeof(double)));

	CUDA_SAFE_CALL(cudaMalloc((void**)&MChash, maxNumbers * sizeof(uint64_t)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&sortIndex, maxNumbers * sizeof(uint32_t)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&BoundaryType, vertex_num * sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&tempBoundaryType, vertex_num * sizeof(int)));

	CUDA_SAFE_CALL(cudaMemset(BoundaryType, 0, vertex_num * sizeof(int)));

	//CUDA_SAFE_CALL(cudaMalloc((void**)&sortVertIndex, vertex_num * sizeof(uint32_t)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&sortMapVertIndex, vertex_num * sizeof(uint32_t)));

	CUDA_SAFE_CALL(cudaMalloc((void**)&DmInverses, tetradedra_num * sizeof(__GEIGEN__::Matrix3x3d)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&Constraints, vertex_num * sizeof(__GEIGEN__::Matrix3x3d)));

	CUDA_SAFE_CALL(cudaMalloc((void**)&tempMat3x3, maxNumbers * sizeof(__GEIGEN__::Matrix3x3d)));
	//CUDA_SAFE_CALL(cudaMalloc((void**)&tempConstraints, vertex_num * sizeof(__GEIGEN__::Matrix3x3d)));
	using namespace zs;
	verts = tiles_t{{{"m", 1},
                     {"x", 3},
                     {"x0", 3},
                     {"v", 3},
                     {"temp_double3Mem", 3},
                     {"xtilde", 3}},
                    (std::size_t)vertex_num,
                    memsrc_e::device,
                    0};
	eles = tiles_t{{{"vol", 1}, {"IB", 9}, {"inds", 4}},
                   (std::size_t)tetradedra_num,
                   memsrc_e::device,
                   0};
	vtemp = dtiles_t{{{"grad", 3},
                           {"gc", 3},
                           {"gE", 3},
                           {"P", 9},
                           {"dir", 3},
                           {"xn", 3},
                           {"xn0", 3},
                           {"xtilde", 3},
                           {"temp", 3},
                           {"r", 3},
                           {"p", 3},
                           {"q", 3},
			   {"H", 9}},
                     (std::size_t)vertex_num,
                     memsrc_e::device,
                     0};
	etemp = dtiles_t{{{"temp", 3}}, (std::size_t)tetradedra_num, memsrc_e::device, 0};
}


void device_TetraData::init(const tetrahedra_obj& tetMesh) {
	using namespace zs;
	numVerts = tetMesh.vertexNum;
	numEles = tetMesh.tetrahedraNum;
	constexpr auto space = execspace_e::cuda;
	using vec3 = zs::vec<float, 3>;
	using vec4 = zs::vec<float, 4>;
	using vec4i = zs::vec<int, 4>;
	using mat3 = zs::vec<float, 3, 3>;
	auto cudaPol = cuda_exec();
	auto dmass = from_std_vector(tetMesh.masses, MemoryLocation{memsrc_e::device, 0});
	auto dvertexes = from_std_vector(tetMesh.vertexes, MemoryLocation{memsrc_e::device, 0});
	auto dvels = from_std_vector(tetMesh.velocities, MemoryLocation{memsrc_e::device, 0});
	auto dcons = from_std_vector(tetMesh.constraints, MemoryLocation{memsrc_e::device, 0});
	cudaPol(range(tetMesh.vertexNum), [this, 
		verts = proxy<space>({}, verts), vtemp = proxy<space>({}, vtemp), 
		dmass = proxy<space>(dmass),
		dcons = proxy<space>(dcons),
		dvels = proxy<space>(dvels),
		dvertexes = proxy<space>(dvertexes)
		]__device__(int vi) mutable {
		verts("m", vi) = dmass[vi];
		auto x = vec3{dvertexes[vi].x, dvertexes[vi].y, dvertexes[vi].z};
		auto v = vec3{dvels[vi].x, dvels[vi].y, dvels[vi].z};
		verts.template tuple<3>("x", vi) = x;
		verts.template tuple<3>("x0", vi) = x;
		verts.template tuple<3>("v", vi) = v;
		auto con = dcons[vi];
		vtemp.template tuple<9>("H", vi) = mat3{con.m[0][0], con.m[0][1], con.m[0][2], 
			con.m[1][0], con.m[1][1], con.m[1][2], con.m[2][0], con.m[2][1], con.m[2][2]};
	});
	auto dvols = from_std_vector(tetMesh.volum, MemoryLocation{memsrc_e::device, 0});
	auto dquads = from_std_vector(tetMesh.tetrahedras, MemoryLocation{memsrc_e::device, 0});;
	auto dIBs = from_std_vector(tetMesh.DM_inverse, MemoryLocation{memsrc_e::device, 0});;
	cudaPol(range(tetMesh.tetrahedraNum), [this, 
		eles = proxy<space>({}, eles), 
		dquads = proxy<space>(dquads), 
		dIBs = proxy<space>(dIBs), 
		dvols = proxy<space>(dvols)
		]__device__(int ei) mutable {
		eles("vol", ei) = dvols[ei];
		auto quad = vec4i{dquads[ei].x, dquads[ei].y, dquads[ei].z, dquads[ei].w};
		eles.template tuple<4>("inds", ei) = quad.template reinterpret_bits<float>();
		auto ib = dIBs[ei];
		eles.template tuple<9>("IB", ei) = mat3{ib.m[0][0], ib.m[0][1], ib.m[0][2], 
			ib.m[1][0], ib.m[1][1], ib.m[1][2], ib.m[2][0], ib.m[2][1], ib.m[2][2]};
	});
}

void device_TetraData::retrieve() {
	using namespace zs;
	constexpr auto space = execspace_e::cuda;
	using vec3 = zs::vec<float, 3>;
	using vec3d = zs::vec<double, 3>;
	using vec4 = zs::vec<float, 4>;
	using vec4i = zs::vec<int, 4>;
	using mat3 = zs::vec<float, 3, 3>;
	auto cudaPol = cuda_exec();
	cudaPol(range(numVerts), [masses = this->masses,
		vertexes = this->vertexes,
		rest_vertexes = this->rest_vertexes,
		velocities = this->velocities,
		Constraints = this->Constraints,
		verts = proxy<space>({}, verts), vtemp = proxy<space>({}, vtemp) 
		]__device__(int vi) mutable {
		verts("m", vi) = masses[vi];
		auto x = vec3d{vertexes[vi].x, vertexes[vi].y, vertexes[vi].z};
		auto x0 = vec3d{rest_vertexes[vi].x, rest_vertexes[vi].y, rest_vertexes[vi].z};
		auto v = vec3{velocities[vi].x, velocities[vi].y, velocities[vi].z};
		verts.template tuple<3>("x", vi) = x;
		verts.template tuple<3>("x0", vi) = x0;
		verts.template tuple<3>("v", vi) = v;
		// vtemp
		auto con = Constraints[vi];
		vtemp.template tuple<9>("H", vi) = mat3{con.m[0][0], con.m[0][1], con.m[0][2], 
			con.m[1][0], con.m[1][1], con.m[1][2], con.m[2][0], con.m[2][1], con.m[2][2]};

		vtemp.template tuple<3>("xn", vi) = x;
	});
	cudaPol(range(numEles), [
		volum = this->volum,
		tetrahedras = this->tetrahedras,
		DmInverses = this->DmInverses,
		eles = proxy<space>({}, eles)
		]__device__(int ei) mutable {
		eles("vol", ei) = volum[ei];
		auto quad = vec4i{tetrahedras[ei].x, tetrahedras[ei].y, 
			tetrahedras[ei].z, tetrahedras[ei].w};
		eles.template tuple<4>("inds", ei) = quad.template reinterpret_bits<float>();
		auto ib = DmInverses[ei];
		eles.template tuple<9>("IB", ei) = mat3{ib.m[0][0], ib.m[0][1], ib.m[0][2], 
			ib.m[1][0], ib.m[1][1], ib.m[1][2], ib.m[2][0], ib.m[2][1], ib.m[2][2]};
	});
}

device_TetraData::~device_TetraData() {
	// FREE_DEVICE_MEM();
}

void device_TetraData::FREE_DEVICE_MEM() {
	CUDA_SAFE_CALL(cudaFree(sortIndex));
	CUDA_SAFE_CALL(cudaFree(sortMapVertIndex));
	CUDA_SAFE_CALL(cudaFree(vertexes)); 
	CUDA_SAFE_CALL(cudaFree(o_vertexes));
	CUDA_SAFE_CALL(cudaFree(temp_double3Mem));
	CUDA_SAFE_CALL(cudaFree(velocities));
	CUDA_SAFE_CALL(cudaFree(rest_vertexes));
	CUDA_SAFE_CALL(cudaFree(xTilta));
	CUDA_SAFE_CALL(cudaFree(fb));
	CUDA_SAFE_CALL(cudaFree(tetrahedras));
	CUDA_SAFE_CALL(cudaFree(tempTetrahedras));
	CUDA_SAFE_CALL(cudaFree(volum));
	CUDA_SAFE_CALL(cudaFree(masses));
	CUDA_SAFE_CALL(cudaFree(DmInverses));
	CUDA_SAFE_CALL(cudaFree(Constraints));
	CUDA_SAFE_CALL(cudaFree(tempMat3x3));
	CUDA_SAFE_CALL(cudaFree(MChash));
	CUDA_SAFE_CALL(cudaFree(tempDouble));
	CUDA_SAFE_CALL(cudaFree(BoundaryType));
	CUDA_SAFE_CALL(cudaFree(tempBoundaryType));
}


