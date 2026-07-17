#pragma once

#include "optixCommon.h"

#include "Octree.h"
#include <sutil/Aabb.h>
// #include <sutil/vec_math.h>
#include <sutil/Exception.h>

#include <nanovdb/NanoVDB.h>
#include <nanovdb/util/GridHandle.h>
#include <nanovdb/util/GridBuilder.h>
#include <nanovdb/util/OpenToNanoVDB.h>

#include <glm/glm.hpp>
#include <glm/ext.hpp>

#include <rapidjson/document.h>
#include <rapidjson/rapidjson.h>

#include <cmath>
#include <algorithm>
#include <vector>
#include <string>
#include <memory>
#include <cstdint>
#include <chrono>

#include <functional>

#include <iostream>
#include <filesystem>

#include "volume.h"
#include "magic_enum.hpp"
#include "VDBDensityBake.h"

#include <zeno/utils/vec.h>
#include <zeno/utils/type_traits.h>
#include <zeno/types/TextureObject.h>

struct GridWrapper {
	nanovdb::GridHandle<> handle{};
	xinxinoptix::raii<CUdeviceptr> buffer;

	GridWrapper(){}
	GridWrapper(GridWrapper&&) = default;

	virtual ~GridWrapper() {}

	virtual void parsing(openvdb::GridBase::Ptr& grid_ptr) = 0;

	virtual const nanovdb::Map& nanoMAP() = 0;

	virtual const nanovdb::BBox<nanovdb::Coord>& indexedBox() = 0;

	virtual void analysis(const std::string& path) = 0;

    virtual bool isFloatGrid() const = 0;
};

namespace openvdb {
	using Vec4dTree = tree::Tree4<Vec4d, 5, 4, 3>::Type;
	using Vec4dfGrid = Grid<Vec4dTree>;

	using Vec4fTree = tree::Tree4<Vec4f, 5, 4, 3>::Type;
	using Vec4fGrid = Grid<Vec4fTree>;

	using Vec4iTree = tree::Tree4<Vec4i, 5, 4, 3>::Type;
	using Vec4iGrid = Grid<Vec4iTree>;

	using UInt64Tree = tree::Tree4<uint64_t, 5, 4, 3>::Type;
	using UInt64Grid = Grid<UInt64Tree>;
	using UInt32Grid = Grid<UInt32Tree>;

	using Int16Tree = tree::Tree4<int16_t, 5, 4, 3>::Type;
	using Int16Grid = Grid<Int16Tree>;

	using Int8Tree = tree::Tree4<int8_t, 5, 4, 3>::Type;
	using Int8Grid = Grid<Int8Tree>;

	using Rgba8 = math::Vec4<uint8_t>;
	using Rgba8Tree = tree::Tree4<Rgba8, 5, 4, 3>::Type;
	using Rgba8Grid = Grid<Rgba8Tree>;
};

template<typename oGridType, typename nValueType>
struct GridPair {
	using oGridT = oGridType;
    using nValueT = nValueType;
};

using SupportedGrids = std::tuple<
	GridPair<openvdb::Vec3dGrid,  nanovdb::Vec4d>,
	GridPair<openvdb::Vec3dGrid,  nanovdb::Vec3d>,
    GridPair<openvdb::DoubleGrid, double>,

	GridPair<openvdb::Vec4fGrid,  nanovdb::Vec4f>,
	GridPair<openvdb::Vec3fGrid,  nanovdb::Vec3f>,
	GridPair<openvdb::FloatGrid,  float>,
	
	GridPair<openvdb::FloatGrid,  nanovdb::Fp16>,
	GridPair<openvdb::FloatGrid,  nanovdb::Fp8>,
    GridPair<openvdb::FloatGrid,  nanovdb::Fp4>,

	GridPair<openvdb::Int64Grid,  int64_t>,
	GridPair<openvdb::UInt32Grid, uint32_t>,

	GridPair<openvdb::Vec4iGrid,  nanovdb::Vec4i>,
	GridPair<openvdb::Vec3IGrid,  nanovdb::Vec3i>,
    GridPair<openvdb::Int32Grid,  int32_t>,
	GridPair<openvdb::Int16Grid,  int16_t>
>;

using nvdb_type_list = std::tuple<nanovdb::Float4, nanovdb::Float3, float, nanovdb::Fp16, nanovdb::Fp8, nanovdb::Fp4, nanovdb::Long, uint32_t, nanovdb::Int4, nanovdb::Int3, int, nanovdb::Short>;

template <typename T>
inline std::ostream& operator<<(std::ostream& os, const nanovdb::Vec3<T>& v) {
    os << "Vec3{" << v[0] << "," << v[1] << ","<< v[2] << "}";
    return os;
}
template <typename T>
inline std::ostream& operator<<(std::ostream& os, const nanovdb::Vec4<T>& v) {
    os << "Vec4{" << v[0] << "," << v[1] << ","<< v[2] << "," << v[3] <<"}";
    return os;
}

template<typename T>
struct TypedGridWrapper: GridWrapper {

	template<typename Pair>
	bool tryConvert(openvdb::GridBase::Ptr& grid)
	{
		using OpenVDBGridT = typename Pair::oGridT;
		using NanoValueT  = typename Pair::nValueT;

		bool match = true;
		if (!grid->isType<OpenVDBGridT>()) {
			std::cerr << "Error: " << "NanoVDB data type doesn't match openVDB data type" << std::endl;
			std::cerr << "Error: " << "Using fallback empty grid" << std::endl;
			
			grid = OpenVDBGridT::create( typename OpenVDBGridT::ValueType() );
			grid->setName("dummy");
			grid->setTransform(openvdb::math::Transform::createLinearTransform(1.0));
			match = false;
		}

		auto src = openvdb::gridPtrCast<OpenVDBGridT>(grid);
		std::cout << "VDB convert begin {" << grid->getName() << "}" << std::endl;
		handle = nanovdb::openToNanoVDB<nanovdb::HostBuffer, OpenVDBGridT::TreeType, T>(*src, nanovdb::StatsMode::MinMax);
		return match;
	}

	template<std::size_t I = 0>
	bool dispatch(openvdb::GridBase::Ptr& grid)
	{
		if constexpr (I == std::tuple_size_v<SupportedGrids>) {
			return false; // end of loop
		} else {
			using Pair = std::tuple_element_t<I, SupportedGrids>;
			
			using OpenVDBGridT = typename Pair::oGridT;
			using NanoValueT  = typename Pair::nValueT;
			
			if constexpr (std::is_same_v<NanoValueT, T>) {
				return tryConvert<Pair>(grid);
			}
			return dispatch<I + 1>(grid); // next iteration
		}
	}

	void parsing(openvdb::GridBase::Ptr& grid_ptr) override {

		if (!dispatch(grid_ptr)) {
			std::cerr << "Error: " << "NanoVDB convert failed" << std::endl;
		}
		return;
	}

	const nanovdb::Map& nanoMAP() override {
		auto* grid_handle = handle.grid<T>();
		return grid_handle->map();
	}

	const nanovdb::BBox<nanovdb::Coord>& indexedBox() override {
		auto* grid_handle = handle.grid<T>();
		return grid_handle->indexBBox();
	}

	void analysis(const std::string& path) override {
		auto* grid_handle = handle.grid<T>();
		auto mini = grid_handle->tree().root().minimum();
		auto maxi = grid_handle->tree().root().maximum();

		auto average   = grid_handle->tree().root().average();
		auto variance  = grid_handle->tree().root().variance();
		auto deviation = grid_handle->tree().root().stdDeviation();

		auto gridname = grid_handle->gridName();    
		std::cout << "max value is " << maxi << " in " << path << " {" << gridname << "}" << std::endl;

		auto ibb = grid_handle->indexBBox();
		std::cout << "gird indexed box min: {" << ibb.min().x() << ", " << ibb.min().y() << ", "<< ibb.min().z() << "}" << std::endl;
		std::cout << "gird indexed box max: {" << ibb.max().x() << ", " << ibb.max().y() << ", "<< ibb.max().z() << "}" << std::endl;
	}

    bool isFloatGrid() const override {
        return std::is_same_v<T, float>;
    }

	TypedGridWrapper(): GridWrapper() {}
	~TypedGridWrapper() {}
};

inline static void makeTypedGridWrapper(zeno::TextureObjectVDB::ElementType et, std::vector<std::shared_ptr<GridWrapper>>& gwl) {

	const auto enum_idx = magic_enum::enum_integer(et);

	auto matched = zeno::static_for<0, std::tuple_size_v<nvdb_type_list>>([&] (auto i) {

		if (i == enum_idx) {
			using GridT = std::tuple_element_t<i, nvdb_type_list>;
			gwl.push_back( std::make_shared<TypedGridWrapper<GridT>>() );
			return true;
		}
		return false;
    });
}

using VolumeAccel = SceneNode;

struct BakedSparseVolume
{
    BakedSparseVolumeDevice device {};
    xinxinoptix::raii<CUdeviceptr> d_descriptor;
    xinxinoptix::raii<CUdeviceptr> d_brick_table;
    xinxinoptix::raii<CUdeviceptr> d_brick_origins;
    xinxinoptix::raii<CUdeviceptr> d_voxel_values;
    xinxinoptix::raii<CUdeviceptr> d_brick_min;
    xinxinoptix::raii<CUdeviceptr> d_brick_max;
    xinxinoptix::raii<CUdeviceptr> d_octree;

    bool valid() const {
        return d_descriptor.handle != 0 && device.brick_count != 0 && device.octree != nullptr && device.octree_node_count != 0;
    }

    void reset() {
        device = {};
        d_descriptor.reset();
        d_brick_table.reset();
        d_brick_origins.reset();
        d_voxel_values.reset();
        d_brick_min.reset();
        d_brick_max.reset();
        d_octree.reset();
    }
};

struct VolumeWrapper
{
	bool dirty = true;
	//openvdb::math::Transform::Ptr transform; // openvdb::math::Mat4f::identity();
	uint8_t bounds;
	glm::mat4 transform;
	sutil::Aabb aabb;
	std::shared_ptr<xinxinoptix::raii<CUdeviceptr>> d_aabb = std::make_shared<xinxinoptix::raii<CUdeviceptr>>();

	VolumeAggregate aggregate;
	std::shared_ptr<xinxinoptix::raii<CUdeviceptr>> d_octree = std::make_shared<xinxinoptix::raii<CUdeviceptr>>();
	std::shared_ptr<BakedSparseVolume> baked_density = std::make_shared<BakedSparseVolume>();
	// Single per-volume control for CPU/GPU octree build depth. Valid SVO range is clamped.
	uint32_t octreeBuildDepth = BAKED_SPARSE_VOLUME_DEFAULT_OCTREE_DEPTH;
	bool use_gpu_baked_octree = true;
	bool validate_gpu_baked_octree = false;
	bool use_custom_density_sample_bbox = false;
	int3 custom_density_sample_min {};
	// Exclusive max in VDB index space.
	int3 custom_density_sample_max {};

	std::vector<std::string> selected;
	uint density_grid_index = 0;

	std::filesystem::file_time_type file_time;
	std::string density_bake_key;

	std::vector<std::shared_ptr<GridWrapper>> grids;
	std::vector<std::function<void()>> tasks;

	zeno::TextureObjectVDB::ElementType type;
	std::shared_ptr<VolumeAccel> node = std::make_shared<VolumeAccel>();

	~VolumeWrapper() = default;
};

bool loadVolume( VolumeWrapper& volume, const std::string& path );
void loadVolumeVDB( VolumeWrapper& volume, const std::string& path);
void loadVolumeNVDB( VolumeWrapper& volume, const std::string& path);

void checkGridName( const std::string& path, std::string& name);
std::string fetchGridName( const std::string& path, uint index );

void loadGrid( GridWrapper& grid, const std::string& path, const uint index );
void loadGrid( GridWrapper& grid, const std::string& path, const std::string& gridname );

void unloadGrid(GridWrapper& grid);
void cleanupVolume( VolumeWrapper& volume );
void releaseVolumeDeviceData(VolumeWrapper& volume);

bool bakeDensityGridToSparseBricksOnGPU(
    GridWrapper& grid,
    BakedSparseVolume& baked_volume,
    const xinxinoptix::VDBDensityBakeInputs& inputs,
    const xinxinoptix::VDBDensityBakeOptions& options,
    xinxinoptix::VDBDensityBakeResult* result = nullptr);

void getOptixTransform( const VolumeWrapper& volume, float transform[] );
sutil::Aabb worldAabb( const VolumeWrapper& volume );

// The VolumeAccel struct contains a volume's geometric representation for
// Optix: a traversalbe handle, and the (compacted) GAS device-buffer.


void buildVolumeAccel( VolumeWrapper& volume, const OptixDeviceContext& context );
void cleanupVolumeAccel( VolumeAccel& accel );