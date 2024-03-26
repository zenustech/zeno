#pragma once

#include <optix.h>
#include <optix_stubs.h>

#include <sutil/Aabb.h>
#include <sutil/vec_math.h>
#include <sutil/Exception.h>

#include <nanovdb/NanoVDB.h>
#include <nanovdb/util/GridHandle.h>
#include <nanovdb/util/OpenToNanoVDB.h>

#include <glm/glm.hpp>
#include <glm/ext.hpp>

#include <rapidjson/document.h>
#include <rapidjson/rapidjson.h>

#include <cmath>
#include <vector>
#include <string>
#include <memory>

#include <functional>

#include <iostream>
#include <filesystem>

#include "volume.h"
#include "magic_enum.hpp"

#include <zeno/utils/type_traits.h>
#include <zeno/types/TextureObject.h>

#ifndef uint
typedef unsigned int uint;
#endif

struct GridWrapper {
	nanovdb::GridHandle<> handle{};
	CUdeviceptr deviceptr = 0;
	float max_value = 1.0;

	GridWrapper(){}
	GridWrapper(GridWrapper&&) = default;

	virtual ~GridWrapper() {}
	// {
	// 	handle.reset();
	// 	if (0 != deviceptr) {
    //     	CUDA_CHECK_NOTHROW( cudaFree( reinterpret_cast<void*>( deviceptr ) ) );
    //     	deviceptr = 0;
    // 	}
	// }

	virtual void parsing(openvdb::GridBase::Ptr& grid_ptr) = 0;

	virtual const nanovdb::Map& nanoMAP() = 0;

	virtual const nanovdb::BBox<nanovdb::Coord>& indexedBox() = 0;

	virtual float analysis(const std::string& path) = 0;
};

template<typename T>
struct TypedGridWrapper: GridWrapper {

	void parsing(openvdb::GridBase::Ptr& grid_ptr) override {

		openvdb::FloatGrid::Ptr srcGrid = openvdb::gridPtrCast<openvdb::FloatGrid>(grid_ptr);
        handle = nanovdb::openToNanoVDB<nanovdb::HostBuffer, openvdb::FloatGrid::TreeType, T>(*srcGrid, nanovdb::StatsMode::All); 
		std::cout << "Grid byte size is "<< handle.size() << std::endl;
	}

	const nanovdb::Map& nanoMAP() override {
		auto* grid_handle = handle.grid<T>();
		return grid_handle->map();
	}

	const nanovdb::BBox<nanovdb::Coord>& indexedBox() override {
		auto* grid_handle = handle.grid<T>();
		return grid_handle->indexBBox();
	}

	float analysis(const std::string& path) override {
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

		return maxi;
	}

	TypedGridWrapper(): GridWrapper() {}
	~TypedGridWrapper() {}
};

using nvdb_type_list = std::tuple<nanovdb::Fp32, nanovdb::Fp16, nanovdb::Fp8, nanovdb::Fp4>;

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

struct VolumeAccel
{
	OptixTraversableHandle handle = 0;
	CUdeviceptr            d_buffer = 0;

	VolumeAccel() = default;
	VolumeAccel(VolumeAccel&&) = default;

	~VolumeAccel() {
		if (0 != d_buffer) {
        	CUDA_CHECK_NOTHROW( cudaFree( (void*)( d_buffer ) ) );
        	d_buffer = 0; handle = 0;
    	}
	}
};

struct VolumeWrapper
{
	//openvdb::math::Transform::Ptr transform; // openvdb::math::Mat4f::identity();
	glm::mat4 transform; 
	std::vector<std::string> selected;

	std::filesystem::file_time_type file_time;

	std::vector<std::shared_ptr<GridWrapper>> grids;
	std::vector<std::function<void()>> tasks;

	zeno::TextureObjectVDB::ElementType type;

	VolumeAccel accel {};

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

void getOptixTransform( const VolumeWrapper& volume, float transform[] );
sutil::Aabb worldAabb( const VolumeWrapper& volume );

// The VolumeAccel struct contains a volume's geometric representation for
// Optix: a traversalbe handle, and the (compacted) GAS device-buffer.


void buildVolumeAccel( VolumeAccel& accel, const VolumeWrapper& volume, const OptixDeviceContext& context );
void cleanupVolumeAccel( VolumeAccel& accel );

static inline void prepareVolumeTransform(std::string &raw, glm::mat4& linear_transform) {
	rapidjson::Document d;
	d.Parse(raw.c_str());
	if ( d.IsNull() ) {return;}

	glm::f64vec3 scale_vector(1), translate_vector(0);
	glm::f64vec4 rotate_vector(1, 1, 1, 0);
	
	auto parsing = [&d](std::string key, auto &result) {

		if (!d.HasMember(key.c_str())) return;
		
		auto& _ele = d[key.c_str()];
		if (_ele.IsArray()) {
			auto _array = _ele.GetArray();

			for (uint i=0; i<result.length(); ++i) {
				if (i<_array.Size() && _array[i].IsNumber()) {
					result[i] = _array[i].GetFloat();
				} // if
			} // for
		}
	};

	parsing("scale", scale_vector);
	parsing("rotate", rotate_vector);
	parsing("translate", translate_vector);

	glm::f64mat4 _transform(1.0); 

	if (translate_vector != glm::f64vec3(0)) {
		_transform = glm::translate(_transform, translate_vector);
	}

	glm::f64vec3 rotate_axis = glm::f64vec3(rotate_vector);
	if (rotate_vector.w != 0.0 && rotate_axis != glm::f64vec3(1, 1, 1)) {
		_transform = glm::rotate(_transform, glm::radians(rotate_vector.w), rotate_axis);
	}

	if (scale_vector != glm::f64vec3(1)) {
		_transform = glm::scale(_transform, scale_vector);
	}

	linear_transform = _transform;
}