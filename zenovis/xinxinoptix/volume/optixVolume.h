#pragma once

#include <optix.h>

#include <sutil/Aabb.h>
#include <sutil/vec_math.h>

#include <cuda/Light.h>
#include <cuda/BufferView.h>

#include <nanovdb/NanoVDB.h>
#include <nanovdb/util/GridHandle.h>
#include <nanovdb/util/OpenToNanoVDB.h>

#include <glm/glm.hpp>
#include <glm/ext.hpp>

#include <rapidjson/document.h>
#include <rapidjson/rapidjson.h>

#include <cmath>
#include <string>
#include <filesystem>

#ifndef uint
typedef unsigned int uint;
#endif

struct GridWrapper {
	nanovdb::GridHandle<> handle;
	CUdeviceptr deviceptr = 0;
	float max_value = 1.0;
};

struct VolumeWrapper
{
	//openvdb::math::Transform::Ptr transform; // openvdb::math::Mat4f::identity();
	glm::f64mat4 transform; 
	std::vector<std::string> selected;

	std::filesystem::file_time_type file_time;

	std::vector<GridWrapper> grids;
	std::vector<std::function<void()>> tasks;
};

bool loadVolume( VolumeWrapper& volume, const std::string& path );
void loadVolumeVDB( VolumeWrapper& volume, const std::string& path);
void loadVolumeNVDB( VolumeWrapper& volume, const std::string& path);

void fetchGridName( const std::string& path, std::string& name);
std::string fetchGridName( const std::string& path, uint index );

void loadGrid( GridWrapper& grid, const std::string& path, const uint index );
void loadGrid( GridWrapper& grid, const std::string& path, const std::string& gridname );

void unloadGrid(GridWrapper& grid);
void cleanupVolume( VolumeWrapper& volume );

void getOptixTransform( const VolumeWrapper& volume, float transform[] );
sutil::Aabb worldAabb( const VolumeWrapper& volume );

// The VolumeAccel struct contains a volume's geometric representation for
// Optix: a traversalbe handle, and the (compacted) GAS device-buffer.
struct VolumeAccel
{
	OptixTraversableHandle handle = 0;
	CUdeviceptr            d_buffer = 0;
};

void buildVolumeAccel( VolumeAccel& accel, const VolumeWrapper& volume, const OptixDeviceContext& context );
void cleanupVolumeAccel( VolumeAccel& accel );

static inline void prepareVolumeTransform(std::string &raw, glm::f64mat4& linear_transform) {
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

	if (translate_vector != glm::f64vec3(0)) {
		linear_transform = glm::translate(linear_transform, translate_vector);
	}

	glm::f64vec3 rotate_axis = glm::f64vec3(rotate_vector);
	if (rotate_vector.w != 0.0 && rotate_axis != glm::f64vec3(1, 1, 1)) {
		linear_transform = glm::rotate(linear_transform, glm::radians(rotate_vector.w), rotate_axis);
	}

	if (scale_vector != glm::f64vec3(1)) {
		linear_transform = glm::scale(linear_transform, scale_vector);
	}
}