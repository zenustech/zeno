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
