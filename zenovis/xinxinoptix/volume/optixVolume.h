#pragma once

#include <optix.h>

#include <sutil/Aabb.h>
#include <sutil/vec_math.h>

#include <cuda/Light.h>
#include <cuda/BufferView.h>

#include <nanovdb/NanoVDB.h>
#include <nanovdb/util/GridHandle.h>
#include <nanovdb/util/OpenToNanoVDB.h>

#include <string>
#include <cmath>

struct GridWrapper {
	nanovdb::GridHandle<> handle;
	CUdeviceptr deviceptr = 0;
	float max_value = 1.0;
};

struct VolumeWrapper
{
	GridWrapper grid_density;
    // nanovdb::GridHandle<> handle_density;
	// CUdeviceptr d_density = 0;

	GridWrapper grid_temp;
	// nanovdb::GridHandle<> handle_temp;
	// CUdeviceptr d_temp = 0;

	openvdb::math::Transform::Ptr transform; // openvdb::math::Mat4f::identity();  
};

bool loadVolume( VolumeWrapper& volume, const std::string& filename );
void loadVDB( VolumeWrapper& volume, const std::string& path);
void loadNVDB( VolumeWrapper& volume, const std::string& path);

void cleanupVolume( VolumeWrapper& volume );
void createGrid( GridWrapper& grid, const std::string& path, const std::string& gridname );
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
