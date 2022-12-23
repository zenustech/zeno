#pragma once

#include <optix.h>

#include <sutil/Aabb.h>
#include <sutil/vec_math.h>

#include <cuda/Light.h>
#include <cuda/BufferView.h>

#include <nanovdb/NanoVDB.h>
#include <nanovdb/util/GridHandle.h>

#include <string>
#include <cmath>

// The Volume struct ties together the NanoVDB host representation
// (NanoVDB Grid) and the device-buffer containing the sparse volume
// representation (NanoVDB Tree). In addition to the Tree, the Grid
// also contains an affine transform relating index space (i.e. voxel
// indices) to world-space.
//

struct GridWrapper {
	nanovdb::GridHandle<> handle;
	CUdeviceptr deviceptr = 0;
	float max_value = 1.0;
};

struct Volume
{
	GridWrapper grid_density;
    // nanovdb::GridHandle<> handle_density;
	// CUdeviceptr d_density = 0;

	GridWrapper grid_temp;
	// nanovdb::GridHandle<> handle_temp;
	// CUdeviceptr d_temp = 0;
};

void loadVolume( Volume& volume, const std::string& filename );
void cleanupVolume( Volume& volume );
void createGrid( GridWrapper& grid, std::string filename, std::string gridname );
void getOptixTransform( const Volume& volume, float transform[] );
sutil::Aabb worldAabb( const Volume& volume );

// The VolumeAccel struct contains a volume's geometric representation for
// Optix: a traversalbe handle, and the (compacted) GAS device-buffer.
struct VolumeAccel
{
	OptixTraversableHandle handle = 0;
	CUdeviceptr            d_buffer = 0;
};

void buildVolumeAccel( VolumeAccel& accel, const Volume& volume, const OptixDeviceContext& context );
void cleanupVolumeAccel( VolumeAccel& accel );
