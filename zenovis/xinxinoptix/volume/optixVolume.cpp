#include "optixVolume.h"

#include <sutil/Exception.h>
#include <nanovdb/util/IO.h>

#include <optix_stubs.h>
#include <optix_function_table_definition.h>

// ----------------------------------------------------------------------------
// Functions for manipulating Volume instances
// ----------------------------------------------------------------------------

void loadVolume( Volume& grid, const std::string& filename )
{
    // NanoVDB files are containers for NanoVDB Grids.
    // Each Grid represents a distinct volume, point-cloud, or level-set.
    // For the purpose of this sample only the first grid is loaded, additional
    // grids are ignored.
    auto list = nanovdb::io::readGridMetaData( filename );
    std::cerr << "Opened file " << filename << std::endl;
    std::cerr << "    grids:" << std::endl;
    for (auto& m : list) {
        std::cerr << "        " << m.gridName << std::endl;
    }
    assert( list.size() > 0 );
    // load the first grid in the file 
    createGrid( grid, filename, list[0].gridName );
}

 void createGrid( Volume& grid, std::string filename, std::string gridname )
{
    nanovdb::GridHandle<> gridHdl;

	if( gridname.length() > 0 )
		gridHdl = nanovdb::io::readGrid<>( filename, gridname );
	else
		gridHdl = nanovdb::io::readGrid<>( filename );

    if( !gridHdl ) 
    {
        std::stringstream ss;
        ss << "Unable to read " << gridname << " from " << filename;
        throw std::runtime_error( ss.str() );
    }

    // NanoVDB Grids can represent several kinds of 3D data, but this sample is
    // only concerned with volumetric data.
    auto* meta = gridHdl.gridMetaData();
    if( meta->isPointData() )
        throw std::runtime_error("NanoVDB Point Data cannot be handled by VolumeViewer");
    if( meta->isLevelSet() )
        throw std::runtime_error("NanoVDB Level Sets cannot be handled by VolumeViewer");

    // NanoVDB files represent the sparse data-structure as flat arrays that can be
    // uploaded to the device "as-is".
    assert( gridHdl.size() != 0 );
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &grid.d_volume ), gridHdl.size() ) );
    CUDA_CHECK( cudaMemcpy( reinterpret_cast<void*>( grid.d_volume ), gridHdl.data(), gridHdl.size(),
        cudaMemcpyHostToDevice ) );

    grid.handle = std::move( gridHdl );
}


void cleanupVolume( Volume& volume )
{
    // OptiX cleanup
	CUDA_CHECK_NOTHROW( cudaFree( reinterpret_cast<void*>( volume.d_volume ) ) );
}

void buildVolumeAccel( VolumeAccel& accel, const Volume& volume, const OptixDeviceContext& context )
{
    // Build accel for the volume and store it in a VolumeAccel struct.
    //
    // For Optix the NanoVDB volume is represented as a 3D box in index coordinate space. The volume's
    // GAS is created from a single AABB. Because the index space is by definition axis aligned with the
    // volume's voxels, this AABB is the bounding-box of the volume's "active voxels".

    {
        auto grid_handle = volume.handle.grid<float>();

		// get this grid's aabb
        sutil::Aabb aabb;
        {
            // indexBBox returns the extrema of the (integer) voxel coordinates.
            // Thus the actual bounds of the space covered by those voxels extends
            // by one unit (or one "voxel size") beyond those maximum indices.
            auto bbox = grid_handle->indexBBox();
            nanovdb::Coord boundsMin( bbox.min() );
            nanovdb::Coord boundsMax( bbox.max() + nanovdb::Coord( 1 ) ); // extend by one unit

            float3 min = { 
                static_cast<float>( boundsMin[0] ), 
                static_cast<float>( boundsMin[1] ), 
                static_cast<float>( boundsMin[2] )};
            float3 max = {
                static_cast<float>( boundsMax[0] ),
                static_cast<float>( boundsMax[1] ),
                static_cast<float>( boundsMax[2] )};

            aabb =sutil::Aabb( min, max );
        }

		// up to device
        CUdeviceptr d_aabb;
        CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_aabb ), sizeof( sutil::Aabb ) ) );
        CUDA_CHECK( cudaMemcpy( reinterpret_cast<void* >(  d_aabb ), &aabb, 
            sizeof( sutil::Aabb ), cudaMemcpyHostToDevice ) );

        // Make build input for this grid
        uint32_t aabb_input_flags = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;
        OptixBuildInput build_input = {};
        build_input.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
        build_input.customPrimitiveArray.aabbBuffers = &d_aabb;
        build_input.customPrimitiveArray.flags = &aabb_input_flags;
        build_input.customPrimitiveArray.numSbtRecords = 1;
        build_input.customPrimitiveArray.numPrimitives = 1;
        build_input.customPrimitiveArray.sbtIndexOffsetBuffer = 0;
        build_input.customPrimitiveArray.sbtIndexOffsetSizeInBytes = 0;
        build_input.customPrimitiveArray.primitiveIndexOffset = 0;

        OptixAccelBuildOptions accel_options = {};
        accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
        accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

        OptixAccelBufferSizes gas_buffer_sizes;
        OPTIX_CHECK( optixAccelComputeMemoryUsage( context, &accel_options, 
            &build_input, 1, &gas_buffer_sizes ) );

        CUdeviceptr d_temp_buffer_gas;
        CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_temp_buffer_gas ),
            gas_buffer_sizes.tempSizeInBytes ) );
        CUdeviceptr d_output_buffer_gas;
        CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_output_buffer_gas ),
            gas_buffer_sizes.outputSizeInBytes ) );
        CUdeviceptr d_compacted_size;
        CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_compacted_size ), sizeof( size_t ) ) );

        OptixAccelEmitDesc emit_property = {};
        emit_property.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
        emit_property.result = d_compacted_size;

        OPTIX_CHECK( optixAccelBuild( context,
            0,
            &accel_options,
            &build_input,
            1,
            d_temp_buffer_gas,
            gas_buffer_sizes.tempSizeInBytes,
            d_output_buffer_gas,
            gas_buffer_sizes.outputSizeInBytes,
            &accel.handle,
            &emit_property,
            1 ) );
        CUDA_CHECK( cudaFree( reinterpret_cast<void*>( d_aabb ) ) );
        size_t compacted_size;
        CUDA_CHECK( cudaMemcpy( &compacted_size, reinterpret_cast<void*>( emit_property.result ),
            sizeof( size_t ), cudaMemcpyDeviceToHost ) );
        CUDA_CHECK( cudaFree( reinterpret_cast<void*>( d_compacted_size ) ) );
        if( compacted_size < gas_buffer_sizes.outputSizeInBytes ) 
        {
            CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &accel.d_buffer ), compacted_size ) );
            OPTIX_CHECK( optixAccelCompact( context, 0, accel.handle,
                accel.d_buffer, compacted_size, &accel.handle ) );
            CUDA_CHECK( cudaFree( reinterpret_cast<void*>( d_output_buffer_gas ) ) );
        }
        else 
        {
            accel.d_buffer = d_output_buffer_gas;
        }
        CUDA_CHECK( cudaFree( reinterpret_cast<void*>( d_temp_buffer_gas ) ) );
    }
}

void cleanupVolumeAccel( VolumeAccel& accel )
{
	CUDA_CHECK_NOTHROW( cudaFree( reinterpret_cast<void*>( accel.d_buffer ) ) );
}

void getOptixTransform( const Volume& grid, float transform[] )
{
    // Extract the index-to-world-space affine transform from the Grid and convert
    // to 3x4 row-major matrix for Optix.
	auto* grid_handle = grid.handle.grid<float>();
	const nanovdb::Map& map = grid_handle->map();
	transform[0] = map.mMatF[0]; transform[1] = map.mMatF[1]; transform[2]  = map.mMatF[2]; transform[3]  = map.mVecF[0];
	transform[4] = map.mMatF[3]; transform[5] = map.mMatF[4]; transform[6]  = map.mMatF[5]; transform[7]  = map.mVecF[1];
	transform[8] = map.mMatF[6]; transform[9] = map.mMatF[7]; transform[10] = map.mMatF[8]; transform[11] = map.mVecF[2];
}

sutil::Aabb worldAabb( const Volume& grid )
{
	auto* meta = grid.handle.gridMetaData();
	auto bbox = meta->worldBBox();
	float3 min = { static_cast<float>( bbox.min()[0] ),
                   static_cast<float>( bbox.min()[1] ),
                   static_cast<float>( bbox.min()[2] ) };
	float3 max = { static_cast<float>( bbox.max()[0] ),
                   static_cast<float>( bbox.max()[1] ),
                   static_cast<float>( bbox.max()[2] ) };

	return sutil::Aabb( min, max );
}