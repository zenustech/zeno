#include "optixVolume.h"

#include <iostream>
#include <filesystem>
#include <nanovdb/util/IO.h>
#include <nanovdb/util/GridStats.h>
#include <nanovdb/util/OpenToNanoVDB.h>

// ----------------------------------------------------------------------------
// Functions for manipulating Volume instances
// ----------------------------------------------------------------------------

void checkGridName( const std::string& path, std::string& name) {
    openvdb::initialize();
    openvdb::io::File file(path);
    
    file.open(); 

    if ( file.hasGrid(name) ) { return; }

    std::cout << "<<< Error: " << path << " >>>" << std::endl;
    std::cout << "This VDB file doesn't have grid named << " << name << " >>" << std::endl;
    
    auto ni = file.beginName();
    name = ni.gridName();

    std::cout << "Trying to read grid << " << name << " >> instead." << std::endl;
}

std::string fetchGridName( const std::string& path, uint index ) {
    openvdb::initialize();
    openvdb::io::File file(path);
    
    file.open(); 

    const auto grid_count = file.getGrids()->size();

    if (grid_count == 0) {
        throw std::runtime_error("This VDB file doesn't have any grid");
    }

    if (index >= grid_count) {
        std::cout << "<<< Error: " << path << " >>>" << std::endl;

        std::cout << "This VDB file doesn't have grid at index " << index << std::endl;
        std::cout << "Trying to read grid at index 0" << std::endl; 
        index = 0;
    }

    auto ni = file.beginName();

    for (uint i=0; i<index && ni != file.endName(); ++i)
    {
        ++ni;//tmp_grids.push_back(file.readGrid(nameIter.gridName()));
    } 
    return ni.gridName();
}

bool loadVolume( VolumeWrapper& volume, const std::string& path )
{
    std::filesystem::path filePath = path;

    if ( !std::filesystem::exists(filePath) ) {
        std::cout << filePath.filename() << " doesn't exist";
        return false;
    }

    if (filePath.extension() == ".vdb")
    {
        loadVolumeVDB(volume, path);
    }
    // else if(filePath.extension() == ".nvdb")
    // {
    //     //loadVolumeNVDB(volume, path);
    // } 
    else {
        std::cout << filePath.filename() << " is unsupported type";
        return false;
    }
    return true;
}

void loadVolumeNVDB( VolumeWrapper& volume, const std::string& path) {

    auto list= nanovdb::io::readGridMetaData( path );
    assert( list.size() > 0 );

    std::cerr << "Opened file " << path << std::endl;
    std::cerr << "    grids:" << std::endl;
    for (auto& m : list) {
        std::cerr << "        " << m.gridName << std::endl;
    }

    volume.grids.clear();
    volume.grids.resize(list.size());
    
    volume.tasks.clear();

    for (uint i=0; i<list.size(); ++i) {
        volume.tasks.emplace_back([&volume, path, i] {
            loadGrid( *volume.grids[i], path, i);
        });
    }
}

void loadVolumeVDB(VolumeWrapper& volume, const std::string& path) {
    openvdb::initialize();
    openvdb::io::File file(path);
    
    file.open(); 

    const bool picking = volume.selected.size() > 0;

    const auto grid_count = picking?  volume.selected.size():file.getGrids()->size();  
    if (grid_count == 0) {
        throw std::runtime_error("This VDB file doesn't have any grid");
    }

    std::vector<openvdb::GridBase::Ptr> tmp_grids;

    if (picking) {

        for (uint i=0; i<grid_count; ++i) {
            auto selected = volume.selected.at(i);

            if (file.hasGrid(selected)) {
                tmp_grids.push_back(file.readGrid(volume.selected.at(i)));
            } else {

                std::cout << "<<< Error: " << path << " >>>" << std::endl;
                std::cout << "This VDB file doesn't have grid named << " << selected << " >>" << std::endl;

                auto nameIter = file.beginName();
                auto realName = nameIter.gridName();
                volume.selected[i] = realName;

                std::cout << "Trying to read grid << " << realName << " >> instead." << std::endl;

                tmp_grids.push_back(file.readGrid(realName));
            } 
        }
    } 
    else {
        for (auto nameIter = file.beginName(); nameIter != file.endName(); ++nameIter)
            tmp_grids.push_back(file.readGrid(nameIter.gridName()));
    }

    assert(tmp_grids.size() != 0);
    auto baseGrid = tmp_grids.front();
    
    file.close();

    const auto parent_matrix = volume.transform;

    const auto child_matrix = [&]() -> auto {

        auto tmp = baseGrid->transform().baseMap()->getAffineMap()->getMat4();
        glm::mat4 result;
        for (uint i=0; i<16; ++i) {
            auto ele = *(tmp[0]+i);
            result[i/4][i%4] = ele;
        }
        return result;
    }();

    auto result_matrix = parent_matrix * child_matrix;  

    auto vdb_transform = baseGrid->transform().copy(); //.createLinearTransform();
    auto vdb_matrix = vdb_transform->baseMap()->getAffineMap()->getMat4();

    for (uint i=0; i<16; ++i) {
        *(vdb_matrix[0]+i) = result_matrix[i/4][i%4];
    }

    auto result_transform = openvdb::math::Transform::createLinearTransform(vdb_matrix);

    volume.grids.clear();
    volume.grids.reserve(tmp_grids.size());

    volume.tasks.clear();
    volume.tasks.reserve(tmp_grids.size());

    for (uint i=0; i<grid_count; ++i) {
        auto grid = tmp_grids[i];
        grid->setTransform(result_transform);
         
        //volume.grids.push_back(GridWrapper());
        //parsing(grid, volume.grids[i].handle);

        makeTypedGridWrapper(volume.type, volume.grids);
        volume.grids[i]->parsing(grid);

        if (picking) {
            auto name = volume.selected[i];
            volume.tasks.emplace_back([&volume, path, name, i] {
                loadGrid(*volume.grids[i], path, name);
            });
        } else {
            volume.tasks.emplace_back([&volume, path, i] {
                loadGrid(*volume.grids[i], path, i);
            });
        } // else 
    };

    std::cout << "-----------------------------------------------" << std::endl;
}

static void processGrid(GridWrapper& grid, const std::string& path) 
{
    auto& gridHdl = grid.handle;

    auto* meta = gridHdl.gridMetaData();
    if( meta->isPointData() )
        throw std::runtime_error("NanoVDB Point Data cannot be handled by Zeno Optix");
    if( meta->isLevelSet() )
        throw std::runtime_error("NanoVDB Level Sets cannot be handled by Zeno Optix");

    if (grid.deviceptr != 0) { return; }

    // NanoVDB files represent the sparse data-structure as flat arrays that can be
    // uploaded to the device "as-is".
    assert( gridHdl.size() != 0 );
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &grid.deviceptr ), gridHdl.size() ) );
    CUDA_CHECK( cudaMemcpy( reinterpret_cast<void*>( grid.deviceptr ), gridHdl.data(), gridHdl.size(), cudaMemcpyHostToDevice ) );

    grid.max_value = grid.analysis(path);
}

void loadGrid( GridWrapper& grid, const std::string& path, const uint index ) 
{
    auto& gridHdl = grid.handle;

    if ( gridHdl.size() == 0 ) {
        grid.handle = nanovdb::io::readGrid<>( path, index );
    }

    processGrid(grid, path);
}

 void loadGrid( GridWrapper& grid, const std::string& path, const std::string& gridname )
{
    auto& gridHdl = grid.handle;

    if ( gridHdl.size() == 0 ) {
        if( gridname.length() > 0 )
            gridHdl = nanovdb::io::readGrid<>( path, gridname );
        else
            gridHdl = nanovdb::io::readGrid<>( path );
    }

    if( !gridHdl ) 
    {
        std::stringstream ss;
        ss << "Unable to read " << gridname << " from " << path;
        throw std::runtime_error( ss.str() );
    }

    processGrid(grid, path);
}

void unloadGrid(GridWrapper& grid) {
    //grid.handle.reset();
    if (0 != grid.deviceptr) {
        CUDA_CHECK_NOTHROW( cudaFree( reinterpret_cast<void*>( grid.deviceptr ) ) );
        grid.deviceptr = 0;
    }
}

void cleanupVolume( VolumeWrapper& volume )
{
    // OptiX cleanup
    for (auto grid : volume.grids) {
        unloadGrid(*grid);
    }
}

void buildVolumeAccel( VolumeAccel& accel, const VolumeWrapper& volume, const OptixDeviceContext& context )
{
    // Build accel for the volume and store it in a VolumeAccel struct.
    //
    // For Optix the NanoVDB volume is represented as a 3D box in index coordinate space. The volume's
    // GAS is created from a single AABB. Because the index space is by definition axis aligned with the
    // volume's voxels, this AABB is the bounding-box of the volume's "active voxels".
    {
		// get this grid's aabb
        sutil::Aabb aabb = [&]()
        {
            if (volume.grids.size() == 0) {
                return sutil::Aabb( {-0.5, -0.5, -0.5}, {0.5, 0.5, 0.5} );
            }
            // indexBBox returns the extrema of the (integer) voxel coordinates.
            // Thus the actual bounds of the space covered by those voxels extends
            // by one unit (or one "voxel size") beyond those maximum indices.
            auto baseGrid = volume.grids.front();
            auto bbox = baseGrid->indexedBox();
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

            return sutil::Aabb( min, max );
        }();

		// up to device
        CUdeviceptr d_aabb;
        CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_aabb ), sizeof( sutil::Aabb ) ) );
        CUDA_CHECK( cudaMemcpy( reinterpret_cast<void* >(  d_aabb ), &aabb, 
            sizeof( sutil::Aabb ), cudaMemcpyHostToDevice ) );

        // Make build input for this grid
        uint32_t aabb_input_flags = OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL;
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
        accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION; //| OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS | OPTIX_BUILD_FLAG_ALLOW_RANDOM_INSTANCE_ACCESS;
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
    if (accel.d_buffer != 0) {
	    CUDA_CHECK_NOTHROW( cudaFree( reinterpret_cast<void*>( accel.d_buffer ) ) );

        accel.d_buffer = 0u;
        accel.handle = 0u;
    }
}

void getOptixTransform( const VolumeWrapper& volume, float transform[] )
{
    // Extract the index-to-world-space affine transform from the Grid and convert
    // to 3x4 row-major matrix for Optix.
    if (volume.grids.size() == 0) {
        auto dummy = glm::transpose(volume.transform);
        auto dummy_ptr = glm::value_ptr( dummy );
        for (size_t i=0; i<12; ++i) {   
            transform[i] = dummy_ptr[i];
        }
        return;
    }

    auto baseGrid = volume.grids.front();
    const nanovdb::Map& map = baseGrid->nanoMAP();

	transform[0] = map.mMatF[0]; transform[1] = map.mMatF[1]; transform[2]  = map.mMatF[2]; transform[3]  = map.mVecF[0];
	transform[4] = map.mMatF[3]; transform[5] = map.mMatF[4]; transform[6]  = map.mMatF[5]; transform[7]  = map.mVecF[1];
	transform[8] = map.mMatF[6]; transform[9] = map.mMatF[7]; transform[10] = map.mMatF[8]; transform[11] = map.mVecF[2];
}

sutil::Aabb worldAabb( const VolumeWrapper& volume )
{
    auto baseGrid = volume.grids.front();
	auto* meta = baseGrid->handle.gridMetaData();

	auto bbox = meta->worldBBox();
	float3 min = { static_cast<float>( bbox.min()[0] ),
                   static_cast<float>( bbox.min()[1] ),
                   static_cast<float>( bbox.min()[2] ) };
	float3 max = { static_cast<float>( bbox.max()[0] ),
                   static_cast<float>( bbox.max()[1] ),
                   static_cast<float>( bbox.max()[2] ) };

	return sutil::Aabb( min, max );
}