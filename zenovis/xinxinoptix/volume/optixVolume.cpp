#include "optixVolume.h"

#include "vec_math.h"
// #include "OptiXStuff.h"

#include <cstdint>
#include <filesystem>

#include <sutil/Exception.h>
#include <nanovdb/util/IO.h>
#include <nanovdb/util/GridStats.h>
#include <nanovdb/util/OpenToNanoVDB.h>

#include <glm/gtx/matrix_decompose.hpp>

#include <optix_stubs.h>
#include <optix_function_table_definition.h>

// ----------------------------------------------------------------------------
// Functions for manipulating Volume instances
// ----------------------------------------------------------------------------

template< class T >
using decay_t = typename std::decay<T>::type;

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
    else if(filePath.extension() == ".nvdb")
    {
        loadVolumeNVDB(volume, path);
    } else {
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

    volume.loadTasks.emplace_back([&volume, path, gridName=list[0].gridName] {
        loadGrid( volume.grid_density, path, gridName );
    });

    if (list.size() > 1) {
        volume.loadTasks.emplace_back([&volume, path, gridName=list[1].gridName] {
            loadGrid( volume.grid_temp, path, gridName );
        });
    }
}

void loadVolumeVDB(VolumeWrapper& volume, const std::string& path) {
    openvdb::initialize();
    openvdb::io::File file(path);
    
    file.open(); 

    openvdb::GridBase::Ptr baseGrid, tempGrid;

    if (file.getGrids()->size()>1) {

        for (openvdb::io::File::NameIterator nameIter = file.beginName();
        nameIter != file.endName(); ++nameIter)
        {
            // Read in only the grid we are interested in.
            if (nameIter.gridName() == "density") { // temperature
                baseGrid = file.readGrid(nameIter.gridName());
            } else if (nameIter.gridName() == "temperature") {
                tempGrid = file.readGrid(nameIter.gridName());
            }else {
                std::cout << "skipping grid " << nameIter.gridName() << std::endl;
            }
        }

    } else if(file.getGrids()->size()==1) {
        auto nameIter = file.beginName();
        baseGrid = file.readGrid(nameIter.gridName());
    } else {
        throw std::runtime_error("This VDB file doesn't have any grid");
    }
    
    file.close();

    auto new_matrix = volume.transform;

    glm::f64mat4 test;

    auto old_matrix = [&]() -> glm::mat4 {

        auto tmp = baseGrid->transform().baseMap()->getAffineMap()->getMat4();
        glm::mat4 result;
        for (uint i=0; i<16; ++i) {
            auto ele = *(tmp[0]+i);
            result[i/4][i%4] = float(ele);
        }
        return result;
    }();

    openvdb::math::Mat4f transformMat = openvdb::math::Mat4f::identity();
    auto linearTransform = openvdb::math::Transform::createLinearTransform(transformMat);     

    auto new_transform = //openvdb::math::Transform::Ptr(linearTransform);
        baseGrid->transform().copy(); //.createLinearTransform();

    //scale_ *= _scale;
    //translation_ += _translation;

    new_matrix = new_matrix * old_matrix;

    //new_transform->preScale(openvdb::math::Vec3(scale_.x, scale_.y, scale_.z));
    //new_transform->preTranslate(openvdb::math::Vec3(translation_.x, translation_.y, translation_.z));

    auto test_mat = new_transform->baseMap()->getAffineMap()->getMat4();

    for (uint i=0; i<16; ++i) {
        *(test_mat[0]+i) = double(new_matrix[i/4][i%4]);
    }

    openvdb::math::Transform::Ptr test_transform =
    openvdb::math::Transform::createLinearTransform(test_mat);

    baseGrid->setTransform(test_transform);
    if (tempGrid != nullptr) {
        tempGrid->setTransform(test_transform);
    }

    glm::vec3 _scale;
    glm::quat _rotation;
    glm::vec3 _translation;
    glm::vec3 _skew;
    glm::vec4 _perspective;
    glm::decompose(old_matrix, _scale, _rotation, _translation, _skew, _perspective);

    glm::vec3 scale_;
    glm::quat rotation_;
    glm::vec3 translation_;
    glm::vec3 skew_;
    glm::vec4 perspective_;
    glm::decompose(new_matrix, scale_, rotation_, translation_, skew_, perspective_);

    auto parsing = [](openvdb::GridBase::Ptr& grid_ptr, nanovdb::GridHandle<>& result) {

        openvdb::FloatGrid::Ptr srcGrid = openvdb::gridPtrCast<openvdb::FloatGrid>(grid_ptr);
        result = nanovdb::openToNanoVDB(*srcGrid, nanovdb::StatsMode::All); 
    };

    parsing(baseGrid, volume.grid_density.handle);
    volume.loadTasks.emplace_back([&volume, path, gridName=baseGrid->getName()] {
        loadGrid(volume.grid_density, path, gridName);
    });

    if (tempGrid != nullptr) {
        parsing(tempGrid, volume.grid_temp.handle);
        volume.loadTasks.emplace_back([&volume, path, gridName=tempGrid->getName()] {
            loadGrid(volume.grid_temp, path, gridName);
        });
    }
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

    auto* meta = gridHdl.gridMetaData();
    if( meta->isPointData() )
        throw std::runtime_error("NanoVDB Point Data cannot be handled by Zeno Optix");
    if( meta->isLevelSet() )
        throw std::runtime_error("NanoVDB Level Sets cannot be handled by Zeno Optix");

    if (grid.deviceptr != 0) { return; }

    // auto x = gridHdl.grid<nanovdb::Fp16>();
    // const nanovdb::Map& map = gridHdl.grid<float>()->map();

    // NanoVDB files represent the sparse data-structure as flat arrays that can be
    // uploaded to the device "as-is".
    assert( gridHdl.size() != 0 );
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &grid.deviceptr ), gridHdl.size() ) );
    CUDA_CHECK( cudaMemcpy( reinterpret_cast<void*>( grid.deviceptr ), gridHdl.data(), gridHdl.size(), cudaMemcpyHostToDevice ) );

        auto* tmp_grid = gridHdl.grid<float>(); //.grid<nanovdb::FloatGrid>();
        
        auto vmax = tmp_grid->indexBBox().max();
        auto vmin = tmp_grid->indexBBox().min();
        auto accessor = tmp_grid->getAccessor();

        float max_value = 1.0;

        //vsize[0]; vsize[1]; vsize[2];
        for (int32_t i=vmin[0]; i<vmax[0]; ++i) {
            for (int32_t j=vmin[1]; j<vmax[1]; ++j) {
                for (int32_t k=vmin[2]; k<vmax[2]; ++k) {
                    auto coord = nanovdb::Coord(i, j, k);
                    auto value = accessor.getValue(coord);
                    max_value = fmaxf(max_value, value);
                }
            }
        }

        grid.max_value = max_value;

    std::cout << "max value in filename: " << path << " girdname " << gridname << " is " << max_value << std::endl;
}

void unloadGrid(GridWrapper& grid) {
    //grid.handle.reset();
    CUDA_CHECK_NOTHROW( cudaFree( reinterpret_cast<void*>( grid.deviceptr ) ) );
}

void cleanupVolume( VolumeWrapper& volume )
{
    // OptiX cleanup
    if ( volume.grid_density.deviceptr != 0) {
	    CUDA_CHECK_NOTHROW( cudaFree( reinterpret_cast<void*>( volume.grid_density.deviceptr ) ) );
        volume.grid_density.deviceptr = 0;
    }
    if ( volume.grid_temp.deviceptr != 0) {
        CUDA_CHECK_NOTHROW( cudaFree( reinterpret_cast<void*>( volume.grid_temp.deviceptr ) ) ); 
        volume.grid_temp.deviceptr = 0;
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
        auto grid_handle = volume.grid_density.handle.grid<float>();

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
        uint32_t aabb_input_flags = OPTIX_GEOMETRY_FLAG_NONE;
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
    if (accel.d_buffer != 0) {
	    CUDA_CHECK_NOTHROW( cudaFree( reinterpret_cast<void*>( accel.d_buffer ) ) );
    }
}

void getOptixTransform( const VolumeWrapper& volume, float transform[] )
{
    // Extract the index-to-world-space affine transform from the Grid and convert
    // to 3x4 row-major matrix for Optix.
	auto* grid_handle = volume.grid_density.handle.grid<float>();
	const nanovdb::Map& map = grid_handle->map();
	transform[0] = map.mMatF[0]; transform[1] = map.mMatF[1]; transform[2]  = map.mMatF[2]; transform[3]  = map.mVecF[0];
	transform[4] = map.mMatF[3]; transform[5] = map.mMatF[4]; transform[6]  = map.mMatF[5]; transform[7]  = map.mVecF[1];
	transform[8] = map.mMatF[6]; transform[9] = map.mMatF[7]; transform[10] = map.mMatF[8]; transform[11] = map.mVecF[2];
}

sutil::Aabb worldAabb( const VolumeWrapper& volume )
{
	auto* meta = volume.grid_density.handle.gridMetaData();

	auto bbox = meta->worldBBox();
	float3 min = { static_cast<float>( bbox.min()[0] ),
                   static_cast<float>( bbox.min()[1] ),
                   static_cast<float>( bbox.min()[2] ) };
	float3 max = { static_cast<float>( bbox.max()[0] ),
                   static_cast<float>( bbox.max()[1] ),
                   static_cast<float>( bbox.max()[2] ) };

	return sutil::Aabb( min, max );
}