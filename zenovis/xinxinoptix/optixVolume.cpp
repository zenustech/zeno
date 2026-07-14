#include "optixVolume.h"

#include "xinxinoptixapi.h"
#include "TypeCaster.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <filesystem>
#include <random>
#include <sstream>
#include <vector>
#include <limits>
#include <cstring>
#include <chrono>
#include <cuda_runtime_api.h>
#include <nanovdb/util/IO.h>
#include <nanovdb/util/GridStats.h>
#include <nanovdb/util/OpenToNanoVDB.h>

// ----------------------------------------------------------------------------
// Functions for manipulating Volume instances
// ----------------------------------------------------------------------------

void checkGridName( const std::string& path, std::string& name) {
    openvdb::initialize();
    openvdb::io::File file(path);

    file.setCopyMaxBytes(0);
    file.open(true);

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

    file.setCopyMaxBytes(0);
    file.open(true);

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

static uint selectDensityGridIndex(const std::vector<openvdb::GridBase::Ptr>& grids) {
    uint first_float_grid = 0;
    bool has_float_grid = false;

    for (uint i = 0; i < grids.size(); ++i) {
        auto& grid = grids[i];
        if (!grid || !grid->isType<openvdb::FloatGrid>()) {
            continue;
        }
        if (!has_float_grid) {
            first_float_grid = i;
            has_float_grid = true;
        }
        if (isDensityVDBChannelName(grid->getName())) {
            return i;
        }
    }

    return has_float_grid ? first_float_grid : 0;
}

static std::shared_ptr<GridWrapper> densityGridForVolume(const VolumeWrapper& volume) {
    if (volume.grids.empty()) {
        return nullptr;
    }

    const auto index = std::min<size_t>(volume.density_grid_index, volume.grids.size() - 1);
    return volume.grids[index];
}

void loadVolumeVDB(VolumeWrapper& volume, const std::string& path) {
    openvdb::initialize();
    openvdb::io::File file(path);
    file.open(false);

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
    
    file.close();

    volume.grids.clear();
    volume.grids.reserve(tmp_grids.size());
    volume.density_grid_index = selectDensityGridIndex(tmp_grids);
    volume.aggregate = {};

    volume.tasks.clear();
    volume.tasks.reserve(tmp_grids.size());

    for (uint i=0; i<grid_count; ++i) {
        auto grid = tmp_grids[i];
        //grid->setTransform(result_transform);
         
        //volume.grids.push_back(GridWrapper());
        //parsing(grid, volume.grids[i].handle);

        makeTypedGridWrapper(volume.type, volume.grids);
        volume.grids[i]->parsing(grid);
        if (!volume.use_gpu_baked_octree && i == volume.density_grid_index && grid->isType<openvdb::FloatGrid>())
        {
            const auto fgrid = openvdb::gridPtrCast<openvdb::FloatGrid>(grid);
            auto& aggregate = volume.aggregate;
            const uint32_t octreeBuildDepth = bakedSparseVolumeClampOctreeBuildDepth(volume.octreeBuildDepth);
            aggregate.aggregate(*fgrid, int(octreeBuildDepth));
        }

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
        std::cerr << "NanoVDB Point Data cannot be handled by Zeno Optix";
    if( meta->isLevelSet() )
        std::cerr << "NanoVDB Level Sets cannot be handled by Zeno Optix";

    if (grid.buffer.handle != 0) { return; }

    // NanoVDB files represent the sparse data-structure as flat arrays that can be
    // uploaded to the device "as-is".
    assert( gridHdl.size() != 0 );
    //grid.buffer.resize(gridHdl.size());
    grid.buffer.allocAndUpload(gridHdl.size(), gridHdl.data());

    grid.analysis(path);
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
    grid.buffer.reset();
    grid.handle.reset();
}

static int roundUpInt(int value, int multiple)
{
    return ((value + multiple - 1) / multiple) * multiple;
}

static std::vector<int3> makeDomainBrickOrigins(const int3& voxel_min, const int3& brick_dim)
{
    std::vector<int3> origins;
    origins.reserve(uint64_t(brick_dim.x) * uint64_t(brick_dim.y) * uint64_t(brick_dim.z));
    for (int z = 0; z < brick_dim.z; ++z) {
        for (int y = 0; y < brick_dim.y; ++y) {
            for (int x = 0; x < brick_dim.x; ++x) {
                origins.push_back(make_int3(
                    voxel_min.x + x * int(BAKED_SPARSE_VOLUME_BRICK_SIZE),
                    voxel_min.y + y * int(BAKED_SPARSE_VOLUME_BRICK_SIZE),
                    voxel_min.z + z * int(BAKED_SPARSE_VOLUME_BRICK_SIZE)));
            }
        }
    }
    return origins;
}

static uint32_t bakedSparseDenseOctreeNodeCapacity(uint32_t octreeBuildDepth)
{
    return ((1u << (3u * (octreeBuildDepth + 1u))) - 1u) / 7u;
}

static uint16_t halfBits(const __half& value)
{
    uint16_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static bool sameOcNode(const OcNode& lhs, const OcNode& rhs)
{
    return lhs.data == rhs.data
        && halfBits(lhs.min_d) == halfBits(rhs.min_d)
        && halfBits(lhs.max_d) == halfBits(rhs.max_d);
}
bool bakeDensityGridToSparseBricksOnGPU(
    GridWrapper& grid,
    BakedSparseVolume& baked_volume,
    const xinxinoptix::VDBDensityBakeInputs& inputs,
    const xinxinoptix::VDBDensityBakeOptions& options,
    xinxinoptix::VDBDensityBakeResult* result)
{

    if (grid.handle.size() == 0 || grid.handle.data() == nullptr) {
        std::cerr << "VDB sparse brick bake skipped: NanoVDB host grid is empty" << std::endl;
        return false;
    }
    double resident_upload_ms = 0.0;
    if (grid.buffer.handle == 0) {
        grid.buffer.allocAndUpload(grid.handle.size(), grid.handle.data());
    }

    const auto bbox = grid.indexedBox();
    if (bbox.empty()) {
        std::cerr << "VDB sparse brick bake skipped: empty density bbox" << std::endl;
        return false;
    }

    const uint32_t octreeBuildDepth = bakedSparseVolumeClampOctreeBuildDepth(options.octreeBuildDepth);
    const int3 source_sample_min {
        bbox.min().x(),
        bbox.min().y(),
        bbox.min().z()
    };
    const int3 source_sample_max {
        bbox.max().x() + 1,
        bbox.max().y() + 1,
        bbox.max().z() + 1
    };
    const int3 sample_min = options.use_custom_sample_bbox
        ? options.custom_sample_min
        : source_sample_min;
    const int3 sample_max = options.use_custom_sample_bbox
        ? options.custom_sample_max
        : source_sample_max;
    if (sample_min.x >= sample_max.x || sample_min.y >= sample_max.y || sample_min.z >= sample_max.z) {
        std::cerr << "VDB sparse brick bake skipped: empty sample bbox" << std::endl;
        return false;
    }

    const int3 sample_dim {
        sample_max.x - sample_min.x,
        sample_max.y - sample_min.y,
        sample_max.z - sample_min.z
    };
    const int leaf_res = 1 << octreeBuildDepth;
    const int padded_x = roundUpInt(sample_dim.x + options.topology_padding_voxels * 2, leaf_res);
    const int padded_y = roundUpInt(sample_dim.y + options.topology_padding_voxels * 2, leaf_res);
    const int padded_z = roundUpInt(sample_dim.z + options.topology_padding_voxels * 2, leaf_res);
    const int3 voxel_min {
        sample_min.x - options.topology_padding_voxels,
        sample_min.y - options.topology_padding_voxels,
        sample_min.z - options.topology_padding_voxels
    };
    const int3 voxel_dim { padded_x, padded_y, padded_z };
    const int3 voxel_max { voxel_min.x + voxel_dim.x, voxel_min.y + voxel_dim.y, voxel_min.z + voxel_dim.z };
    const int3 brick_dim {
        roundUpInt(voxel_dim.x, int(BAKED_SPARSE_VOLUME_BRICK_SIZE)) / int(BAKED_SPARSE_VOLUME_BRICK_SIZE),
        roundUpInt(voxel_dim.y, int(BAKED_SPARSE_VOLUME_BRICK_SIZE)) / int(BAKED_SPARSE_VOLUME_BRICK_SIZE),
        roundUpInt(voxel_dim.z, int(BAKED_SPARSE_VOLUME_BRICK_SIZE)) / int(BAKED_SPARSE_VOLUME_BRICK_SIZE)
    };
    const uint64_t table_count_u64 = uint64_t(brick_dim.x) * uint64_t(brick_dim.y) * uint64_t(brick_dim.z);
    if (table_count_u64 > std::numeric_limits<uint32_t>::max()) {
        std::cerr << "VDB sparse brick bake skipped: brick table is too large" << std::endl;
        return false;
    }
    const auto brick_origins = makeDomainBrickOrigins(voxel_min, brick_dim);
    const uint32_t brick_count = uint32_t(brick_origins.size());
    if (brick_count == 0) {
        std::cerr << "VDB sparse brick bake skipped: empty brick domain" << std::endl;
        return false;
    }
    const auto domain_end = std::chrono::steady_clock::now();

    const auto alloc_begin = std::chrono::steady_clock::now();
    baked_volume.reset();
    baked_volume.device.voxel_min = voxel_min;
    baked_volume.device.voxel_dim = voxel_dim;
    baked_volume.device.voxel_max = voxel_max;
    baked_volume.device.sample_min = sample_min;
    baked_volume.device.sample_max = sample_max;
    baked_volume.device.brick_dim = brick_dim;
    baked_volume.device.brick_size = BAKED_SPARSE_VOLUME_BRICK_SIZE;
    baked_volume.device.brick_count = brick_count;
    baked_volume.device.brick_table_count = uint32_t(table_count_u64);
    baked_volume.device.octreeBuildDepth = octreeBuildDepth;
    baked_volume.device.octree_node_count = bakedSparseDenseOctreeNodeCapacity(octreeBuildDepth);
    baked_volume.device.octree_format = 0;

    baked_volume.d_brick_table.resize(sizeof(int) * baked_volume.device.brick_table_count);
    baked_volume.d_brick_origins.resize(sizeof(int3) * brick_count);
    const auto origin_upload_begin = std::chrono::steady_clock::now();
    const auto origin_copy = cudaMemcpy(
        reinterpret_cast<void*>(baked_volume.d_brick_origins.handle),
        brick_origins.data(),
        sizeof(int3) * brick_origins.size(),
        cudaMemcpyHostToDevice);
    if (origin_copy != cudaSuccess) {
        std::cerr << "VDB sparse brick bake failed to upload brick origins: " << cudaGetErrorString(origin_copy) << std::endl;
        baked_volume.reset();
        return false;
    }
    const auto origin_upload_end = std::chrono::steady_clock::now();
    baked_volume.d_voxel_values.resize(sizeof(uint16_t) * uint64_t(brick_count) * 512ull);
    baked_volume.d_brick_min.resize(sizeof(uint16_t) * brick_count);
    baked_volume.d_brick_max.resize(sizeof(uint16_t) * brick_count);
    baked_volume.d_octree.resize(sizeof(OcNode) * bakedSparseDenseOctreeNodeCapacity(octreeBuildDepth));
    baked_volume.d_descriptor.resize(sizeof(BakedSparseVolumeDevice));

    baked_volume.device.brick_table = reinterpret_cast<int*>(baked_volume.d_brick_table.handle);
    baked_volume.device.brick_origins = reinterpret_cast<int3*>(baked_volume.d_brick_origins.handle);
    baked_volume.device.voxel_values = reinterpret_cast<uint16_t*>(baked_volume.d_voxel_values.handle);
    baked_volume.device.brick_min = reinterpret_cast<uint16_t*>(baked_volume.d_brick_min.handle);
    baked_volume.device.brick_max = reinterpret_cast<uint16_t*>(baked_volume.d_brick_max.handle);
    baked_volume.device.octree = reinterpret_cast<OcNode*>(baked_volume.d_octree.handle);
    xinxinoptix::VDBDensityBakeInputs bake_inputs = inputs;

    const bool baked = xinxinoptix::bakeNanoVDBGridToSparseBricks(
        grid.buffer.size,
        baked_volume.device,
        bake_inputs,
        options,
        result);
    if (!baked) {
        std::cout << "VDB sparse bake host profile {"
            << (bake_inputs.validation_label != nullptr ? bake_inputs.validation_label : "density") << std::endl;
        baked_volume.reset();
        return false;
    }

    cudaMemcpy(
        reinterpret_cast<void*>(baked_volume.d_descriptor.handle),
        &baked_volume.device,
        sizeof(BakedSparseVolumeDevice),
        cudaMemcpyHostToDevice);
    const uint64_t brick_bytes = uint64_t(brick_count) * 512ull * sizeof(uint16_t);
    if (result != nullptr) {
        std::cout << "VDB sparse GPU bake {"
            << (bake_inputs.validation_label != nullptr ? bake_inputs.validation_label : "density")
            << "} total_wall=" << result->sparse_total_wall_ms << " ms"
            << " density_gpu=" << result->sparse_density_gpu_ms << " ms"
            << " octree_wall=" << result->sparse_octree_wall_ms << " ms"
            << " octree_gpu=" << result->sparse_octree_gpu_ms << " ms"
            << " accumulate=" << result->sparse_octree_accumulate_ms << " ms"
            << " reduce=" << result->sparse_octree_reduce_ms << " ms"
            << " compact=" << result->sparse_octree_compact_ms << " ms"
            << std::endl;
    }
    return true;
}

static bool hasVolumeDeviceData(const VolumeWrapper& volume)
{
    for (const auto& grid : volume.grids) {
        if (grid && grid->buffer.handle != 0) {
            return true;
        }
    }
    return volume.node->handle != 0
        || volume.d_aabb->handle != 0
        || volume.d_octree->handle != 0
        || (volume.baked_density && volume.baked_density->valid());
}

void releaseVolumeDeviceData(VolumeWrapper& volume)
{
    // Keep NanoVDB host handles and the CPU octree cached; only drop device-side residency.
    volume.dirty = true;
    if (!hasVolumeDeviceData(volume)) {
        return;
    }

    for (auto& grid : volume.grids) {
        if (grid) {
            grid->buffer.reset();
        }
    }
    volume.density_bake_key.clear();
    if (volume.baked_density) {
        volume.baked_density->reset();
    }
    cleanupVolumeAccel(*volume.node);
    volume.d_aabb->reset();
    volume.d_octree->reset();
}

void cleanupVolume( VolumeWrapper& volume )
{
    // OptiX cleanup
    for (auto grid : volume.grids) {
        unloadGrid(*grid);
    }
    volume = {};
}

void buildVolumeAccel( VolumeWrapper& volume, const OptixDeviceContext& context )
{
    // Build accel for the volume and store it in a VolumeAccel struct.
    //
    // For Optix the NanoVDB volume is represented as a 3D box in index coordinate space. The volume's
    // GAS is created from a single AABB. Because the index space is by definition axis aligned with the
    // volume's voxels, this AABB is the bounding-box of the volume's "active voxels".
    auto& accel = *volume.node;
    {
		// get this grid's aabb
        sutil::Aabb aabb = sutil::Aabb( make_float3(0), make_float3(1) );

        if (volume.baked_density && volume.baked_density->valid()) {
            const auto& sample_min = volume.baked_density->device.sample_min;
            const auto& sample_max = volume.baked_density->device.sample_max;
            aabb.m_min = make_float3(sample_min.x, sample_min.y, sample_min.z);
            aabb.m_max = make_float3(sample_max.x, sample_max.y, sample_max.z);
        } else if (auto grid = densityGridForVolume(volume)) {
            auto bbox = grid->indexedBox();
            auto min3s = bbox.min().asVec3s();
            auto max3s = bbox.max().asVec3s();
            aabb.m_min = make_float3(min3s[0], min3s[1], min3s[2]);
            aabb.m_max = make_float3(max3s[0] + 1.0f, max3s[1] + 1.0f, max3s[2] + 1.0f);
        }

        auto aabb_ptr = &aabb;
        auto count = 1;
        // if (!volume.aabbs.empty()) {
        //     aabb_ptr = volume.aabbs.data();
        //     count = volume.aabbs.size();
        // }
        size_t byte_size = sizeof(sutil::Aabb) * count;

		// up to device
        auto& d_aabb = *volume.d_aabb;
        d_aabb.allocAndUpload(byte_size, (const uint8_t*)aabb_ptr);

        uint64_t octree_handle = 0;
        uint64_t baked_density_handle = 0;
        uint32_t octreeBuildDepth = bakedSparseVolumeClampOctreeBuildDepth(volume.octreeBuildDepth);
        const bool use_gpu_octree = volume.use_gpu_baked_octree
            && volume.baked_density
            && volume.baked_density->valid();

        if (use_gpu_octree) {
            octree_handle = volume.baked_density->d_octree.handle;
            baked_density_handle = volume.baked_density->d_descriptor.handle;
            octreeBuildDepth = volume.baked_density->device.octreeBuildDepth;
            volume.d_octree->reset();
        } else if (!volume.aggregate.octree.empty()) {
            auto byte_size = sizeof(OcNode) * volume.aggregate.octree.size();
            volume.d_octree->allocAndUpload( byte_size, (const uint8_t*)volume.aggregate.octree.data() );
            octree_handle = volume.d_octree->handle;
            octreeBuildDepth = bakedSparseVolumeClampOctreeBuildDepth(uint32_t(volume.aggregate.octreeBuildDepth));
        } else {
            volume.d_octree->reset();
        }
        octreeBuildDepth = bakedSparseVolumeClampOctreeBuildDepth(octreeBuildDepth);

        // Make build input for this grid
        uint32_t aabb_input_flags = OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL;
        OptixBuildInput build_input = {};
        build_input.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
        build_input.customPrimitiveArray.aabbBuffers = &d_aabb.handle;
        
        build_input.customPrimitiveArray.flags = &aabb_input_flags;
        build_input.customPrimitiveArray.numSbtRecords = 1;
        build_input.customPrimitiveArray.numPrimitives = count;
        build_input.customPrimitiveArray.sbtIndexOffsetBuffer = 0;
        build_input.customPrimitiveArray.sbtIndexOffsetSizeInBytes = 0;
        build_input.customPrimitiveArray.primitiveIndexOffset = 0;

        OptixAccelBuildOptions accel_options = {};
        accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS | OPTIX_BUILD_FLAG_ALLOW_RANDOM_INSTANCE_ACCESS;
        accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

        xinxinoptix::buildXAS(context, accel_options, build_input, accel.buffer, accel.handle, 128);
        cudaMemcpy((char*)accel.buffer.handle+128-1, &volume.bounds, sizeof(uint8_t), cudaMemcpyHostToDevice);
        cudaMemcpy((char*)accel.buffer.handle+128-8-24, &aabb, sizeof(sutil::Aabb), cudaMemcpyHostToDevice);

        cudaMemcpy((char*)accel.buffer.handle+128-40, &octree_handle, sizeof(uint64_t), cudaMemcpyHostToDevice);
        cudaMemcpy((char*)accel.buffer.handle+128-48, &baked_density_handle, sizeof(uint64_t), cudaMemcpyHostToDevice);
        cudaMemcpy((char*)accel.buffer.handle+128-52, &octreeBuildDepth, sizeof(uint32_t), cudaMemcpyHostToDevice);
        if (octree_handle != 0) {
            printf("d_octree = %llu (%s octreeBuildDepth=%u)\n", octree_handle, use_gpu_octree ? "gpu" : "cpu", octreeBuildDepth);
        }
        return;
    }
}

void cleanupVolumeAccel( VolumeAccel& accel )
{
    if (accel.buffer != 0) {

        accel.buffer.reset();
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

    auto baseGrid = densityGridForVolume(volume);
    if (!baseGrid) {
        return;
    }
    const nanovdb::Map& map = baseGrid->nanoMAP();

	transform[0] = map.mMatF[0]; transform[1] = map.mMatF[1]; transform[2]  = map.mMatF[2]; transform[3]  = map.mVecF[0];
	transform[4] = map.mMatF[3]; transform[5] = map.mMatF[4]; transform[6]  = map.mMatF[5]; transform[7]  = map.mVecF[1];
	transform[8] = map.mMatF[6]; transform[9] = map.mMatF[7]; transform[10] = map.mMatF[8]; transform[11] = map.mVecF[2];
}

sutil::Aabb worldAabb( const VolumeWrapper& volume )
{
    auto baseGrid = densityGridForVolume(volume);
    if (!baseGrid) {
        return {};
    }
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