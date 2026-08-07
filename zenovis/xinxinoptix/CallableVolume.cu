#include <nanovdb/NanoVDB.h>
#include <nanovdb/util/Ray.h>
#include <nanovdb/util/HDDA.h>
#include <nanovdb/util/SampleFromVoxels.h>

#ifdef __VDB_DENSITY_BAKE__

#include <cuda_fp16.h>
#include <cuda/random.h>

#include "Octree.h"
#include "volume.h"
#include "zxxglslvec.h"
#include "math_constants.h"
#include "optixPathTracer.h"

extern "C" __constant__ Params params;

#else

#include "volume.h"
#include "TraceStuff.h"

#endif

#include "IOMat.h"
#include "zxxglslvec.h"
#include "math_constants.h"

// #include <cuda_fp16.h>
// #include "nvfunctional"

static __forceinline__ __device__ float sanitizeVolumeDensity(float value)
{
    return fmaxf(value, 0.0f);
    // return fminf(value, 65504.0f);
}

enum struct VolumeEmissionScaleType {
    Raw, Density, Absorption
};

#ifndef __FORWARD__
using DataTypeNVDB0 = nanovdb::Float;
using GridTypeNVDB0 = nanovdb::NanoGrid<DataTypeNVDB0>;
#define VolumeEmissionScale VolumeEmissionScaleType::Raw
#else
//COMMON_CODE
#endif

inline __device__ float _LERP_(float t, float s1, float s2)
{
    //return (1 - t) * s1 + t * s2;
    return fma(t, s2, fma(-t, s1, s1));
}

struct VolumeInX : VolumeIn {

	inline float rndf() const {
		return rnd(*seed);
	}

    __device__ vec3 localPosLazy() const {
#ifdef __VDB_DENSITY_BAKE__
        // A bake invocation already supplies the exact GAS-local/index
        // lattice coordinate. Do not reconstruct it through any transform.
        return pos_view;
#else
        return transformPoint(pos_view, this->worldToObject);
#endif
    };

    __device__ vec3 uniformPosLazy() const {

        using GridTypeNVDB = GridTypeNVDB0;
        const HitGroupData* sbt_data = (HitGroupData*)( sbt_ptr );

        const auto grid_ptr = sbt_data->vdb_grids[0];
        const auto* _grid = reinterpret_cast<const GridTypeNVDB*>(grid_ptr);

        auto local_pos = localPosLazy();
        if (_grid == nullptr) {
            return local_pos + 0.5f;
        }

        auto bbox = _grid->indexBBox();

        nanovdb::Coord boundsMin( bbox.min() );
        nanovdb::Coord boundsMax( bbox.max() + nanovdb::Coord( 1 ) ); // extend by one unit

        vec3 min = { 
            static_cast<float>( boundsMin[0] ), 
            static_cast<float>( boundsMin[1] ), 
            static_cast<float>( boundsMin[2] )};
        vec3 max = {
            static_cast<float>( boundsMax[0] ),
            static_cast<float>( boundsMax[1] ),
            static_cast<float>( boundsMax[2] )};

        auto _uniform_pos_ = (local_pos - min) / (max - min);
        _uniform_pos_ = clamp(_uniform_pos_, vec3(0.0f), vec3(1.0f));
        // assert(_uniform_pos_.x >= 0);
        // assert(_uniform_pos_.y >= 0);
        // assert(_uniform_pos_.z >= 0);
        return _uniform_pos_;
    };
};

template <typename Acc, uint8_t Order, typename DataTypeNVDB, typename ReturnType>
inline __device__ ReturnType nanoSampling(Acc& acc, nanovdb::Vec3f& point_indexd, const VolumeInX& volin) {
    
    using GridTypeNVDB = nanovdb::NanoGrid<DataTypeNVDB>;

#ifdef __VDB_DENSITY_BAKE__
    const int3 coord = make_int3(
        __float2int_rn(point_indexd[0]),
        __float2int_rn(point_indexd[1]),
        __float2int_rn(point_indexd[2]));
    return acc.getValue(reinterpret_cast<const nanovdb::Coord&>(coord));
#else
    if constexpr(0 == Order) {
        using Sampler = nanovdb::SampleFromVoxels<typename GridTypeNVDB::AccessorType, 0, false>;
        return Sampler(acc)(point_indexd);
    }

    if constexpr(1 == Order) {
        auto iii = interp_trilinear_stochastic(reinterpret_cast<float3&>(point_indexd), volin.rndf());
        return acc.getValue(reinterpret_cast<nanovdb::Coord&>(iii));
    }

    if constexpr(2 == Order) {
        auto fff = reinterpret_cast<float3&>(point_indexd);
        fff += make_float3(0.5f);
        fff = interp_triquadratic_to_trilinear_stochastic(fff, volin.rndf());
        using Sampler = nanovdb::SampleFromVoxels<typename GridTypeNVDB::AccessorType, 1, false>;
        const nanovdb::Vec3f trilinearPoint(fff.x, fff.y, fff.z);
        return Sampler(acc)(trilinearPoint);
    }

    if constexpr(3 == Order) {
        auto fff = reinterpret_cast<float3&>(point_indexd);
        fff = interp_tricubic_to_trilinear_stochastic(fff, volin.rndf());
        using Sampler = nanovdb::SampleFromVoxels<typename GridTypeNVDB::AccessorType, 1, false>;
        const nanovdb::Vec3f trilinearPoint(fff.x, fff.y, fff.z);
        return Sampler(acc)(trilinearPoint);
    }
    
    if constexpr(4 == Order) {
        auto uuu = nanovdb::Vec3f(volin.rndf(), volin.rndf(), volin.rndf());
        auto pick = nanovdb::RoundDown<nanovdb::Vec3f>(point_indexd + uuu);
        auto coord = nanovdb::Coord(pick[0], pick[1], pick[2]);
        return acc.getValue(coord);
    }
#endif

    return ReturnType{};
}

template <uint8_t Order, bool WorldSpace, bool cihou, typename DataTypeNVDB, typename ReturnType>
__inline__ __device__ ReturnType samplingVDB(const unsigned long long grid_ptr, const vec3& att_pos, const VolumeInX& volin) {
    using GridTypeNVDB = nanovdb::NanoGrid<DataTypeNVDB>;

    const auto* _grid = reinterpret_cast<const GridTypeNVDB*>(grid_ptr);
    if (_grid == nullptr) { return {}; }
    const auto& _acc = _grid->tree().getAccessor();

    auto pos_indexed = reinterpret_cast<const nanovdb::Vec3f&>(att_pos);

    if constexpr(WorldSpace) 
    {
        if constexpr(cihou) {
            pos_indexed = volin.localPosLazy();
        } else {
            pos_indexed = _grid->worldToIndexF(pos_indexed);
        }
    } //_grid->tree().root().maximum();

    return nanoSampling<decltype(_acc), Order, DataTypeNVDB, ReturnType>(_acc, pos_indexed, volin);
}

template <uint8_t Order, bool WorldSpace, bool cihou, typename DataTypeNVDB>
__inline__ __device__ vec2 XsamplingVDB(const unsigned long long grid_ptr, const vec3& att_pos, const VolumeInX& volin) {
    using GridTypeNVDB = nanovdb::NanoGrid<DataTypeNVDB>;

    const auto* _grid = reinterpret_cast<const GridTypeNVDB*>(grid_ptr);
    if (_grid == nullptr) { return {}; }

    float value = samplingVDB<Order, WorldSpace, cihou, DataTypeNVDB, float>(grid_ptr, att_pos, volin);
    float maxi  = _grid->tree().root().maximum();
    return vec2 { value, maxi };
}

template <bool DENSITY>
__device__ void evalVolumeMaterialCore(VolumeInX& attrs, bool shadowRay, VolumeOut& output) {

    let uniforms = params.d_uniforms;
    let buffers = params.global_buffers;

    auto& prd = attrs;

#ifdef __VDB_DENSITY_BAKE__
    // Keep the material position in the baker's local/index coordinate system.
    // This is an exact integer-valued float3 for every lattice invocation.
    vec3 att_pos = attrs.pos_view;
#else
    vec3 att_pos = attrs.pos_view + params.cam.eye;
#endif
    auto att_clr = vec3(0);
    auto att_uv = vec3(0);
    auto att_nrm = vec3(0);
    auto att_tang = vec3(0);
	
    HitGroupData* sbt_data = reinterpret_cast<HitGroupData*>(attrs.sbt_ptr);
    auto zenotex = sbt_data->textures;
    auto vdb_grids = sbt_data->vdb_grids;
    auto vdb_max_v = sbt_data->vdb_max_v;

    auto att_isBackFace = false;
    auto att_isShadowRay = shadowRay;
    float albedoAmp = 1.0f;
#ifdef __FORWARD__
    //GENERATED_BEGIN_MARK
    
    //GENERATED_END_MARK
#else
	auto anisotropy = 0.0f;
    auto density = 0.1f;

	vec3 tmp = { 1, 0, 1 };

    vec3 emission = tmp / 50.f;
    vec3 albedo = tmp;
    auto extinction = vec3(1.0f);

#endif // _FALLBACK_

    density = sanitizeVolumeDensity(density);

    output.albedo = clamp(albedo, 0.0f, 1.0f);
    output.anisotropy = __half( clamp(anisotropy, -1.0f, 1.0f) );
    output.extinction = extinction;
    output.albedoAmp = albedoAmp;
    
    output.emission = fmaxf(emission, vec3(0.0f));

	if constexpr(VolumeEmissionScale == VolumeEmissionScaleType::Raw) {
		//output.emission = output.emission; 
	} else if constexpr(VolumeEmissionScale == VolumeEmissionScaleType::Density) {
		output.emission = density * output.emission;
	} else if constexpr(VolumeEmissionScale == VolumeEmissionScaleType::Absorption) {
        output.emission = density * output.emission;
	}

    if constexpr(DENSITY) {
        output.density = __half(density);
    }
}

template <bool DENSITY>
__device__ void __proxy_callable__evalmat(void* attrs_ptr, bool shadowRay, VolumeOut& output) {
    auto& attrs = *reinterpret_cast<VolumeInX*>(attrs_ptr);
    evalVolumeMaterialCore<DENSITY>(attrs, shadowRay, output);
}

#ifdef __VDB_DENSITY_BAKE__
static __forceinline__ __device__ unsigned int densityBakeFloatToOrderedUInt(float value)
{
    value = sanitizeVolumeDensity(value);
    return static_cast<unsigned int>(__float_as_uint(value));
}

static __forceinline__ __device__ uint32_t bakedSparseOctreeLevelStart(uint8_t level)
{
    return ((1u << (3u * level)) - 1u) / 7u;
}

static __forceinline__ __device__ uint32_t bakedSparseOctreeLevelNodeCount(uint8_t level)
{
    return 1u << (3u * level);
}

static __forceinline__ __device__ uint32_t bakedSparseOctreeLeafRes(uint8_t depth)
{
    return 1u << depth;
}

static __forceinline__ __device__ uint32_t bakedSparseDenseIndex(uint32_t x, uint32_t y, uint32_t z, uint32_t res)
{
    return (z * res + y) * res + x;
}

static __forceinline__ __device__ uint3 bakedSparseDecodeDenseIndex(uint32_t idx, uint32_t res)
{
    const uint32_t x = idx % res;
    idx /= res;
    const uint32_t y = idx % res;
    const uint32_t z = idx / res;
    return make_uint3(x, y, z);
}

static __forceinline__ __device__ uint32_t bakedSparseDenseChildIndex(uint32_t parentIndex, uint8_t slot, uint32_t parentRes)
{
    const uint3 parent = bakedSparseDecodeDenseIndex(parentIndex, parentRes);
    const uint32_t childRes = parentRes << 1u;
    return bakedSparseDenseIndex(
        parent.x * 2u + uint32_t((slot >> 0u) & 1u),
        parent.y * 2u + uint32_t((slot >> 1u) & 1u),
        parent.z * 2u + uint32_t((slot >> 2u) & 1u),
        childRes);
}

static __forceinline__ __device__ uint32_t bakedSparseBrickTableIndex(int3 brickCoord, int3 brickDim)
{
    return (uint32_t(brickCoord.z) * uint32_t(brickDim.y) + uint32_t(brickCoord.y)) * uint32_t(brickDim.x) + uint32_t(brickCoord.x);
}

static __forceinline__ __device__ bool bakedSparseInsideBrickDomain(int3 brickCoord, int3 brickDim)
{
    return brickCoord.x >= 0 && brickCoord.y >= 0 && brickCoord.z >= 0
        && brickCoord.x < brickDim.x && brickCoord.y < brickDim.y && brickCoord.z < brickDim.z;
}

static __forceinline__ __device__ bool bakedSparseInsideSampleDomain(const nanovdb::Coord& coord, const BakedSparseVolumeDevice& volume)
{
    return coord[0] >= volume.sample_min.x && coord[1] >= volume.sample_min.y && coord[2] >= volume.sample_min.z
        && coord[0] < volume.sample_max.x && coord[1] < volume.sample_max.y && coord[2] < volume.sample_max.z;
}

static __forceinline__ __device__ void bakedSparseAtomicMinFloatBits(unsigned int* address, float value)
{
    atomicMin(address, __float_as_uint(value));
}

static __forceinline__ __device__ void bakedSparseAtomicMaxFloatBits(unsigned int* address, float value)
{
    atomicMax(address, __float_as_uint(value));
}

static constexpr float kBakedSparseLeafAverageQuantizationScale = 4096.0f;

extern "C" __global__ void initBakedSparseVolumeBuffers(
    BakedSparseVolumeDevice volume,
    unsigned int* leafMinBits,
    unsigned int* leafMaxBits,
    unsigned int* leafCoverage,
    unsigned long long* leafQuantizedSum)
{
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t stride = blockDim.x * gridDim.x;
    const bool initOctreeLeaves = leafMinBits != nullptr
        && leafMaxBits != nullptr
        && leafCoverage != nullptr
        && leafQuantizedSum != nullptr;
    const uint32_t bottomCount = initOctreeLeaves ? bakedSparseOctreeLevelNodeCount(volume.octreeBuildDepth) : 0u;
    const uint32_t maxCount = max(volume.brick_table_count, bottomCount);

    for (uint32_t i = tid; i < maxCount; i += stride) {
        if (i < volume.brick_table_count) {
            volume.brick_table[i] = -1;
        }
        if (i < bottomCount) {
            leafMinBits[i] = __float_as_uint(CUDART_INF_F);
            leafMaxBits[i] = 0u;
            leafCoverage[i] = 0u;
            leafQuantizedSum[i] = 0ull;
        }
    }
}

extern "C" __global__ void bakeDensityToSparseBricks(
    BakedSparseVolumeDevice volume,
    HitGroupData hitGroup,
    int clampNegative,
    uint32_t seedBase)
{
    if (volume.voxel_values == nullptr || volume.brick_table == nullptr ||
        volume.brick_origins == nullptr) {
        return;
    }

    for (uint32_t brickIndex = blockIdx.x; brickIndex < volume.brick_count; brickIndex += gridDim.x) {
        const int3 origin = volume.brick_origins[brickIndex];
        if (threadIdx.x == 0) {
            const int3 brickCoord {
                (origin.x - volume.voxel_min.x) / int(volume.brick_size),
                (origin.y - volume.voxel_min.y) / int(volume.brick_size),
                (origin.z - volume.voxel_min.z) / int(volume.brick_size)
            };
            if (bakedSparseInsideBrickDomain(brickCoord, volume.brick_dim)) {
                volume.brick_table[bakedSparseBrickTableIndex(brickCoord, volume.brick_dim)] = int(brickIndex);
            }
        }

        for (uint32_t offset = threadIdx.x; offset < 512u; offset += blockDim.x) {
            const nanovdb::Coord coord(
                origin.x + int(offset & 7u),
                origin.y + int((offset >> 3u) & 7u),
                origin.z + int((offset >> 6u) & 7u));
            if (!bakedSparseInsideSampleDomain(coord, volume)) {
                volume.voxel_values[brickIndex * 512u + offset] = 0u;
                continue;
            }
            uint32_t seed = seedBase;

            VolumeInX attrs = {};
            // The volume GAS local domain is the indexed VDB box. Feed the
            // material the lattice coordinate directly: no NanoVDB map and no
            // world/local round trip are part of the density-bake contract.
            attrs.pos_view = make_float3(
                static_cast<float>(coord[0]),
                static_cast<float>(coord[1]),
                static_cast<float>(coord[2]));
            attrs.seed = &seed;
            attrs.sbt_ptr = &hitGroup;
            attrs.objectToWorld[0] = make_float4(1.0f, 0.0f, 0.0f, 0.0f);
            attrs.objectToWorld[1] = make_float4(0.0f, 1.0f, 0.0f, 0.0f);
            attrs.objectToWorld[2] = make_float4(0.0f, 0.0f, 1.0f, 0.0f);
            attrs.worldToObject[0] = attrs.objectToWorld[0];
            attrs.worldToObject[1] = attrs.objectToWorld[1];
            attrs.worldToObject[2] = attrs.objectToWorld[2];

            VolumeOut output = {};
            output.density = __half(1.0f);
            evalVolumeMaterialCore<true>(attrs, false, output);

            float value = __half2float(output.density);
            if (clampNegative && value < 0.0f) {
                value = 0.0f;
            }
            value = fminf(sanitizeVolumeDensity(value), 65504.0f);

            // Round upward so zero stays exactly zero while every positive
            // shader result is enclosed by [previous_half(bits), bits].
            volume.voxel_values[brickIndex * 512u + offset] =
                __half_as_ushort(__float2half_ru(value));
        }
        __syncthreads();
    }
}

struct BakedSparseInterval
{
    float lower;
    float upper;
};

struct BakedSparseWeightInterval
{
    float lower;
    float upper;
};

static __forceinline__ __device__ 
BakedSparseInterval bakedSparseLoadDensityInterval(const BakedSparseVolumeDevice& volume, const int3& coord)
{
    const int3 rel {
        coord.x - volume.voxel_min.x,
        coord.y - volume.voxel_min.y,
        coord.z - volume.voxel_min.z
    };
    if (rel.x < 0 || rel.y < 0 || rel.z < 0 ||
        rel.x >= volume.voxel_dim.x || rel.y >= volume.voxel_dim.y || rel.z >= volume.voxel_dim.z) {
        return { 0.0f, 0.0f };
    }

    const int brickSize = int(volume.brick_size);
    const int3 brickCoord { rel.x / brickSize, rel.y / brickSize, rel.z / brickSize };
    if (!bakedSparseInsideBrickDomain(brickCoord, volume.brick_dim)) {
        return { 0.0f, 0.0f };
    }
    const int brickIndex = volume.brick_table[bakedSparseBrickTableIndex(brickCoord, volume.brick_dim)];
    if (brickIndex < 0 || uint32_t(brickIndex) >= volume.brick_count) {
        return { 0.0f, 0.0f };
    }

    const int lx = rel.x - brickCoord.x * brickSize;
    const int ly = rel.y - brickCoord.y * brickSize;
    const int lz = rel.z - brickCoord.z * brickSize;
    const uint32_t offset = uint32_t((lz * brickSize + ly) * brickSize + lx);
    const uint16_t upperBits = volume.voxel_values[uint32_t(brickIndex) * 512u + offset];
    if (upperBits == 0u) {
        return { 0.0f, 0.0f };
    }

    // Lattice values were rounded toward +infinity. For finite non-negative
    // binary16, adjacent bit patterns are adjacent representable values.
    const uint16_t lowerBits = upperBits - 1u;
    return {
        __half2float(__ushort_as_half(lowerBits)),
        __half2float(__ushort_as_half(upperBits))
    };
}

static __forceinline__ __device__ int3 bakedSparseCellCoord(const BakedSparseVolumeDevice& volume, uint64_t cellIndex)
{
    const uint32_t brickIndex = uint32_t(cellIndex >> 9u);
    const uint32_t offset = uint32_t(cellIndex & 511ull);
    const int3 brickOrigin = volume.brick_origins[brickIndex];
    return {
        brickOrigin.x + int(offset & 7u),
        brickOrigin.y + int((offset >> 3u) & 7u),
        brickOrigin.z + int((offset >> 6u) & 7u)
    };
}

static __forceinline__ __device__ bool bakedSparseValidCell(const BakedSparseVolumeDevice& volume, const int3& cellCoord)
{
    return cellCoord.x >= volume.sample_min.x && cellCoord.y >= volume.sample_min.y && cellCoord.z >= volume.sample_min.z &&
        cellCoord.x < volume.sample_max.x && cellCoord.y < volume.sample_max.y && cellCoord.z < volume.sample_max.z;
}

static __forceinline__ __device__ BakedSparseInterval bakedSparseWeightedPair(
    const BakedSparseInterval& first,
    const BakedSparseInterval& second,
    const BakedSparseWeightInterval& firstWeight,
    const BakedSparseWeightInterval& secondWeight)
{
    return {
        __fadd_rd(
            __fmul_rd(firstWeight.lower, first.lower),
            __fmul_rd(secondWeight.lower, second.lower)),
        __fadd_ru(
            __fmul_ru(firstWeight.upper, first.upper),
            __fmul_ru(secondWeight.upper, second.upper))
    };
}

static __forceinline__ __device__ BakedSparseWeightInterval bakedSparseWeightOneSeventh()
{
    // Precomputed downward/upward IEEE-754 neighbors. These are bit-identical
    // to __fdiv_rd/__fdiv_ru, without repeating divisions for every cell.
    return { __uint_as_float(0x3e124924u), __uint_as_float(0x3e124925u) };
}

static __forceinline__ __device__ BakedSparseWeightInterval bakedSparseWeightSixSevenths()
{
    return { __uint_as_float(0x3f5b6db6u), __uint_as_float(0x3f5b6db7u) };
}

static __forceinline__ __device__ BakedSparseWeightInterval bakedSparseWeightOneFifth()
{
    return { __uint_as_float(0x3e4cccccu), __uint_as_float(0x3e4ccccdu) };
}

static __forceinline__ __device__ BakedSparseWeightInterval bakedSparseWeightFourFifths()
{
    return { __uint_as_float(0x3f4cccccu), __uint_as_float(0x3f4ccccdu) };
}

template <uint8_t Order>
static __forceinline__ __device__ 
BakedSparseInterval bakedSparseFilterCoefficient(const BakedSparseInterval* input, int stride, int coefficient)
{
    if constexpr (Order == 2u) {
        switch (coefficient) {
        case 0:
            return bakedSparseWeightedPair(
                input[0], input[stride],
                bakedSparseWeightOneSeventh(), bakedSparseWeightSixSevenths());
        case 1:
            return input[stride];
        case 2:
            return input[stride * 2];
        default:
            return input[stride * 3];
        }
    } else {
        switch (coefficient) {
        case 0:
            return bakedSparseWeightedPair(
                input[0], input[stride],
                bakedSparseWeightOneFifth(), bakedSparseWeightFourFifths());
        case 1:
            return input[stride];
        case 2:
            return input[stride * 2];
        default:
            return bakedSparseWeightedPair(
                input[stride * 2], input[stride * 3],
                bakedSparseWeightFourFifths(), bakedSparseWeightOneFifth());
        }
    }
}

static __forceinline__ __device__ 
float bakedSparseStoreCellInterval(BakedSparseVolumeDevice& volume, uint64_t cellIndex, float cellLower, float cellUpper)
{
    cellLower = fmaxf(cellLower, 0.0f);
    cellUpper = fminf(fmaxf(cellUpper, 0.0f), 65504.0f);
    volume.cell_min[cellIndex] = __half_as_ushort(__float2half_rd(cellLower));
    volume.cell_max[cellIndex] = __half_as_ushort(__float2half_ru(cellUpper));
    return cellUpper;
}

extern "C" __global__ void bakeBakedSparseVolumeCellBoundsLinear(BakedSparseVolumeDevice volume, unsigned int* maxDensityBits)
{
    if (volume.voxel_values == nullptr || volume.cell_min == nullptr || volume.cell_max == nullptr) {
        return;
    }

    const uint64_t totalCellCount = uint64_t(volume.brick_count) * 512ull;
    const uint64_t firstCell = uint64_t(blockIdx.x) * uint64_t(blockDim.x) + uint64_t(threadIdx.x);
    const uint64_t cellStride = uint64_t(gridDim.x) * uint64_t(blockDim.x);
    float threadMaximum = 0.0f;

    for (uint64_t cellIndex = firstCell; cellIndex < totalCellCount; cellIndex += cellStride) {
        const int3 cellCoord = bakedSparseCellCoord(volume, cellIndex);
        if (!bakedSparseValidCell(volume, cellCoord)) {
            volume.cell_min[cellIndex] = 0u;
            volume.cell_max[cellIndex] = 0u;
            continue;
        }

        float cellLower = CUDART_INF_F;
        float cellUpper = 0.0f;
#pragma unroll
        for (int z = 0; z < 2; ++z) {
#pragma unroll
            for (int y = 0; y < 2; ++y) {
#pragma unroll
                for (int x = 0; x < 2; ++x) {
                    const auto interval = bakedSparseLoadDensityInterval(
                        volume, make_int3(cellCoord.x + x, cellCoord.y + y, cellCoord.z + z));
                    cellLower = fminf(cellLower, interval.lower);
                    cellUpper = fmaxf(cellUpper, interval.upper);
                }
            }
        }

        cellUpper = bakedSparseStoreCellInterval(volume, cellIndex, cellLower, cellUpper);
        threadMaximum = fmaxf(threadMaximum, cellUpper);
    }

    __shared__ float blockMaximum[256];
    blockMaximum[threadIdx.x] = threadMaximum;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1u; stride != 0u; stride >>= 1u) {
        if (threadIdx.x < stride) {
            blockMaximum[threadIdx.x] = fmaxf(blockMaximum[threadIdx.x], blockMaximum[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0u && maxDensityBits != nullptr && blockMaximum[0] > 0.0f) {
        atomicMax(maxDensityBits, densityBakeFloatToOrderedUInt(blockMaximum[0]));
    }
}

template <uint8_t Order>
static __forceinline__ __device__ void bakeBakedSparseVolumeCellBoundsFiltered(
    BakedSparseVolumeDevice volume,
    unsigned int* maxDensityBits,
    BakedSparseInterval* raw,
    BakedSparseInterval* stageX,
    BakedSparseInterval* stageY,
    float* blockMaximum)
{
    constexpr uint32_t warpSizeValue = 32u;
    constexpr uint32_t warpsPerBlock = 8u;
    constexpr int sampleCount = 4;
    // The quadratic envelope originally has six rows. Its 1/2 and 1/7--6/7
    // blends of samples 1 and 2 are convex combinations of the two selector
    // rows already present. By multilinearity, tensor products containing
    // either row are convex combinations of tensor products of those selector
    // rows, so they cannot introduce a new minimum or maximum. The four rows
    // retained here therefore give the same exact extrema with 64 instead of
    // 216 three-dimensional candidates.
    constexpr int coefficientCount = 4;
    constexpr uint32_t rawCount = sampleCount * sampleCount * sampleCount;
    constexpr uint32_t stageXCount = coefficientCount * sampleCount * sampleCount;
    constexpr uint32_t stageYCount = coefficientCount * coefficientCount * sampleCount;
    constexpr uint32_t coefficientTotal = coefficientCount * coefficientCount * coefficientCount;

    if (volume.voxel_values == nullptr || volume.cell_min == nullptr || volume.cell_max == nullptr) {
        return;
    }

    const uint32_t lane = threadIdx.x & (warpSizeValue - 1u);
    const uint32_t warp = threadIdx.x / warpSizeValue;
    const uint64_t firstCell = uint64_t(blockIdx.x) * warpsPerBlock + warp;
    const uint64_t cellStride = uint64_t(gridDim.x) * warpsPerBlock;
    const uint64_t totalCellCount = uint64_t(volume.brick_count) * 512ull;
    BakedSparseInterval* warpRaw = raw + warp * rawCount;
    BakedSparseInterval* warpStageX = stageX + warp * stageXCount;
    BakedSparseInterval* warpStageY = stageY + warp * stageYCount;
    float warpMaximum = 0.0f;

    for (uint64_t cellIndex = firstCell; cellIndex < totalCellCount; cellIndex += cellStride) {
        const int3 cellCoord = bakedSparseCellCoord(volume, cellIndex);
        if (!bakedSparseValidCell(volume, cellCoord)) {
            if (lane == 0u) {
                volume.cell_min[cellIndex] = 0u;
                volume.cell_max[cellIndex] = 0u;
            }
            continue;
        }

        for (uint32_t i = lane; i < rawCount; i += warpSizeValue) {
            uint32_t index = i;
            const int sx = int(index & 3u); index >>= 2u;
            const int sy = int(index & 3u);
            const int sz = int(index >> 2u);
            warpRaw[i] = bakedSparseLoadDensityInterval(
                volume,
                make_int3(cellCoord.x + sx - 1, cellCoord.y + sy - 1, cellCoord.z + sz - 1));
        }
        __syncwarp();

        for (uint32_t i = lane; i < stageXCount; i += warpSizeValue) {
            uint32_t index = i;
            const int coefficientX = int(index % coefficientCount); index /= coefficientCount;
            const int sampleY = int(index & 3u);
            const int sampleZ = int(index >> 2u);
            const auto* input = warpRaw + (sampleZ * sampleCount + sampleY) * sampleCount;
            warpStageX[i] = bakedSparseFilterCoefficient<Order>(input, 1, coefficientX);
        }
        __syncwarp();

        for (uint32_t i = lane; i < stageYCount; i += warpSizeValue) {
            uint32_t index = i;
            const int coefficientX = int(index % coefficientCount); index /= coefficientCount;
            const int coefficientY = int(index % coefficientCount);
            const int sampleZ = int(index / coefficientCount);
            const auto* input = warpStageX + sampleZ * sampleCount * coefficientCount + coefficientX;
            warpStageY[i] = bakedSparseFilterCoefficient<Order>(input, coefficientCount, coefficientY);
        }
        __syncwarp();

        float cellLower = CUDART_INF_F;
        float cellUpper = 0.0f;
        for (uint32_t i = lane; i < coefficientTotal; i += warpSizeValue) {
            uint32_t index = i;
            const int coefficientX = int(index % coefficientCount); index /= coefficientCount;
            const int coefficientY = int(index % coefficientCount);
            const int coefficientZ = int(index / coefficientCount);
            const auto* input = warpStageY + coefficientY * coefficientCount + coefficientX;
            const auto interval = bakedSparseFilterCoefficient<Order>(
                input, coefficientCount * coefficientCount, coefficientZ);
            cellLower = fminf(cellLower, interval.lower);
            cellUpper = fmaxf(cellUpper, interval.upper);
        }

        for (uint32_t delta = 16u; delta != 0u; delta >>= 1u) {
            cellLower = fminf(cellLower, __shfl_down_sync(0xFFFFFFFFu, cellLower, delta));
            cellUpper = fmaxf(cellUpper, __shfl_down_sync(0xFFFFFFFFu, cellUpper, delta));
        }
        if (lane == 0u) {
            cellUpper = bakedSparseStoreCellInterval(volume, cellIndex, cellLower, cellUpper);
            warpMaximum = fmaxf(warpMaximum, cellUpper);
        }
        __syncwarp();
    }

    if (lane == 0u) {
        blockMaximum[warp] = warpMaximum;
    }
    __syncthreads();
    if (warp == 0u) {
        float maximum = lane < warpsPerBlock ? blockMaximum[lane] : 0.0f;
        for (uint32_t delta = 16u; delta != 0u; delta >>= 1u) {
            maximum = fmaxf(maximum, __shfl_down_sync(0xFFFFFFFFu, maximum, delta));
        }
        if (lane == 0u && maxDensityBits != nullptr && maximum > 0.0f) {
            atomicMax(maxDensityBits, densityBakeFloatToOrderedUInt(maximum));
        }
    }
}

extern "C" __global__ void bakeBakedSparseVolumeCellBoundsQuadratic(BakedSparseVolumeDevice volume, unsigned int* maxDensityBits)
{
    constexpr uint32_t warpsPerBlock = 8u;
    __shared__ BakedSparseInterval raw[warpsPerBlock * 64u];
    __shared__ BakedSparseInterval stageX[warpsPerBlock * 64u];
    __shared__ BakedSparseInterval stageY[warpsPerBlock * 64u];
    __shared__ float blockMaximum[warpsPerBlock];
    bakeBakedSparseVolumeCellBoundsFiltered<2u>(
        volume, maxDensityBits, raw, stageX, stageY, blockMaximum);
}

extern "C" __global__ void bakeBakedSparseVolumeCellBoundsCubic(BakedSparseVolumeDevice volume, unsigned int* maxDensityBits)
{
    constexpr uint32_t warpsPerBlock = 8u;
    __shared__ BakedSparseInterval raw[warpsPerBlock * 64u];
    __shared__ BakedSparseInterval stageX[warpsPerBlock * 64u];
    __shared__ BakedSparseInterval stageY[warpsPerBlock * 64u];
    __shared__ float blockMaximum[warpsPerBlock];
    bakeBakedSparseVolumeCellBoundsFiltered<3u>(
        volume, maxDensityBits, raw, stageX, stageY, blockMaximum);
}

extern "C" __global__ void accumulateBakedSparseVolumeOctreeLeaves(
    BakedSparseVolumeDevice volume,
    unsigned int* leafMinBits,
    unsigned int* leafMaxBits,
    unsigned int* leafCoverage,
    unsigned long long* leafQuantizedSum)
{
    const int3 leafSize {
        max(volume.voxel_dim.x >> volume.octreeBuildDepth, 1),
        max(volume.voxel_dim.y >> volume.octreeBuildDepth, 1),
        max(volume.voxel_dim.z >> volume.octreeBuildDepth, 1)
    };
    const uint32_t leafRes = bakedSparseOctreeLeafRes(volume.octreeBuildDepth);

    for (uint32_t brickIndex = blockIdx.x; brickIndex < volume.brick_count; brickIndex += gridDim.x) {
        const int3 brickOrigin = volume.brick_origins[brickIndex];
        for (uint32_t offset = threadIdx.x; offset < 512u; offset += blockDim.x) {
            const int cx = int(offset & 7u);
            const int cy = int((offset >> 3u) & 7u);
            const int cz = int((offset >> 6u) & 7u);
            const int3 cellCoord { brickOrigin.x + cx, brickOrigin.y + cy, brickOrigin.z + cz };
            if (cellCoord.x < volume.sample_min.x || cellCoord.y < volume.sample_min.y || cellCoord.z < volume.sample_min.z ||
                cellCoord.x >= volume.sample_max.x || cellCoord.y >= volume.sample_max.y || cellCoord.z >= volume.sample_max.z) {
                continue;
            }

            const uint64_t cellIndex = uint64_t(brickIndex) * 512ull + uint64_t(offset);
            const float cellMin = __half2float(__ushort_as_half(volume.cell_min[cellIndex]));
            const float cellMax = __half2float(__ushort_as_half(volume.cell_max[cellIndex]));
            if (!(cellMax > 0.0f)) {
                continue;
            }

            const int3 rel {
                cellCoord.x - volume.voxel_min.x,
                cellCoord.y - volume.voxel_min.y,
                cellCoord.z - volume.voxel_min.z
            };
            const uint32_t leafX = min(uint32_t(rel.x / leafSize.x), leafRes - 1u);
            const uint32_t leafY = min(uint32_t(rel.y / leafSize.y), leafRes - 1u);
            const uint32_t leafZ = min(uint32_t(rel.z / leafSize.z), leafRes - 1u);
            const uint32_t leafIndex = bakedSparseDenseIndex(leafX, leafY, leafZ, leafRes);
            bakedSparseAtomicMinFloatBits(leafMinBits + leafIndex, cellMin);
            bakedSparseAtomicMaxFloatBits(leafMaxBits + leafIndex, cellMax);
            atomicAdd(leafCoverage + leafIndex, 1u);
            const auto quantizedAverage = static_cast<unsigned long long>(
                0.5f * (cellMin + cellMax) * kBakedSparseLeafAverageQuantizationScale + 0.5f);
            atomicAdd(leafQuantizedSum + leafIndex, quantizedAverage);
        }
    }
}

extern "C" __global__ void reduceBakedSparseVolumeOctreeLevel(
    BakedSparseVolumeDevice volume,
    unsigned int* leafMinBits,
    unsigned int* leafMaxBits,
    unsigned int* leafCoverage,
    unsigned long long* leafQuantizedSum,
    uint8_t level)
{
    const uint32_t nodeCount = bakedSparseOctreeLevelNodeCount(level);
    const bool preserveAverageForOffsetFallback =
        uint64_t(nodeCount) * 8ull - 7ull > uint64_t(OCTREE_MAX_RELATIVE_OFFSET);
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nodeCount || volume.octree == nullptr) {
        return;
    }

    const uint32_t levelStart = bakedSparseOctreeLevelStart(level);
    OcNode node {};

    if (level == volume.octreeBuildDepth) {
        const uint32_t coverage = leafCoverage[tid];
        const float maxValue = __uint_as_float(leafMaxBits[tid]);
        if (coverage != 0u && maxValue > 0.0f) {
            const int3 leafSize {
                max(volume.voxel_dim.x >> volume.octreeBuildDepth, 1),
                max(volume.voxel_dim.y >> volume.octreeBuildDepth, 1),
                max(volume.voxel_dim.z >> volume.octreeBuildDepth, 1)
            };
            const uint32_t cellCount = uint32_t(leafSize.x * leafSize.y * leafSize.z);
            const float minValue = coverage < cellCount ? 0.0f : __uint_as_float(leafMinBits[tid]);
            const float avgValue = float(double(leafQuantizedSum[tid]) / (double(kBakedSparseLeafAverageQuantizationScale) * double(cellCount)));
            // Tracking requires conservative bounds after FP16 storage.
            node.min_d = __float2half_rd(minValue);
            node.max_d = __float2half_ru(maxValue);
            auto avg_d = __float2half(avgValue);
            node.setLeafAverageBits(*(ushort*)&avg_d);
        }
        volume.octree[levelStart + tid] = node;
        return;
    }

    const uint32_t parentRes = 1u << level;
    const uint32_t childLevelStart = bakedSparseOctreeLevelStart(level + 1u);
    float minValue = CUDART_INF_F;
    float maxValue = 0.0f;
    float averageValue = 0.0f;
    uint8_t childMask = 0u;
    for (uint8_t slot = 0; slot < 8u; ++slot) {
        const uint32_t childIndex = bakedSparseDenseChildIndex(tid, slot, parentRes);
        const OcNode& child = volume.octree[childLevelStart + childIndex];
        if (preserveAverageForOffsetFallback) {
            averageValue += __half2float(__ushort_as_half(child.leafAverageBits())) * 0.125f;
        }
        const float childMax = __half2float(child.max_d);
        if (!(childMax > 0.0f)) {
            continue;
        }
        childMask |= uint8_t(1u << slot);
        minValue = fminf(minValue, __half2float(child.min_d));
        maxValue = fmaxf(maxValue, childMax);
    }

    if (childMask != 0u) {
        // Missing children are empty spatial octants. Include their zero in
        // the parent bound so a sparse node cannot collapse into a solid leaf.
        if (childMask != 0xFFu) {
            minValue = 0.0f;
        }
        const __half minHalf = __float2half_rd(minValue);
        const __half maxHalf = __float2half_ru(maxValue);
        node.min_d = minHalf;
        node.max_d = maxHalf;
        if (preserveAverageForOffsetFallback) {
            const __half averageHalf = __float2half(averageValue);
            node.setLeafAverageBits(*(ushort*)&averageHalf);
        }
        if (minHalf == maxHalf) {
            node.setLeafAverageBits(*(ushort*)&maxHalf);
        } else {
            node.setChildMask(childMask);
        }
    }
    volume.octree[levelStart + tid] = node;
}

extern "C" __global__ void countBakedSparseVolumeCompactChildren(
    const OcNode* denseOctree,
    const uint32_t* parentDenseIndices,
    uint32_t parentCount,
    uint8_t level,
    uint8_t octreeDepth,
    uint32_t* childCounts)
{
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= parentCount || level >= octreeDepth) {
        return;
    }

    const uint32_t parentDenseIndex = parentDenseIndices[tid];
    const OcNode& parent = denseOctree[bakedSparseOctreeLevelStart(level) + parentDenseIndex];
    childCounts[tid] = uint32_t(__popc(uint32_t(parent.childMask())));
}

extern "C" __global__ void prefixBakedSparseVolumeCompactChildren(
    uint32_t* childCounts,
    uint32_t* childOffsets,
    uint32_t parentCount,
    uint32_t* levelCounts,
    uint8_t level,
    uint8_t octreeDepth)
{
    if (blockIdx.x != 0u || threadIdx.x != 0u || level >= octreeDepth) {
        return;
    }

    uint32_t total = 0u;
    for (uint32_t i = 0u; i < parentCount; ++i) {
        childOffsets[i] = total;
        const uint32_t childCount = childCounts[i];
        if (childCount == 0u) {
            continue;
        }
        const uint32_t childRelativeOffset = parentCount + total - i;
        if (childRelativeOffset > OCTREE_MAX_RELATIVE_OFFSET) {
            childCounts[i] = 0u;
            continue;
        }
        total += childCount;
    }
    levelCounts[level + 1u] = total;
}

extern "C" __global__ void emitBakedSparseVolumeCompactLevel(
    const OcNode* denseOctree,
    OcNode* compactOctree,
    const uint32_t* parentDenseIndices,
    uint32_t* childDenseIndices,
    const uint32_t* childOffsets,
    const uint32_t* childCounts,
    uint32_t parentCount,
    uint8_t level,
    uint8_t octreeDepth,
    uint32_t compactLevelStart,
    uint32_t compactNextLevelStart)
{
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= parentCount || level > octreeDepth) {
        return;
    }

    const uint32_t parentDenseIndex = parentDenseIndices[tid];
    OcNode node = denseOctree[bakedSparseOctreeLevelStart(level) + parentDenseIndex];

    if (level < octreeDepth && childCounts[tid] != 0u) {
        const uint8_t childMask = node.childMask();
        const uint32_t childStart = childOffsets[tid];
        uint32_t childRank = 0u;
        for (uint8_t slot = 0u; slot < 8u; ++slot) {
            const uint8_t bit = uint8_t(1u << slot);
            if ((childMask & bit) == 0u) {
                continue;
            }
            childDenseIndices[childStart + childRank] = bakedSparseDenseChildIndex(parentDenseIndex, slot, 1u << level);
            ++childRank;
        }
        const uint32_t currentNodeIndex = compactLevelStart + tid;
        const uint32_t firstChildIndex = compactNextLevelStart + childStart;
        const uint32_t childRelativeOffset = firstChildIndex - currentNodeIndex;
        node.data = (uint32_t(childMask) << 24u) | childRelativeOffset;
    } else if (level < octreeDepth && node.childMask() != 0u) {
        node.setChildMask(0u);
    }

    compactOctree[compactLevelStart + tid] = node;
}
#else
extern "C" __device__ void __direct_callable__evalmat(void* attrs_ptr, bool shadowRay, VolumeOut& output) {
    if (output.density < __half(0))
        __proxy_callable__evalmat<false>(attrs_ptr, shadowRay, output);
    else
        __proxy_callable__evalmat<true>(attrs_ptr, shadowRay, output);
}
#endif
