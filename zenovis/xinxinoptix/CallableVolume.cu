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

inline int3 interp_trilinear_stochastic(const float3& P, float randu)
{
    const float ix = floorf(P.x);
    const float iy = floorf(P.y);
    const float iz = floorf(P.z);
    int idx[3] = {(int)ix, (int)iy, (int)iz};

    const float tx = P.x - ix;
    const float ty = P.y - iy;
    const float tz = P.z - iz;

    if (randu < tx) {
        idx[0]++;
        randu /= tx;
    }
    else {
        randu = (randu - tx) / (1 - tx);
    }

    if (randu < ty) {
        idx[1]++;
        randu /= ty;
    }
    else {
        randu = (randu - ty) / (1 - ty);
    }

    if (randu < tz) {
        idx[2]++;
    }

    return make_int3(idx[0], idx[1], idx[2]);
}

inline float3 interp_triquadratic_to_trilinear_stochastic(const float3& P, float randu)
{
    const float3 p = floor(P);
    const float3 t = P - p;

    // Corrected quadratic B-spline weights
    const float3 w_minus1 = 0.5f * (1.0f - t) * (1.0f - t);
    const float3 w_0 = 0.5f + t - t * t;
    const float3 w_plus1 = 0.5f * t * t;

    const float3 g0 = w_minus1 + w_0;

    const float3 P0 = p + (w_0 / g0) - 1.0f;
    const float3 P1 = p + 1.0f;

    float3 Pnew = P0;

    if (randu < g0.x) {
        randu /= g0.x;
    }
    else {
        Pnew.x = P1.x;
        randu = (randu - g0.x) / (1.0f - g0.x);
    }

    if (randu < g0.y) {
        randu /= g0.y;
    }
    else {
        Pnew.y = P1.y;
        randu = (randu - g0.y) / (1.0f - g0.y);
    }

    if (randu < g0.z) {
        // stay with P0.z
    }
    else {
        Pnew.z = P1.z;
    }

    return Pnew;
}

inline float3 interp_tricubic_to_trilinear_stochastic(const float3& P, float randu)
{
    const float3 p = floor(P);
    const float3 t = P - p;

    /* Cubic weights. */
    const float3 w0 = (1.0f / 6.0f) * (t * (t * (-t + 3.0f) - 3.0f) + 1.0f);
    const float3 w1 = (1.0f / 6.0f) * (t * t * (3.0f * t - 6.0f) + 4.0f);
    //    float3 w2 = (1.0f / 6.0f) * (t * (t * (-3.0f * t + 3.0f) + 3.0f) + 1.0f);
    const float3 w3 = (1.0f / 6.0f) * (t * t * t);

    const float3 g0 = w0 + w1;
    const float3 P0 = p + (w1 / g0) - 1.0f;
    const float3 P1 = p + (w3 / (make_float3(1.0f) - g0)) + 1.0f;

    float3 Pnew = P0;

    if (randu < g0.x) {
        randu /= g0.x;
    }
    else {
        Pnew.x = P1.x;
        randu = (randu - g0.x) / (1 - g0.x);
    }

    if (randu < g0.y) {
        randu /= g0.y;
    }
    else {
        Pnew.y = P1.y;
        randu = (randu - g0.y) / (1 - g0.y);
    }

    if (randu < g0.z) {
    }
    else {
        Pnew.z = P1.z;
    }

    return Pnew;
}

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
        return transformPoint(pos_view, this->worldToObject);
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
        int(point_indexd[0]),
        int(point_indexd[1]),
        int(point_indexd[2]));
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
        auto iii = interp_trilinear_stochastic(fff, volin.rndf());
        return acc.getValue(reinterpret_cast<nanovdb::Coord&>(iii));
    }

    if constexpr(3 == Order) {
        auto fff = reinterpret_cast<float3&>(point_indexd);
        fff = interp_tricubic_to_trilinear_stochastic(fff, volin.rndf());
        auto iii = interp_trilinear_stochastic(fff, volin.rndf());
        return acc.getValue(reinterpret_cast<nanovdb::Coord&>(iii));
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

    vec3 att_pos = attrs.pos_view + params.cam.eye;
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

extern "C" __global__ void bakeNanoVDBDensityToSparseBricks(
    BakedSparseVolumeDevice volume,
    HitGroupData hitGroup,
    int clampNegative,
    uint32_t seedBase,
    unsigned int* maxDensityBits)
{
    if (volume.voxel_values == nullptr || volume.brick_table == nullptr ||
        volume.brick_origins == nullptr || volume.brick_min == nullptr || volume.brick_max == nullptr) {
        return;
    }

    __shared__ float bakedValues[512];
    __shared__ float reduceMin[256];
    __shared__ float reduceMax[256];

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

        float threadMin = CUDART_INF_F;
        float threadMax = 0.0f;

        for (uint32_t offset = threadIdx.x; offset < 512u; offset += blockDim.x) {
            const nanovdb::Coord coord(
                origin.x + int(offset & 7u),
                origin.y + int((offset >> 3u) & 7u),
                origin.z + int((offset >> 6u) & 7u));
            if (!bakedSparseInsideSampleDomain(coord, volume)) {
                bakedValues[offset] = 0.0f;
                volume.voxel_values[brickIndex * 512u + offset] = 0u;
                continue;
            }
            uint32_t seed = seedBase;

            VolumeInX attrs = {};
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
            value = sanitizeVolumeDensity(value);

            bakedValues[offset] = value;
            volume.voxel_values[brickIndex * 512u + offset] = __half_as_ushort(__float2half_rn(value));
            threadMin = fminf(threadMin, value);
            threadMax = fmaxf(threadMax, value);
        }

        reduceMin[threadIdx.x] = threadMin;
        reduceMax[threadIdx.x] = threadMax;
        __syncthreads();

        for (uint32_t stride = blockDim.x >> 1u; stride > 0u; stride >>= 1u) {
            if (threadIdx.x < stride) {
                reduceMin[threadIdx.x] = fminf(reduceMin[threadIdx.x], reduceMin[threadIdx.x + stride]);
                reduceMax[threadIdx.x] = fmaxf(reduceMax[threadIdx.x], reduceMax[threadIdx.x + stride]);
            }
            __syncthreads();
        }

        if (threadIdx.x == 0u) {
            const float brickMin = reduceMax[0] > 0.0f ? reduceMin[0] : 0.0f;
            volume.brick_min[brickIndex] = __half_as_ushort(__float2half_rn(brickMin));
            volume.brick_max[brickIndex] = __half_as_ushort(__float2half_rn(reduceMax[0]));
            if (maxDensityBits != nullptr && reduceMax[0] > 0.0f) {
                atomicMax(maxDensityBits, densityBakeFloatToOrderedUInt(reduceMax[0]));
            }
        }
        __syncthreads();
    }
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
            const uint16_t bits = volume.voxel_values[brickIndex * 512u + offset];
            const float value = __half2float(__ushort_as_half(bits));
            if (!(value > 0.0f)) {
                continue;
            }

            const int lx = int(offset & 7u);
            const int ly = int((offset >> 3u) & 7u);
            const int lz = int((offset >> 6u) & 7u);
            const int3 rel {
                brickOrigin.x + lx - volume.voxel_min.x,
                brickOrigin.y + ly - volume.voxel_min.y,
                brickOrigin.z + lz - volume.voxel_min.z
            };
            if (rel.x < 0 || rel.y < 0 || rel.z < 0 ||
                rel.x >= volume.voxel_dim.x || rel.y >= volume.voxel_dim.y || rel.z >= volume.voxel_dim.z) {
                continue;
            }

            const uint32_t cx = min(uint32_t(rel.x / leafSize.x), leafRes - 1u);
            const uint32_t cy = min(uint32_t(rel.y / leafSize.y), leafRes - 1u);
            const uint32_t cz = min(uint32_t(rel.z / leafSize.z), leafRes - 1u);
            const uint32_t cellIndex = bakedSparseDenseIndex(cx, cy, cz, leafRes);
            bakedSparseAtomicMinFloatBits(leafMinBits + cellIndex, value);
            bakedSparseAtomicMaxFloatBits(leafMaxBits + cellIndex, value);
            atomicAdd(leafCoverage + cellIndex, 1u);
            const auto quantized = static_cast<unsigned long long>(value * kBakedSparseLeafAverageQuantizationScale + 0.5f);
            atomicAdd(leafQuantizedSum + cellIndex, quantized);
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
            const uint32_t cellVolume = uint32_t(leafSize.x * leafSize.y * leafSize.z);
            const float minValue = coverage < cellVolume ? 0.0f : __uint_as_float(leafMinBits[tid]);
            const float avgValue = float(double(leafQuantizedSum[tid]) / (double(kBakedSparseLeafAverageQuantizationScale) * double(cellVolume)));
            node.min_d = __float2half(minValue);
            node.max_d = __float2half(maxValue);
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
    uint8_t childMask = 0u;
    for (uint8_t slot = 0; slot < 8u; ++slot) {
        const uint32_t childIndex = bakedSparseDenseChildIndex(tid, slot, parentRes);
        const OcNode& child = volume.octree[childLevelStart + childIndex];
        const float childMax = __half2float(child.max_d);
        if (!(childMax > 0.0f)) {
            continue;
        }
        childMask |= uint8_t(1u << slot);
        minValue = fminf(minValue, __half2float(child.min_d));
        maxValue = fmaxf(maxValue, childMax);
    }

    if (childMask != 0u) {
        const __half minHalf = __float2half(minValue);
        const __half maxHalf = __float2half(maxValue);
        node.min_d = minHalf;
        node.max_d = maxHalf;
        if (minHalf == maxHalf) {
            node.setLeafAverageBits(*(ushort*)&maxHalf);
        } else {
            node.data = uint32_t(childMask) << 24u;
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

    const uint32_t parentDenseIndex = tid;
    const OcNode& parent = denseOctree[bakedSparseOctreeLevelStart(level) + parentDenseIndex];
    childCounts[tid] = uint32_t(__popc(uint32_t(parent.childMask())));
}

extern "C" __global__ void prefixBakedSparseVolumeCompactChildren(
    const uint32_t* childCounts,
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
        total += childCounts[i];
    }
    levelCounts[level + 1u] = total;
}

extern "C" __global__ void emitBakedSparseVolumeCompactLevel(
    const OcNode* denseOctree,
    OcNode* compactOctree,
    const uint32_t* parentDenseIndices,
    uint32_t* childDenseIndices,
    const uint32_t* childOffsets,
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

    if (level < octreeDepth) {
        const uint8_t childMask = node.childMask();
        const uint32_t childStart = childOffsets[parentDenseIndex];
        uint32_t childRank = 0u;
        for (uint8_t slot = 0u; slot < 8u; ++slot) {
            const uint8_t bit = uint8_t(1u << slot);
            if ((childMask & bit) == 0u) {
                continue;
            }
            childDenseIndices[childStart + childRank] = bakedSparseDenseChildIndex(parentDenseIndex, slot, 1u << level);
            ++childRank;
        }
        node.data = (uint32_t(childMask) << 24u) | ((compactNextLevelStart + childStart) & 0x00FFFFFFu);
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
