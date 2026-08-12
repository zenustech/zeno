#include <nanovdb/NanoVDB.h>
#include <nanovdb/util/Ray.h>
#include <nanovdb/util/HDDA.h>
#include <nanovdb/util/SampleFromVoxels.h>

#include <cuda_fp16.h>
#include <cuda/random.h>

#include "volume.h"
#include "TraceStuff.h"

#include "IOMat.h"
#include "zxxglslvec.h"
#include "math_constants.h"

extern "C" __constant__ Params params;

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

    if (volin.forceNearestVDBSampling) {
        const int3 coord = make_int3(
            __float2int_rn(point_indexd[0]),
            __float2int_rn(point_indexd[1]),
            __float2int_rn(point_indexd[2]));
        return acc.getValue(reinterpret_cast<const nanovdb::Coord&>(coord));
    }

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

template <bool DENSITY, bool DENSITY_BAKE = false>
__device__ void evalVolumeMaterialCore(VolumeInX& attrs, bool shadowRay, VolumeOut& output) {

    let uniforms = params.d_uniforms;
    let buffers = params.global_buffers;

    auto& prd = attrs;

    vec3 att_pos;
    if constexpr (DENSITY_BAKE) {
        att_pos = attrs.pos_view;
    } else {
        att_pos = attrs.pos_view + params.cam.eye;
    }
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

extern "C" __device__ void __direct_callable__evalmat(void* attrs_ptr, bool shadowRay, VolumeOut& output) {
    if (output.density < __half(0))
        __proxy_callable__evalmat<false>(attrs_ptr, shadowRay, output);
    else
        __proxy_callable__evalmat<true>(attrs_ptr, shadowRay, output);
}

#include "VolumeDensityBake.cu"
