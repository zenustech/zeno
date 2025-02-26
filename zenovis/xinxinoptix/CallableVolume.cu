#include <nanovdb/NanoVDB.h>
#include <nanovdb/util/Ray.h>
#include <nanovdb/util/HDDA.h>
#include <nanovdb/util/SampleFromVoxels.h>

#include "volume.h"
#include "TraceStuff.h"
#include "zxxglslvec.h"
#include "math_constants.h"

// #include <cuda_fp16.h>
// #include "nvfunctional"

enum struct VolumeEmissionScaleType {
    Raw, Density, Absorption
};

//PLACEHOLDER
using DataTypeNVDB0 = nanovdb::Fp32;
using GridTypeNVDB0 = nanovdb::NanoGrid<DataTypeNVDB0>;
#define VolumeEmissionScale VolumeEmissionScaleType::Raw
//PLACEHOLDER

#define _USING_NANOVDB_ true

//COMMON_CODE

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

struct VolumeIn2 {
    float3 pos_world;
    float3 pos_view;

    bool isShadowRay;

	float sigma_t;
	uint32_t* seed;

    
    void* sbt_ptr;
    float* world2object;

	inline float rndf() const {
		return rnd(*seed);
	}

    vec3 _local_pos_ = vec3(CUDART_NAN_F);
    vec3 _uniform_pos_ = vec3(CUDART_NAN_F);

    __device__ vec3 localPosLazy() {
		if (isfinite(_local_pos_.x)) return _local_pos_;

        if (world2object != nullptr) {
            mat4* _w2o = reinterpret_cast<mat4*>(world2object);
            vec4 tmp = (*_w2o) * vec4(pos_view.x, pos_view.y, pos_view.z, 1.0f);
            
            _local_pos_ = *(vec3*)&tmp;
        }
        return _local_pos_;
    };

    __device__ vec3 uniformPosLazy() {
		if (isfinite(_uniform_pos_.x)) return _uniform_pos_;

        using GridTypeNVDB = GridTypeNVDB0;
        const HitGroupData* sbt_data = reinterpret_cast<HitGroupData*>( sbt_ptr );

        assert(sbt_data != nullptr);

        const auto grid_ptr = sbt_data->vdb_grids[0];
        const auto* _grid = reinterpret_cast<const GridTypeNVDB*>(grid_ptr);

        if (_grid == nullptr) {
            auto local_pos = localPosLazy();
            _uniform_pos_ = local_pos + 0.5f;
            return _uniform_pos_;
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

        auto local_pos = localPosLazy();

        auto _uniform_pos_ = (local_pos - min) / (max - min);
        _uniform_pos_ = clamp(_uniform_pos_, vec3(0.0f), vec3(1.0f));

        // assert(_uniform_pos_.x >= 0);
        // assert(_uniform_pos_.y >= 0);
        // assert(_uniform_pos_.z >= 0);
        return _uniform_pos_;
    };
};

template <typename Acc, typename DataTypeNVDB, uint8_t Order>
inline __device__ float nanoSampling(Acc& acc, nanovdb::Vec3f& point_indexd, const VolumeIn2& volin) {
    
    using GridTypeNVDB = nanovdb::NanoGrid<DataTypeNVDB>;

    if constexpr(1 == Order) {
        auto iii = interp_trilinear_stochastic(reinterpret_cast<float3&>(point_indexd), volin.rndf());
        return acc.getValue(reinterpret_cast<nanovdb::Coord&>(iii));
    }

    if constexpr(3 > Order) {
        using Sampler = nanovdb::SampleFromVoxels<typename GridTypeNVDB::AccessorType, Order, true>;
        return Sampler(acc)(point_indexd);
    }

    if constexpr(3 == Order) {
    
        auto fff = reinterpret_cast<float3&>(point_indexd);
        fff = interp_tricubic_to_trilinear_stochastic(fff, volin.rndf());
        using Sampler = nanovdb::SampleFromVoxels<typename GridTypeNVDB::AccessorType, 1, true>;
        return Sampler(acc)( reinterpret_cast<nanovdb::Vec3f&>(fff) );
    }
    
    if constexpr(4 == Order) {

        auto uuu = nanovdb::Vec3f(volin.rndf(), volin.rndf(), volin.rndf());
             uuu -= nanovdb::Vec3f(0.5f);
        auto pick = nanovdb::RoundDown<nanovdb::Vec3f>(point_indexd + uuu);
        auto coord = nanovdb::Coord(pick[0], pick[1], pick[2]);
        return acc.getValue(coord);
    }

    return 0.0f;
}

template <uint8_t Order, bool WorldSpace, typename DataTypeNVDB>
static __inline__ __device__ vec2 samplingVDB(const unsigned long long grid_ptr, vec3 att_pos, VolumeIn2& volin, bool cihou) {
    using GridTypeNVDB = nanovdb::NanoGrid<DataTypeNVDB>;

    const auto* _grid = reinterpret_cast<const GridTypeNVDB*>(grid_ptr);
    const auto& _acc = _grid->tree().getAccessor();

    if (_grid == nullptr) { return {}; }

    auto pos_indexed = reinterpret_cast<const nanovdb::Vec3f&>(att_pos);

    if constexpr(WorldSpace) 
    {
        if (cihou) {
            pos_indexed = volin.localPosLazy();
        } else {
            pos_indexed = _grid->worldToIndexF(pos_indexed);
        }
    } //_grid->tree().root().maximum();

    return vec2 { nanoSampling<decltype(_acc), DataTypeNVDB, Order>(_acc, pos_indexed, volin), _grid->tree().root().maximum() };
}

extern "C" __device__ void __direct_callable__evalmat(const float4* uniforms, void** buffers, void* attrs_ptr, VolumeOut& output) {

    auto& attrs = *reinterpret_cast<VolumeIn2*>(attrs_ptr);
    auto& prd = attrs;

    vec3& att_pos = reinterpret_cast<vec3&>(attrs.pos_world);
    auto att_clr = vec3(0);
    auto att_uv = vec3(0);
    auto att_nrm = vec3(0);
    auto att_tang = vec3(0);
	
    HitGroupData* sbt_data = reinterpret_cast<HitGroupData*>(attrs.sbt_ptr);
    auto zenotex = sbt_data->textures;
    auto vdb_grids = sbt_data->vdb_grids;
    auto vdb_max_v = sbt_data->vdb_max_v;

    auto att_isBackFace = false;
    auto att_isShadowRay = attrs.isShadowRay;
    
#ifndef _FALLBACK_

    //GENERATED_BEGIN_MARK 
    auto anisotropy = 0.0f;
    auto density = 0.0f;

    vec3 emission = vec3(0.0f);
    vec3 albedo = vec3(0.5f);
    auto extinction = vec3(1.0f);
    //GENERATED_END_MARK
#else
	auto anisotropy = 0.0f;
    auto density = 0.1f;

	vec3 tmp = { 1, 0, 1 };

    vec3 emission = tmp / 50.f;
    vec3 albedo = tmp;
    auto extinction = vec3(1.0f);
#endif // _FALLBACK_

#if _USING_NANOVDB_

    output.albedo = clamp(albedo, 0.0f, 1.0f);
    output.anisotropy = clamp(anisotropy, -1.0f, 1.0f);
    output.extinction = extinction;

    output.density = fmaxf(density, 0.0f);
    output.emission = fmaxf(emission, vec3(0.0f));

	if constexpr(VolumeEmissionScale == VolumeEmissionScaleType::Raw) {
		//output.emission = output.emission; 
	} else if constexpr(VolumeEmissionScale == VolumeEmissionScaleType::Density) {
		output.emission = output.density * output.emission;
	} else if constexpr(VolumeEmissionScale == VolumeEmissionScaleType::Absorption) {

		auto sigma_t = attrs.sigma_t;

		float sigma_a = sigma_t * output.density * average(1.0f - output.albedo);
		sigma_a = fmaxf(sigma_a, 0.0f);
		auto tmp = output.emission * sigma_a;
		output.step_scale = 1.0f / fmaxf(sigma_t, average(tmp)); 
		output.emission = tmp / sigma_t;
	}
    
#else
    //USING 3D ARRAY
    //USING 3D Noise 
#endif
}