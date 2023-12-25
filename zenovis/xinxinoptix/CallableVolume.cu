#include "volume.h"

#include "TraceStuff.h"
#include "zxxglslvec.h"
#include "math_constants.h"

// #include <cuda_fp16.h>
// #include "nvfunctional"
#include <nanovdb/NanoVDB.h>
#include <nanovdb/util/Ray.h>
#include <nanovdb/util/HDDA.h>
#include <nanovdb/util/SampleFromVoxels.h>

enum struct VolumeEmissionScalerType {
    Raw, Density, Absorption
};

//PLACEHOLDER
using DataTypeNVDB0 = nanovdb::Fp32;
using GridTypeNVDB0 = nanovdb::NanoGrid<DataTypeNVDB0>;
#define VolumeEmissionScaler VolumeEmissionScalerType::Raw
//PLACEHOLDER

#define _USING_NANOVDB_ true

//COMMON_CODE

/* w0, w1, w2, and w3 are the four cubic B-spline basis functions. */
inline __device__ float cubic_w0(float a)
{
  return (1.0f / 6.0f) * (a * (a * (-a + 3.0f) - 3.0f) + 1.0f);
}
inline __device__ float cubic_w1(float a)
{
  return (1.0f / 6.0f) * (a * a * (3.0f * a - 6.0f) + 4.0f);
}
inline __device__ float cubic_w2(float a)
{
  return (1.0f / 6.0f) * (a * (a * (-3.0f * a + 3.0f) + 3.0f) + 1.0f);
}
inline __device__ float cubic_w3(float a)
{
  return (1.0f / 6.0f) * (a * a * a);
}

/* g0 and g1 are the two amplitude functions. */
inline __device__ float cubic_g0(float a)
{
  return cubic_w0(a) + cubic_w1(a);
}
inline __device__ float cubic_g1(float a)
{
  return cubic_w2(a) + cubic_w3(a);
}

/* h0 and h1 are the two offset functions */
inline __device__ float cubic_h0(float a)
{
  return (cubic_w1(a) / cubic_g0(a)) - 1.0f;
}
inline __device__ float cubic_h1(float a)
{
  return (cubic_w3(a) / cubic_g1(a)) + 1.0f;
}

template<typename S>
inline __device__ float interp_tricubic_nanovdb(S &s, float x, float y, float z)
{
  float px = floorf(x);
  float py = floorf(y);
  float pz = floorf(z);
  float fx = x - px;
  float fy = y - py;
  float fz = z - pz;

  float g0x = cubic_g0(fx);
  float g1x = cubic_g1(fx);
  float g0y = cubic_g0(fy);
  float g1y = cubic_g1(fy);
  float g0z = cubic_g0(fz);
  float g1z = cubic_g1(fz);

  float x0 = px + cubic_h0(fx);
  float x1 = px + cubic_h1(fx);
  float y0 = py + cubic_h0(fy);
  float y1 = py + cubic_h1(fy);
  float z0 = pz + cubic_h0(fz);
  float z1 = pz + cubic_h1(fz);

  using namespace nanovdb;

  return g0z * (g0y * (g0x * s(Vec3f(x0, y0, z0)) + g1x * s(Vec3f(x1, y0, z0))) +
                g1y * (g0x * s(Vec3f(x0, y1, z0)) + g1x * s(Vec3f(x1, y1, z0)))) +
         g1z * (g0y * (g0x * s(Vec3f(x0, y0, z1)) + g1x * s(Vec3f(x1, y0, z1))) +
                g1y * (g0x * s(Vec3f(x0, y1, z1)) + g1x * s(Vec3f(x1, y1, z1))));
}

inline __device__ float _LERP_(float t, float s1, float s2)
{
    //return (1 - t) * s1 + t * s2;
    return fma(t, s2, fma(-t, s1, s1));
}

template <typename Acc, typename DataTypeNVDB, uint8_t Order>
inline __device__ float nanoSampling(Acc& acc, nanovdb::Vec3f& point_indexd) {
    
    using GridTypeNVDB = nanovdb::NanoGrid<DataTypeNVDB>;

    if constexpr(3 > Order) {
        using Sampler = nanovdb::SampleFromVoxels<typename GridTypeNVDB::AccessorType, Order, true>;
        return Sampler(acc)(point_indexd);
    }

    if constexpr(3 == Order) {
        nanovdb::SampleFromVoxels<typename GridTypeNVDB::AccessorType, 1, true> s(acc);
        return interp_tricubic_nanovdb(s, point_indexd[0], point_indexd[1], point_indexd[2]);
    } 
    
    if constexpr(4 == Order) {
        RadiancePRD* prd = getPRD();
        auto uuu = nanovdb::Vec3f(prd->rndf(), prd->rndf(), prd->rndf());
             uuu -= nanovdb::Vec3f(0.5f);
        auto pick = nanovdb::RoundDown<nanovdb::Vec3f>(point_indexd + uuu);
        auto coord = nanovdb::Coord(pick[0], pick[1], pick[2]);
        return acc.getValue(coord);
    }

    return 0.0f;

    // auto point_floor = nanovdb::RoundDown<nanovdb::Vec3f>(point_indexd); 
    // auto point_a = nanovdb::Coord(point_floor[0], point_floor[1], point_floor[2]);
    // auto delta = point_indexd - point_floor; 

    //     auto value_000 = acc.getValue(point_a);
    //     auto value_100 = acc.getValue(point_a + nanovdb::Coord(1, 0, 0));
    //     auto value_010 = acc.getValue(point_a + nanovdb::Coord(0, 1, 0));
    //     auto value_110 = acc.getValue(point_a + nanovdb::Coord(1, 1, 0));
    //     auto value_001 = acc.getValue(point_a + nanovdb::Coord(0, 0, 1));
    //     auto value_101 = acc.getValue(point_a + nanovdb::Coord(1, 0, 1));
    //     auto value_011 = acc.getValue(point_a + nanovdb::Coord(0, 1, 1));
    //     auto value_111 = acc.getValue(point_a + nanovdb::Coord(1, 1, 1));

    //     auto value_00 = _LERP_(delta[0], value_000, value_100);
    //     auto value_10 = _LERP_(delta[0], value_010, value_110);
    //     auto value_01 = _LERP_(delta[0], value_001, value_101);
    //     auto value_11 = _LERP_(delta[0], value_011, value_111);
        
    //     auto value_0 = _LERP_(delta[1], value_00, value_10);
    //     auto value_1 = _LERP_(delta[1], value_01, value_11);

    // return _LERP_(delta[2], value_0, value_1);
}

template <uint8_t Order, bool WorldSpace, typename DataTypeNVDB>
static __inline__ __device__ vec2 samplingVDB(const unsigned long long grid_ptr, vec3 att_pos) {
    using GridTypeNVDB = nanovdb::NanoGrid<DataTypeNVDB>;

    const auto* _grid = reinterpret_cast<const GridTypeNVDB*>(grid_ptr);
    const auto& _acc = _grid->tree().getAccessor();

    auto pos_indexed = reinterpret_cast<const nanovdb::Vec3f&>(att_pos);

    if constexpr(WorldSpace)
    {
        pos_indexed = _grid->worldToIndexF(pos_indexed);
    } //_grid->tree().root().maximum();

    return vec2 { nanoSampling<decltype(_acc), DataTypeNVDB, Order>(_acc, pos_indexed), _grid->tree().root().maximum() };
}

struct VolumeIn2 {
    float3 pos;
	float sigma_t;
	uint32_t* seed;
    unsigned long long sbt_ptr;

	inline float rndf() {
		return rnd(*seed);
	}

    vec3 _local_pos_ = vec3(CUDART_NAN_F);
    vec3 localPosLazy() {
		if (isfinite(_local_pos_.x)) return _local_pos_;

        using GridTypeNVDB = GridTypeNVDB0;
        const HitGroupData* sbt_data = reinterpret_cast<HitGroupData*>( sbt_ptr );

        assert(sbt_data != nullptr);

        const auto grid_ptr = sbt_data->vdb_grids[0];
        const auto* _grid = reinterpret_cast<const GridTypeNVDB*>(grid_ptr);
        //const auto& _acc = _grid->tree().getAccessor();
        auto pos_indexed = reinterpret_cast<const nanovdb::Vec3f&>(pos);
        pos_indexed = _grid->worldToIndexF(pos_indexed);

        _local_pos_ = reinterpret_cast<vec3&>(pos_indexed);
        return _local_pos_;
    };

	vec3 _uniform_pos_ = vec3(CUDART_NAN_F);
    vec3 uniformPosLazy() {
		if (isfinite(_uniform_pos_.x)) return _uniform_pos_;

        using GridTypeNVDB = GridTypeNVDB0;
        const HitGroupData* sbt_data = reinterpret_cast<HitGroupData*>( sbt_ptr );

        assert(sbt_data != nullptr);

        const auto grid_ptr = sbt_data->vdb_grids[0];
        const auto* _grid = reinterpret_cast<const GridTypeNVDB*>(grid_ptr);

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


extern "C" __device__ VolumeOut __direct_callable__evalmat(const float4* uniforms, VolumeIn& _attrs) {

    VolumeIn2 attrs{_attrs.pos, _attrs.sigma_t, _attrs.seed, _attrs.sbt_ptr };
    auto& prd = attrs;

    auto att_pos = attrs.pos;
    auto att_clr = vec3(0);
    auto att_uv = vec3(0);
    auto att_nrm = vec3(0);
    auto att_tang = vec3(0);
	
    HitGroupData* sbt_data = reinterpret_cast<HitGroupData*>(attrs.sbt_ptr);
    auto zenotex = sbt_data->textures;
    auto vdb_grids = sbt_data->vdb_grids;
    auto vdb_max_v = sbt_data->vdb_max_v;

#ifndef _FALLBACK_

    //GENERATED_BEGIN_MARK 
    auto vol_sample_anisotropy = 0.0f;
    auto vol_sample_density = 0.0f;

    vec3 vol_sample_emission = vec3(0.0f);
    vec3 vol_sample_albedo = vec3(0.5f);
    //GENERATED_END_MARK
#else
	auto vol_sample_anisotropy = 0.0f;
    auto vol_sample_density = 0.1f;

	vec3 tmp = { 1, 0, 1 };

    vec3 vol_sample_emission = tmp / 50.f;
    vec3 vol_sample_albedo = tmp;
#endif // _FALLBACK_

VolumeOut output;

#if _USING_NANOVDB_

    output.albedo = clamp(vol_sample_albedo, 0.0f, 1.0f);
    output.anisotropy = clamp(vol_sample_anisotropy, -1.0f, 1.0f);

    output.density = clamp(vol_sample_density, 0.0f, 1.0f);
    output.emission = max(vol_sample_emission, vec3(0.0f));

	if constexpr(VolumeEmissionScaler == VolumeEmissionScalerType::Raw) {
		//output.emission = output.emission; 
	} else if constexpr(VolumeEmissionScaler == VolumeEmissionScalerType::Density) {
		output.emission = output.density * output.emission;
	} else if constexpr(VolumeEmissionScaler == VolumeEmissionScalerType::Absorption) {

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
	return output;
}