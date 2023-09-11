#include "volume.h"

#include "TraceStuff.h"

#include "DisneyBRDF.h"
#include "DisneyBSDF.h"
#include "zxxglslvec.h"
#include "proceduralSky.h"

// #include <cuda_fp16.h>
// #include "nvfunctional"
#include <math_constants.h>
#include <nanovdb/NanoVDB.h>
#include <nanovdb/util/Ray.h>
#include <nanovdb/util/HDDA.h>
#include <nanovdb/util/SampleFromVoxels.h>

//PLACEHOLDER
static const int   _vol_depth = 99;
static const float _vol_extinction = 1.0f;
using DataTypeNVDB0 = nanovdb::Fp32;
using GridTypeNVDB0 = nanovdb::NanoGrid<DataTypeNVDB0>;
//PLACEHOLDER

#define USING_VDB 1
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

    if constexpr(3 == Order) {
        nanovdb::SampleFromVoxels<typename GridTypeNVDB::AccessorType, 1, true> s(acc);
        return interp_tricubic_nanovdb(s, point_indexd[0], point_indexd[1], point_indexd[2]);
    } else {
        using Sampler = nanovdb::SampleFromVoxels<typename GridTypeNVDB::AccessorType, Order, true>;
        return Sampler(acc)(point_indexd);
    }
        //nanovdb::BaseStencil<typename DerivedType, int SIZE, typename GridT>
        //auto bs = nanovdb::BoxStencil<nanovdb::FloatGrid*>(grid);

    // auto point_floor = nanovdb::RoundDown<nanovdb::Vec3f>(point_indexd); 
    // auto point_a = nanovdb::Coord(point_floor[0], point_floor[1], point_floor[2]);

    //     auto value_000 = acc.getValue(point_a);
    //     auto value_100 = acc.getValue(point_a + nanovdb::Coord(1, 0, 0));
    //     auto value_010 = acc.getValue(point_a + nanovdb::Coord(0, 1, 0));
    //     auto value_110 = acc.getValue(point_a + nanovdb::Coord(1, 1, 0));
    //     auto value_001 = acc.getValue(point_a + nanovdb::Coord(0, 0, 1));
    //     auto value_101 = acc.getValue(point_a + nanovdb::Coord(1, 0, 1));
    //     auto value_011 = acc.getValue(point_a + nanovdb::Coord(0, 1, 1));
    //     auto value_111 = acc.getValue(point_a + nanovdb::Coord(1, 1, 1));

    // auto delta = point_indexd - point_floor; 

    //     auto value_00 = _LERP_(delta[0], value_000, value_100);
    //     auto value_10 = _LERP_(delta[0], value_010, value_110);
    //     auto value_01 = _LERP_(delta[0], value_001, value_101);
    //     auto value_11 = _LERP_(delta[0], value_011, value_111);
        
    //     auto value_0 = _LERP_(delta[1], value_00, value_10);
    //     auto value_1 = _LERP_(delta[1], value_01, value_11);

    // return _LERP_(delta[2], value_0, value_1);
}

struct VolumeIn {
    vec3 pos;

    vec3 _local_pos_ = vec3(CUDART_NAN_F);
    __inline__ __device__ vec3 localPosLazy() {
        if ( isnan(_local_pos_.x) ) {
            using GridTypeNVDB = GridTypeNVDB0;

            const HitGroupData* sbt_data = reinterpret_cast<HitGroupData*>( optixGetSbtDataPointer() );

            const auto grid_ptr = sbt_data->vdb_grids[0];
            const auto* _grid = reinterpret_cast<const GridTypeNVDB*>(grid_ptr);
            //const auto& _acc = _grid->tree().getAccessor();
            auto pos_indexed = reinterpret_cast<const nanovdb::Vec3f&>(pos);
            pos_indexed = _grid->worldToIndexF(pos_indexed);

            _local_pos_ = reinterpret_cast<vec3&>(pos_indexed);
        }
        return _local_pos_;
    }

    vec3 _uniform_pos_ = vec3(CUDART_NAN_F);
    __inline__ __device__ vec3 uniformPosLazy() {
        if ( isnan(_uniform_pos_.x) ) {
            using GridTypeNVDB = GridTypeNVDB0;

            const HitGroupData* sbt_data = reinterpret_cast<HitGroupData*>( optixGetSbtDataPointer() );

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

            _uniform_pos_ = (local_pos - min) / (max - min);
            _uniform_pos_ = clamp(_uniform_pos_, vec3(0.0f), vec3(1.0f));

            assert(_uniform_pos_.x >= 0);
            assert(_uniform_pos_.y >= 0);
            assert(_uniform_pos_.z >= 0);
        }
        return _uniform_pos_;
    }
};

struct VolumeOut {
    float max_density;
    float density;

    float anisotropy;
    vec3 emission;
    vec3 albedo;
};

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

static __inline__ __device__ VolumeOut evalVolume(float4* uniforms, VolumeIn &attrs) {

    auto att_pos = attrs.pos;
    auto att_clr = vec3(0);
    auto att_uv = vec3(0);
    auto att_nrm = vec3(0);
    auto att_tang = vec3(0);

    HitGroupData* sbt_data = (HitGroupData*)optixGetSbtDataPointer();
    auto zenotex = sbt_data->textures;
    auto vdb_grids = sbt_data->vdb_grids;
    auto vdb_max_v = sbt_data->vdb_max_v;

    //GENERATED_BEGIN_MARK   
        auto vol_sample_anisotropy = 0.0f;
        auto vol_sample_density = 0.0f;

        vec3 vol_sample_emission = vec3(0.0f);
        vec3 vol_sample_albedo = vec3(0.5f);
    //GENERATED_END_MARK

#if USING_VDB

    VolumeOut output;

    output.albedo = clamp(vol_sample_albedo, 0.0f, 1.0f);
    output.anisotropy = clamp(vol_sample_anisotropy, -0.99f, 0.99f);

    output.density = clamp(vol_sample_density, 0.0f, 1.0f);
    output.emission = vol_sample_emission;
    
    return output;
#else
    //USING 3D ARRAY
    //USING 3D Noise 
#endif
}

// ----------------------------------------------------------------------------
// Volume programs
// ----------------------------------------------------------------------------

inline __device__ void confine( const nanovdb::BBox<nanovdb::Coord> &bbox, nanovdb::Vec3f &iVec )
{
    // NanoVDB's voxels and tiles are formed from half-open intervals, i.e.
    // voxel[0, 0, 0] spans the set [0, 1) x [0, 1) x [0, 1). To find a point's voxel,
    // its coordinates are simply truncated to integer. Ray-box intersections yield
    // pairs of points that, because of numerical errors, fall randomly on either side
    // of the voxel boundaries.
    // This confine method, given a point and a (integer-based/Coord-based) bounding
    // box, moves points outside the bbox into it. That means coordinates at lower
    // boundaries are snapped to the integer boundary, and in case of the point being
    // close to an upper boundary, it is move one EPS below that bound and into the volume.

    // get the tighter box around active values
    auto iMin = nanovdb::Vec3f( bbox.min() );
    auto iMax = nanovdb::Vec3f( bbox.max() ) + nanovdb::Vec3f( 1.0f );

    // move the start and end points into the bbox
    float eps = 1e-7f;
    if( iVec[0] < iMin[0] ) iVec[0] = iMin[0];
    if( iVec[1] < iMin[1] ) iVec[1] = iMin[1];
    if( iVec[2] < iMin[2] ) iVec[2] = iMin[2];
    if( iVec[0] >= iMax[0] ) iVec[0] = iMax[0] - fmaxf( 1.0f, fabsf( iVec[0] ) ) * eps;
    if( iVec[1] >= iMax[1] ) iVec[1] = iMax[1] - fmaxf( 1.0f, fabsf( iVec[1] ) ) * eps;
    if( iVec[2] >= iMax[2] ) iVec[2] = iMax[2] - fmaxf( 1.0f, fabsf( iVec[2] ) ) * eps;
}

inline __hostdev__ void confine( const nanovdb::BBox<nanovdb::Coord> &bbox, nanovdb::Vec3f &iStart, nanovdb::Vec3f &iEnd )
{
    confine( bbox, iStart );
    confine( bbox, iEnd );
}

template<typename AccT>
inline __device__ float transmittanceHDDA(
    const nanovdb::Vec3f& start,
    const nanovdb::Vec3f& end,
    AccT& acc, const float opacity )
{

    // transmittance along a ray through the volume is computed by
    // taking the negative exponential of volume's density integrated
    // along the ray.
    float transmittance = 1.f;
    auto dir = end - start;
    auto len = dir.length();
    nanovdb::Ray<float> ray( start, dir / len, 0.0f, len );
    nanovdb::Coord ijk = nanovdb::RoundDown<nanovdb::Coord>( ray.start() ); // first hit of bbox

    // Use NanoVDB's HDDA line digitization for fast integration.
    // This algorithm (http://www.museth.org/Ken/Publications_files/Museth_SIG14.pdf)
    // can skip over sparse parts of the data structure.
    //
    nanovdb::HDDA<nanovdb::Ray<float> > hdda( ray, acc.getDim( ijk, ray ) );

    float t = 0.0f;
    float density = acc.getValue( ijk ) * opacity;
    while( hdda.step())
    {
        float dt = hdda.time() - t; // compute length of ray-segment intersecting current voxel/tile
        transmittance *= expf( -density * dt );
        t = hdda.time();
        ijk = hdda.voxel();

        density = acc.getValue( ijk ) * opacity;
        hdda.update( ray, acc.getDim( ijk, ray ) ); // if necessary adjust DDA step size
    }

    return transmittance;
}


extern "C" __global__ void __intersection__volume()
{
    const auto* sbt_data = reinterpret_cast<const HitGroupData*>( optixGetSbtDataPointer() );
    const auto* grid = reinterpret_cast<const GridTypeNVDB0*>( sbt_data->vdb_grids[0] );
    assert( grid );

    const float3 ray_orig = optixGetWorldRayOrigin(); //optixGetObjectRayOrigin();
    const float3 ray_dir  = optixGetWorldRayDirection(); //optixGetObjectRayDirection();

    auto dbox = grid->worldBBox(); //grid->indexBBox();
    float t0 = optixGetRayTmin();
    float t1 = _FLT_MAX_; //optixGetRayTmax();

    auto iray = nanovdb::Ray<float>( reinterpret_cast<const nanovdb::Vec3f&>( ray_orig ),
                                     reinterpret_cast<const nanovdb::Vec3f&>( ray_dir ), t0, t1 );
    // auto fbox = nanovdb::BBox<nanovdb::Vec3f>(nanovdb::Vec3f(dbox.min()), nanovdb::Vec3f(dbox.max()));

    if( iray.intersects( dbox, t0, t1 )) // t0 >= 0
    {
        // report the entry-point as hit-point
        //auto kind = optixGetHitKind();
        t0 = fmaxf(t0, optixGetRayTmin());

        RadiancePRD* prd = getPRD();
        prd->vol_t0 = t0;
        prd->origin_inside_vdb = (t0 == 0);

        prd->vol_t1 = t1; //min(optixGetRayTmax(), t1);
        prd->surface_inside_vdb = (optixGetRayTmax() < t1); // In case triangles were visited before vdb

        if (optixGetRayTmax() > 0) {
            optixReportIntersection(t0, 0);
        }
    } 
}

extern "C" __global__ void __closesthit__radiance_volume()
{
    RadiancePRD* prd = getPRD();
    //if(prd->test_distance) { return; }
    
    prd->countEmitted = false;
    prd->radiance = make_float3(0);

    prd->trace_tmin = 0;
    prd->_mask_ = EverythingMask;

    const HitGroupData* sbt_data = reinterpret_cast<HitGroupData*>( optixGetSbtDataPointer() );

    float3 ray_orig = optixGetWorldRayOrigin();
    float3 ray_dir  = optixGetWorldRayDirection();

        float t0 = prd->vol_t0; // world space
        float t1 = prd->vol_t1; // world space

    RadiancePRD testPRD {};
    testPRD.vol_t1 = _FLT_MAX_;
    testPRD.test_distance = true;
    testPRD.isSS = false;
    testPRD.opacity = 0.0f;
    traceRadianceMasked(
        params.handle,
        ray_orig,
        ray_dir,
        0,
        _FLT_MAX_,
        DefaultMatMask,
        &testPRD);

    if(testPRD.vol_t1 < t1)
    {
        t1 = testPRD.vol_t1;
        prd->surface_inside_vdb = true;
    }

    const float t_max = fmax(0.f, t1 - t0); // world space
    float t_ele = 0;

    auto test_point = ray_orig; 
    float3 emitting = make_float3(0.0);
    float3 scattering = make_float3(1.0);
   
    float sigma_t = _vol_extinction;
    float v_density = 0.0;

    VolumeOut vol_out;

#if (!_DELTA_TRACKING_) 

    test_point = ray_orig + test_t * ray_dir;
    auto test_point_indexd = grid->worldToIndexF(reinterpret_cast<const nanovdb::Vec3f&>(test_point));
    v_density = linearSampling(acc, test_point_indexd);
    
    vec3 new_dir = ray_dir;
    
    if(v_density > 0){
        const auto ray = nanovdb::Ray<float>( reinterpret_cast<const nanovdb::Vec3f&>( ray_orig ),
                                              reinterpret_cast<const nanovdb::Vec3f&>( ray_dir ) );
        auto start = grid->worldToIndexF( ray( t0 ) );
        auto end   = grid->worldToIndexF( ray( test_t ) );

        auto bbox = grid->indexBBox();
        confine( bbox, start, end );

        const float opacity = sbt_data->opacityHDDA;
        
        //scattering *= transmittanceHDDA( start, end, acc, 0.01 );;
        new_dir = DisneyBSDF::SampleScatterDirection(prd->seed);

        pbrt::HenyeyGreenstein hg {sbt_data->greenstein};
        float3 new_dir; float2 uu = {rnd(prd->seed), rnd(prd->seed)};
        auto pdf = hg.Sample_p(-ray_dir,             new_dir, uu);
        // //scattering *= pdf;

        scattering *= sbt_data->colorVDB;        
        ray_dir = (prd->volumeHitSurface )? ray_dir : float3(new_dir);
    }

#else

    auto level = _vol_depth;
    while(--level > 0) {
        auto prob = rnd(prd->seed);
        t_ele -= log(1.0f-prob) / (sigma_t);

        if (t_ele >= t_max) {

            if (prd->surface_inside_vdb) { // Hit other material

                prd->_mask_ = DefaultMatMask;
                prd->trace_tmin = 0;

                test_point = ray_orig;

            } else { // Volume edge

                prd->_mask_ = EverythingMask;
                prd->trace_tmin = 1e-5f;

                test_point = ray_orig + t1 * ray_dir;
                test_point = rtgems::offset_ray(test_point, ray_dir);
            }

            v_density = 0;
            break;
        } // over shoot, outside of volume

        test_point = ray_orig + (t0+t_ele) * ray_dir;

        VolumeIn vol_in { test_point };
        
        vol_out = evalVolume(sbt_data->uniforms, vol_in);
        v_density = vol_out.density;
        
        float s_prob = vol_out.density;
        float n_prob = 1.0f - s_prob;
        
        float3 s_prob_rgb = vol_out.albedo;
        float3 a_prob_rgb = 1.0f - s_prob_rgb;

        float3 n_prob_rgb = ( n_prob/s_prob ) * s_prob_rgb;
        float3 _prob_rgb = n_prob_rgb + a_prob_rgb + s_prob_rgb;

        float3 le = vol_out.emission;

        if ( length(le) > 0.0f ) {
            //float3 emission_prob = a_prob_rgb / _prob_rgb; // scale by emission prob
            //le *= emission_prob; 
            emitting += le;
        }

        if (rnd(prd->seed) < v_density) {

            float3 new_dir; 
            pbrt::HenyeyGreenstein hg { vol_out.anisotropy };
                
            float2 uu = {rnd(prd->seed), rnd(prd->seed)};
            auto prob = hg.Sample_p(-ray_dir, new_dir, uu);              
            //auto relative_prob = prob * (CUDART_PI_F * 4);
            new_dir = normalize(new_dir);
            if (prd->trace_denoise_normal) {
                prd->tmp_normal = 0.5f * (-ray_dir + new_dir);
            }
            ray_dir = new_dir;

            if(prd->trace_denoise_albedo) {
                prd->tmp_albedo = vol_out.albedo;
            }
            scattering = s_prob_rgb * (_prob_rgb - a_prob_rgb) / _prob_rgb; // scattering prob
            break;
        } else {
            v_density = 0; 
        } 
    }

#endif // _DELTA_TRACKING_

    prd->updateAttenuation(scattering);

    ray_orig = test_point;
    prd->origin = ray_orig;
    prd->direction = ray_dir;

    prd->emission = emitting;

    if (v_density == 0) {
        prd->CH = 0.0;
        //prd->depth += 0;
        prd->radiance += prd->emission;
        return;
    }

    float3 light_attenuation = make_float3(1.0f);
    float pl = rnd(prd->seed);
    float sum = 0.0f;
    for(int lidx=0;lidx<params.num_lights;lidx++)
    {
            ParallelogramLight light = params.lights[lidx];
            float3 light_pos = light.corner + light.v1 * 0.5f + light.v2 * 0.5f;

            // Calculate properties of light sample (for area based pdf)
            float Ldist = length(light_pos - test_point);
            float3 L = normalize(light_pos - test_point);
            float nDl = 1.0f;//clamp(dot(N, L), 0.0f, 1.0f);
            float LnDl = clamp(-dot(light.normal, L), 0.000001f, 1.0f);
            float A = length(cross(params.lights[lidx].v1, params.lights[lidx].v2));
            sum += length(light.emission)  * nDl * LnDl * A / (M_PIf * Ldist * Ldist);
    }

    RadiancePRD shadow_prd {};
    shadow_prd.seed = prd->seed;

    shadow_prd.nonThinTransHit = 0;
    shadow_prd.shadowAttanuation = make_float3(1.0f);

    scattering = vol_out.albedo;
    
    if(rnd(prd->seed)<=0.5f) {
        bool computed = false;
        float ppl = 0;
        for (int lidx = 0; lidx < params.num_lights && computed == false; lidx++) {
            ParallelogramLight light = params.lights[lidx];
            float2 z = {rnd(prd->seed), rnd(prd->seed) };
            const float z1 = z.x;
            const float z2 = z.y;
            float3 light_tpos = light.corner + light.v1 * 0.5f + light.v2 * 0.5f;
            float3 light_pos = light.corner + light.v1 * z1 + light.v2 * z2;

            // Calculate properties of light sample (for area based pdf)
            float tLdist = length(light_tpos - test_point);
            float3 tL = normalize(light_tpos - test_point);
            float tnDl = 1.0f; //clamp(dot(N, tL), 0.0f, 1.0f);
            float tLnDl = clamp(-dot(light.normal, tL), 0.000001f, 1.0f);
            float tA = length(cross(params.lights[lidx].v1, params.lights[lidx].v2));
            ppl += length(light.emission) * tnDl * tLnDl * tA / (M_PIf * tLdist * tLdist) / sum;
            if (ppl > pl) {
                float Ldist = length(light_pos - test_point) + 1e-6f;
                float3 L = normalize(light_pos - test_point);
                float nDl = 1.0f; //clamp(dot(N, L), 0.0f, 1.0f);
                float LnDl = clamp(-dot(light.normal, L), 0.0f, 1.0f);
                float A = length(cross(params.lights[lidx].v1, params.lights[lidx].v2));
                float weight = 0.0f;
                if (nDl > 0.0f && LnDl > 0.0f) {
                    
                    traceOcclusion(params.handle, test_point, L,
                                   0,         // tmin
                                   Ldist - 1e-5f, // tmax,
                                   &shadow_prd);

                    light_attenuation = shadow_prd.shadowAttanuation;

                    weight = sum * nDl / tnDl * LnDl / tLnDl * (tLdist * tLdist) / (Ldist * Ldist) /
                                (length(light.emission)+1e-6f);
                }
                // prd->LP = test_point;
                // prd->Ldir = L;
                // prd->Lweight = weight;
                // prd->nonThinTransHit = 0;
                
                pbrt::HenyeyGreenstein hg { vol_out.anisotropy };
                float ray_prob = hg.p(-ray_dir, L);
                float3 lbrdf = scattering * ray_prob;

                prd->radiance = light_attenuation * weight * 2.0f * light.emission * lbrdf;
                computed = true;
            }
        }
    } else {

        vec3 sunLightDir = vec3(params.sunLightDirX, params.sunLightDirY, params.sunLightDirZ);
        auto sun_dir = BRDFBasics::halfPlaneSample(prd->seed, sunLightDir,
                                                   params.sunSoftness * 0.2f); //perturb the sun to have some softness
        sun_dir = normalize(sun_dir);
        // prd->LP = test_point;
        // prd->Ldir = sun_dir;
        // prd->Lweight = 1.0;
        // prd->nonThinTransHit = 1;
        traceOcclusion(params.handle, test_point, sun_dir,
                       0, // tmin
                       1e16f, // tmax,
                       &shadow_prd);

        light_attenuation = shadow_prd.shadowAttanuation;

        pbrt::HenyeyGreenstein hg { vol_out.anisotropy };
        float ray_prob = hg.p(-ray_dir, sun_dir);
        float3 lbrdf = scattering * clamp(ray_prob, 0.0f, 1.0f);

        float tmpPdf;
        auto illum = float3(envSky(sun_dir, sunLightDir, make_float3(0., 0., 1.),
                                     40, // be careful
                                     .45, 15., 1.030725f * 0.3f, params.elapsedTime, tmpPdf));

        prd->radiance = light_attenuation * params.sunLightIntensity * 2.0f * lbrdf * illum;
    }

    prd->CH = 1.0;
    prd->depth += 1;
    prd->radiance += prd->emission;

    return;
}

extern "C" __global__ void __anyhit__occlusion_volume()
{
    const float3 ray_orig = optixGetWorldRayOrigin();
    const float3 ray_dir  = optixGetWorldRayDirection();

    RadiancePRD* prd = getPRD();
    const HitGroupData* sbt_data = reinterpret_cast<HitGroupData*>( optixGetSbtDataPointer() );

    const float t0 = prd->vol_t0;
    const float t1 = prd->vol_t1;

    const float t_max = t1 - t0; // world space
          float t_ele = 0;

    float3 test_point = ray_orig; 
    float3 transmittance = make_float3(1.0f);

    const float sigma_t = _vol_extinction;

#if (!_DELTA_TRACKING_) 

    const auto ray = nanovdb::Ray<float>( reinterpret_cast<const nanovdb::Vec3f&>( ray_orig ),
                                              reinterpret_cast<const nanovdb::Vec3f&>( ray_dir ) );
    auto start = grid->worldToIndexF( ray( t0 ) );
    auto end   = grid->worldToIndexF( ray( t1 ) );

    auto bbox = grid->indexBBox();
    confine( bbox, start, end );

    const float opacity = sbt_data->opacityHDDA;
    float transHDDA = transmittanceHDDA( start, end, acc, sbt_data->opacityHDDA );
    if (transHDDA < 1.0) {
        transmittance *= transHDDA;
        transmittance *= sbt_data->colorVDB;
    }

#else
    auto level = _vol_depth;
    while(--level > 0) {

        auto prob = rnd(prd->seed);
        t_ele -= log(1.0f-prob) / (sigma_t);

        test_point = ray_orig + (t0+t_ele) * ray_dir;

        if (t_ele >= t_max) {
            break;
        } // over shoot, outside of volume

        VolumeIn vol_in { test_point };
        VolumeOut vol_out = evalVolume(sbt_data->uniforms, vol_in);

        const auto v_density = vol_out.density;

        transmittance *= 1.0f - clamp(v_density, 0.0f, 1.0f);
        auto avg = dot(transmittance, make_float3(1.0f/3.0f));
        if (avg < 0.1f) {
            float q = fmax(0.05f, 1 - avg);
            if (rnd(prd->seed) < q) { 
                transmittance = vec3(0);
                break; 
            } else {
                transmittance /= 1-q;
            }
        }
        if (v_density > 0) {
            transmittance *= vol_out.albedo;
        }
    }
#endif

    prd->shadowAttanuation *= transmittance;
    optixIgnoreIntersection();
    //prd->origin = ray_orig;
    //prd->direction = ray_dir;
    return;
}