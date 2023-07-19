#pragma once

#include <optix.h>

#include "zxxglslvec.h"
#include "optixPathTracer.h"

#define _FLT_EPL_ 1.19209290e-7F

#define _FLT_MAX_ 3.40282347e+38F
#define _FLT_MIN_ 1.17549435e-38F

#ifndef uint
using uint = unsigned int;
#endif

#define MISS_HIT 0
#define DIFFUSE_HIT 1
#define SPECULAR_HIT 2
#define TRANSMIT_HIT 3
static __forceinline__ __device__ void* unpackPointer( unsigned int i0, unsigned int i1 )
{
    const unsigned long long uptr = static_cast<unsigned long long>( i0 ) << 32 | i1;
    void*           ptr = reinterpret_cast<void*>( uptr );
    return ptr;
}


static __forceinline__ __device__ void  packPointer( void* ptr, unsigned int& i0, unsigned int& i1 )
{
    const unsigned long long uptr = reinterpret_cast<unsigned long long>( ptr );
    i0 = uptr >> 32;
    i1 = uptr & 0x00000000ffffffff;
}

namespace rtgems {

    constexpr float origin()      { return 1.0f / 32.0f; }
    constexpr float int_scale()   { return 256.0f; }
    constexpr float float_scale() { return 1.0f / 65536.0f; }
    
    // Normal points outward for rays exiting the surface, else is flipped.
    static __inline__ __device__ float3 offset_ray(const float3 p, const float3 n)
    {
        int3 of_i {
            (int)(int_scale() * n.x),
            (int)(int_scale() * n.y), 
            (int)(int_scale() * n.z) };

        float3 p_i {
            __int_as_float(__float_as_int(p.x) + ((p.x < 0) ? -of_i.x : of_i.x)),
            __int_as_float(__float_as_int(p.y) + ((p.y < 0) ? -of_i.y : of_i.y)),
            __int_as_float(__float_as_int(p.z) + ((p.z < 0) ? -of_i.z : of_i.z)) };

        return float3{
                fabsf(p.x) < origin() ? p.x+float_scale()*n.x : p_i.x,
                fabsf(p.y) < origin() ? p.y+float_scale()*n.y : p_i.y,
                fabsf(p.z) < origin() ? p.z+float_scale()*n.z : p_i.z };
    }
}

enum medium{
    vacum,
    isotropicScatter
};

struct RadiancePRD
{
    // TODO: move some state directly into payload registers?
    float3       radiance;
    float3       radiance_d;
    float3       radiance_s;
    float3       radiance_t;
    float3       emission;
    float3       attenuation;
    float3       attenuation2;
    float3       origin;
    float3       direction;
    float        minSpecRough;
    bool         passed;
    bool         next_ray_is_going_inside;
    float        opacity;
    float        prob;
    float        prob2;
    unsigned int seed;
    unsigned int eventseed;
    unsigned int flags;
    bool         hitEnv;
    int          countEmitted;
    int          done;
    int          pad;
    float3       shadowAttanuation;
    int          medium;
    float        scatterDistance;
    float        scatterPDF;
    float        maxDistance;
    int          depth;
    int          diffDepth;
    bool         isSS;
    float        scatterStep;
    int          nonThinTransHit;
    float3       LP;
    float3       Ldir;
    float        Lweight;
    vec3         sigma_t_queue[8];
    vec3         ss_alpha_queue[8];
    int          curMatIdx;
    float        samplePdf;
    bool         fromDiff;

    unsigned char first_hit_type;
    vec3 extinction() {
        auto idx = clamp(curMatIdx, 0, 7);
        return sigma_t_queue[idx];
    }
    float        CH;

    //cihou SS
    vec3 sigma_t;
    vec3 ss_alpha;

    vec3 sigma_s() {
        return sigma_t * ss_alpha;
    }
    vec3 channelPDF;

    bool trace_denoise_albedo = false;
    bool trace_denoise_normal = false;
    float3 tmp_albedo {};
    float3 tmp_normal {};

    // cihou nanovdb
    float vol_t0=0, vol_t1=0;

    bool test_distance = false;
    bool origin_inside_vdb = false;
    bool surface_inside_vdb = false; 

    float trace_tmin = 0;
    float3 geometryNormal;

    void offsetRay() {
        offsetRay(this->origin, this->direction);
    }

    void offsetRay(float3& P, const float3& new_dir) {
        bool forward = dot(geometryNormal, new_dir) > 0;
        P = rtgems::offset_ray(P, forward? geometryNormal:-geometryNormal);
    }

    void offsetUpdateRay(float3& P, float3 new_dir) {
        this->origin = P;
        this->direction = new_dir;
        offsetRay(this->origin, new_dir);
    }

    VisibilityMask _mask_ = EverythingMask;

    void updateAttenuation(float3& multiplier) {
        attenuation2 = attenuation;
        attenuation *= multiplier;
    }
    
    int pushMat(vec3 extinction, vec3 ss_alpha = vec3(-1.0f))
    {
        vec3 d = abs(sigma_t_queue[curMatIdx] - extinction);
        float c = dot(d, vec3(1,1,1));
        if(curMatIdx<7 && c > 1e-6f )
        {
            curMatIdx++;
            sigma_t_queue[curMatIdx] = extinction;
            ss_alpha_queue[curMatIdx] = ss_alpha;
            
        }

        return curMatIdx;
    }

    void readMat(vec3& sigma_t, vec3& ss_alpha) {

        auto idx = clamp(curMatIdx, 0, 7);

        sigma_t = sigma_t_queue[idx];
        ss_alpha = ss_alpha_queue[idx];
    }

    int popMat(vec3& sigma_t, vec3& ss_alpha)
    {
        curMatIdx = clamp(--curMatIdx, 0, 7);
        sigma_t = sigma_t_queue[curMatIdx];
        ss_alpha = ss_alpha_queue[curMatIdx];
        return curMatIdx;
    }
    
};


static __forceinline__ __device__ void  traceRadiance(
        OptixTraversableHandle handle,
        float3                 ray_origin,
        float3                 ray_direction,
        float                  tmin,
        float                  tmax,
        RadiancePRD*           prd
        )
{
    // TODO: deduce stride from num ray-types passed in params

    unsigned int u0, u1;
    packPointer( prd, u0, u1 );
    optixTrace(
            handle,
            ray_origin,
            ray_direction,
            tmin,
            tmax,
            0.0f,                // rayTime
            OptixVisibilityMask( 1 | 2 ),
            OPTIX_RAY_FLAG_NONE,
            RAY_TYPE_RADIANCE,        // SBT offset
            RAY_TYPE_COUNT,           // SBT stride
            RAY_TYPE_RADIANCE,        // missSBTIndex
            u0, u1 );
}


static __forceinline__ __device__ bool traceOcclusion(
        OptixTraversableHandle handle,
        float3                 ray_origin,
        float3                 ray_direction,
        float                  tmin,
        float                  tmax,
        RadiancePRD*           prd
        )
{
    unsigned int u0, u1;
    packPointer( prd, u0, u1 );
    optixTrace(
            handle,
            ray_origin,
            ray_direction,
            tmin,
            tmax,
            0.0f,                    // rayTime
            OptixVisibilityMask( 1 | 2 ),
            OPTIX_RAY_FLAG_ENFORCE_ANYHIT,
            RAY_TYPE_OCCLUSION,      // SBT offset
            RAY_TYPE_COUNT,          // SBT stride
            RAY_TYPE_OCCLUSION,       // missSBTIndex
            u0, u1);
        return false;//???
}

static __forceinline__ __device__ void traceRadianceMasked(
	OptixTraversableHandle handle,
	float3                 ray_origin,
	float3                 ray_direction,
	float                  tmin,
	float                  tmax,
	char                   mask,
	RadiancePRD           *prd)
{
    unsigned int u0, u1;
    packPointer( prd, u0, u1 );

    optixTrace( handle,
            ray_origin, ray_direction,
            tmin, tmax,
            0.0f,                     // rayTime
            OptixVisibilityMask(mask),
            OPTIX_RAY_FLAG_DISABLE_ANYHIT,
            RAY_TYPE_RADIANCE,        // SBT offset
            RAY_TYPE_COUNT,           // SBT stride
            RAY_TYPE_RADIANCE,        // missSBTIndex
            u0, u1);
}


static __forceinline__ __device__ void traceOcclusionMasked(
        OptixTraversableHandle handle,
        float3                 ray_origin,
        float3                 ray_direction,
        float                  tmin,
        float                  tmax,
        char                   mask,
        RadiancePRD           *prd)
{
    unsigned int u0, u1;
    packPointer( prd, u0, u1 );

    optixTrace( handle,
            ray_origin, ray_direction,
            tmin, tmax,
            0.0f,  // rayTime
            OptixVisibilityMask(mask),
            OPTIX_RAY_FLAG_ENFORCE_ANYHIT,  //OPTIX_RAY_FLAG_NONE,
            RAY_TYPE_OCCLUSION,      // SBT offset
            RAY_TYPE_COUNT,          // SBT stride
            RAY_TYPE_OCCLUSION,      // missSBTIndex
            u0, u1);
}

static __forceinline__ __device__ RadiancePRD* getPRD()
{
    const unsigned int u0 = optixGetPayload_0();
    const unsigned int u1 = optixGetPayload_1();
    return reinterpret_cast<RadiancePRD*>( unpackPointer( u0, u1 ) );
}


extern "C" {
extern __constant__ Params params;
}


static __forceinline__ __device__ void setPayloadOcclusion( bool occluded )
{
    optixSetPayload_0( static_cast<unsigned int>( occluded ) );
}


struct Onb
{
  __forceinline__ __device__ Onb(const vec3& normal)
  {
    m_normal = normal;

    if( fabs(m_normal.x) > fabs(m_normal.z) )
    {
      m_binormal.x = -m_normal.y;
      m_binormal.y =  m_normal.x;
      m_binormal.z =  0;
    }
    else
    {
      m_binormal.x =  0;
      m_binormal.y = -m_normal.z;
      m_binormal.z =  m_normal.y;
    }

    m_binormal = normalize(m_binormal);
    m_tangent = cross( m_binormal, m_normal );
  }

  __forceinline__ __device__ void inverse_transform(vec3& p) const
  {
    p = p.x*m_tangent + p.y*m_binormal + p.z*m_normal;
  }

  vec3 m_tangent;
  vec3 m_binormal;
  vec3 m_normal;
};


static __forceinline__ __device__ void cosine_sample_hemisphere(const float u1, const float u2, vec3& p)
{
  // Uniformly sample disk.
  const float r   = sqrtf( u1 );
  const float phi = 2.0f*M_PIf * u2;
  p.x = r * cosf( phi );
  p.y = r * sinf( phi );

  // Project up to hemisphere.
  p.z = sqrtf( fmaxf( 0.0f, 1.0f - p.x*p.x - p.y*p.y ) );
}