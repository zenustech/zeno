#pragma once

#include <optix.h>
#include <optix_device.h>
#include "zxxglslvec.h"
#include "optixPathTracer.h"

#include <Sampling.h>
#include <cuda/random.h>
#include <cuda/climits.h>

#define _FLT_MAX_ __FLT_MAX__
#define _FLT_MIN_ __FLT_MIN__
#define _FLT_EPL_ __FLT_EPSILON__

#ifndef __CUDACC_RTC__
#include "Host.h"
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

enum medium{
    vacum,
    isotropicScatter
};

struct VolumePRD {
    float t0;
    float t1;
};

struct ShadowPRD {
    bool test_distance;
    float maxDistance;
    uint32_t lightIdx = UINT_MAX;
    
    float3 origin;
    uint32_t seed;
    float3 attanuation;
    
    uint8_t depth;
    uint8_t nonThinTransHit;

    VolumePRD vol;
    float3 ShadowNormal;

    float rndf() {
        return rnd(seed);
    }
};

struct RadiancePRD
{

    bool test_distance;
    float maxDistance;

    //zxx seed
    unsigned int offset = 0;
    unsigned int offset2 = 0;
    
    float3       radiance;
    float3       aov[3];
    float3       emission;
    float3       attenuation;
    float3       origin;
    float3       direction;

    float3 tmp_albedo {};
    float3 tmp_normal {};
    
    float        minSpecRough;

    unsigned int seed;
    unsigned int eventseed;

    float        scatterDistance;
    float        scatterPDF;
    float        scatterStep;

    float        pixel_area;

    float        samplePdf;
    vec3         mask_value;
    
    uint16_t lightmask = EverythingMask;
    uint4 record;

    uint8_t      depth;
    uint8_t      max_depth;
    uint8_t      diffDepth;

    bool done         : 1;
    bool countEmitted : 1;

    bool isSS         : 1;
    bool alphaHit     : 1;
    bool fromDiff     : 1; 
    bool denoise      : 1;
    uint8_t hit_type  : 4;

    uint8_t _mask_ = EverythingMask;
    float   _tmin_ = 0.0f;
    float   _tmax_ = FLT_MAX;

    //cihou SS
    vec3 sigma_t;
    vec3 ss_alpha;

    uint8_t medium;
    uint8_t curMatIdx;

    half3 sigma_t_queue[8];
    half3 ss_alpha_queue[8];
    
    vec3 channelPDF;

    // cihou nanovdb
    VolumePRD vol;
    float3 geometryNormal;

    __device__ __forceinline__ float rndf() {
        return rnd(this->seed);
        //return pcg_rng(this->seed); 
    }
    __device__ __forceinline__ float vdcrndf() {
        return vdcrnd(this->offset2);
    }

    __device__ __forceinline__ vec3 sigma_s() {
        return sigma_t * ss_alpha;
    }

    __device__ __forceinline__ void updateAttenuation(float3& multiplier) {
        attenuation *= multiplier;
    }

    __device__ __forceinline__ vec3 extinction() {
        auto idx = min(curMatIdx, 7);
        return half3_to_float3(sigma_t_queue[idx]);
    }

    __device__ int pushMat(vec3 extinction, vec3 ss_alpha = vec3(-1.0f))
    {
        auto cached = this->extinction();
        vec3 d = abs(cached - extinction);
        float c = dot(d, vec3(1,1,1));
        if(curMatIdx<7 && c > 1e-6f )
        {
            curMatIdx++;
            sigma_t_queue[curMatIdx] = float3_to_half3(extinction);
            ss_alpha_queue[curMatIdx] = float3_to_half3(ss_alpha);
        }
        return curMatIdx;
    }

    __device__ void readMat(vec3& sigma_t, vec3& ss_alpha) {

        auto idx = min(curMatIdx, 7);

        sigma_t = half3_to_float3(sigma_t_queue[idx]);
        ss_alpha = half3_to_float3(ss_alpha_queue[idx]);
    }

    __device__ int popMat(vec3& sigma_t, vec3& ss_alpha)
    {
        curMatIdx = min(--curMatIdx, 7);
        sigma_t = half3_to_float3(sigma_t_queue[curMatIdx]);
        ss_alpha = half3_to_float3(ss_alpha_queue[curMatIdx]);
        return curMatIdx;
    }
};

static __forceinline__ __device__ void traceRadiance(
	OptixTraversableHandle handle,
	float3                 ray_origin,
	float3                 ray_direction,
	float                  tmin,
	float                  tmax,
	void                   *prd,
    OptixVisibilityMask    mask=255u)
{
    unsigned int u0, u1;
    packPointer( prd, u0, u1 );

    optixTrace( handle,
            ray_origin, ray_direction,
            tmin, tmax,
            0.0f,                     // rayTime
            (mask),
            OPTIX_RAY_FLAG_DISABLE_ANYHIT,
            RAY_TYPE_RADIANCE,        // SBT offset
            RAY_TYPE_COUNT,           // SBT stride
            RAY_TYPE_RADIANCE,        // missSBTIndex
            u0, u1);
}

static __forceinline__ __device__ bool traceShadowCheap(
        OptixTraversableHandle handle,
        float3                 ray_origin,
        float3                 ray_direction,
        float                  tmin,
        float                  tmax,
        void                   *prd,
        OptixVisibilityMask    mask=255u) 
{
    unsigned int u0, u1;
    packPointer( prd, u0, u1 );

    optixTraverse(handle,
            ray_origin, ray_direction,
            tmin, tmax,
            0.0f,  // rayTime
            (mask),
            OPTIX_RAY_FLAG_FORCE_OPACITY_MICROMAP_2_STATE,
            RAY_TYPE_RADIANCE,      // SBT offset
            RAY_TYPE_COUNT,          // SBT stride
            RAY_TYPE_RADIANCE,      // missSBTIndex
            u0, u1);
    
    return !optixHitObjectIsMiss() ;   
}

static __forceinline__ __device__ void traceOcclusion(
        OptixTraversableHandle handle,
        float3                 ray_origin,
        float3                 ray_direction,
        float                  tmin,
        float                  tmax,
        void                   *prd,
        OptixVisibilityMask    mask=255u)
{
    unsigned int u0, u1;
    packPointer( prd, u0, u1 );

    optixTrace( handle,
            ray_origin, ray_direction,
            tmin, tmax,
            0.0f,  // rayTime
            (mask),
            OPTIX_RAY_FLAG_ENFORCE_ANYHIT,
            RAY_TYPE_OCCLUSION,      // SBT offset
            RAY_TYPE_COUNT,          // SBT stride
            RAY_TYPE_OCCLUSION,      // missSBTIndex
            u0, u1);
}

template <typename TypePRD = RadiancePRD>
static __forceinline__ __device__ TypePRD* getPRD()
{
    const unsigned int u0 = optixGetPayload_0();
    const unsigned int u1 = optixGetPayload_1();
    return reinterpret_cast<TypePRD*>( unpackPointer( u0, u1 ) );
}


extern "C" {
extern __constant__ Params params;
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
