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

enum medium{
    vacum,
    isotropicScatter
};

struct VolumePRD {
    float t0;
    float t1;

    bool afterSingleScatter = false;
};

struct ShadowPRD {
    bool test_distance;
    float maxDistance;
    uint32_t lightIdx = UINT_MAX;

    float3 origin;
    uint32_t seed;
    float3 attanuation;
    uint8_t nonThinTransHit;

    VolumePRD vol;

    float rndf() {
        return rnd(seed);
    }
};

struct RadiancePRD
{
    bool test_distance;
    float maxDistance;
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
    float        prob;
    float        prob2;
    unsigned int seed;
    unsigned int eventseed;
    unsigned int flags;
    int          countEmitted;
    int          done;

    int          medium;
    float        scatterDistance;
    float        scatterPDF;
    int          depth;
    int          diffDepth;
    bool         isSS;
    float        scatterStep;
    float        pixel_area;
    float        Lweight;
    vec3         sigma_t_queue[8];
    vec3         ss_alpha_queue[8];
    int          curMatIdx;
    float        samplePdf;
    bool         fromDiff;
    unsigned char adepth;
    bool         alphaHit;
    vec3         mask_value;
    unsigned char max_depth;

    uint16_t lightmask = EverythingMask;

    __forceinline__ float rndf() {
        return rnd(this->seed);
        //return pcg_rng(this->seed); 
    }

    unsigned char hit_type;
    vec3 extinction() {
        auto idx = clamp(curMatIdx, 0, 7);
        return sigma_t_queue[idx];
    }

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
    VolumePRD vol;

    float _tmin_ = 0;
    float3 geometryNormal;

    void offsetRay() {
        offsetRay(this->origin, this->direction);
    }

    void offsetRay(float3& P, const float3& new_dir) {
        bool forward = dot(geometryNormal, new_dir) > 0;
        auto dir = forward? geometryNormal:-geometryNormal;
        auto offset = rtgems::offset_ray(P, dir);
        P = offset;
//        float l = length( offset - P );
//        float l2 = this->alphaHit? max(l, 1e-4) : max(l, 1e-5);
//        P = P + l2 * dir;
    }

    void offsetUpdateRay(float3& P, float3 new_dir) {
//      double x = (double)(P.x);
//      double y = (double)(P.y);
//      double z = (double)(P.z);
//        auto beforeOffset = make_float3(x, y, z);
        //this->origin = P;
        //this->direction = new_dir;
        offsetRay(P, new_dir);
//        double x2 = (double)(beforeOffset.x);
//        double y2 = (double)(beforeOffset.y);
//        double z2 = (double)(beforeOffset.z);
//        this->origin = make_float3(x2, y2, z2);
    }

    uint8_t _mask_ = EverythingMask;

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
