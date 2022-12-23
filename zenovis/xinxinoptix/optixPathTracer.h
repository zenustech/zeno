// this part of code is modified from nvidia's optix example
#pragma once

#ifndef __CUDACC_RTC__ 
    #include "optixVolume.h"
#else
    #include "volume.h"
#endif

enum RayType
{
    RAY_TYPE_RADIANCE  = 0,
    RAY_TYPE_OCCLUSION = 1,
    RAY_TYPE_COUNT
};


struct ParallelogramLight
{
    float3 corner;
    float3 v1, v2;
    float3 normal;
    float3 emission;
    float  cdf;
};


struct CameraInfo
{
    float3 eye;
    float3 right, up, front;
    //float aspect;
    //float fov;
    float focalPlaneDistance;
    float aperture;
};

const unsigned int common_object_mask = 1;
const unsigned int volume_object_mask = 2;

struct Params
{
    unsigned int subframe_index;
    float4*      accum_buffer;
    uchar4*      frame_buffer;
    unsigned int width;
    unsigned int height;
    unsigned int samples_per_launch;

    CameraInfo cam;

    unsigned int num_lights;
    ParallelogramLight     *lights;
    OptixTraversableHandle handle;

    int usingHdrSky;
    cudaTextureObject_t sky_texture;
    float sky_rot;
    float sky_strength;

    float sunLightDirX;
    float sunLightDirY;
    float sunLightDirZ;

    float windDirX;
    float windDirY;
    float windDirZ;

    float sunSoftness;
    float elapsedTime;
};


struct RayGenData
{
};


struct MissData
{
    float4 bg_color;
};


struct HitGroupData
{
    float4* vertices;
    float4* uv;
    float4* nrm;
    float4* clr;
    float4* tan;
    unsigned short* lightMark;
    float4* uniforms;
    cudaTextureObject_t textures[32];

    // cihou nanovdb
    float opacityHDDA;

    void* density_grid; 
    void* temperature_grid; 
    
    float density_max;
    float temperature_max;

    float3 colorVDB;

    float sigma_a, sigma_s;
    float greenstein; // -1 ~ 1

    #ifdef __CUDACC_RTC__ 

    __device__ float sigma_t() {
        return sigma_a + sigma_s;
    }

    #endif
};
