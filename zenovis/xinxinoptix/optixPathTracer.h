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

enum VisibilityMask {
    NothingMask = 0u,
    DefaultMatMask = 1u,
    VolumeMatMask = 2u,
    EverythingMask = 255u
}; 

enum RayLaunchSource {
    DefaultMatSource = 0u,
    VolumeEdgeSource = 1u,
    VolumeEmptySource = 1u << 1,
    VolumeScatterSource = 1u << 2
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

    float* skycdf;
    int* sky_start;


    int skynx;
    int skyny;

    float sky_rot;
    float sky_rot_x;
    float sky_rot_y;
    float sky_rot_z;
    float sky_strength;

    float sunLightDirX;
    float sunLightDirY;
    float sunLightDirZ;
    float sunLightIntensity;
    float colorTemperatureMix;
    float colorTemperature;

    float windDirX;
    float windDirY;
    float windDirZ;

    float sunSoftness;
    float elapsedTime;
    bool simpleRender;
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
    int* meshIdxs;
    float* meshMats;
    float3* instPos;
    float3* instNrm;
    float3* instUv;
    float3* instClr;
    float3* instTang;
    float4* uniforms;
    cudaTextureObject_t textures[32];

    unsigned long long vdb_grids[8];
    float vdb_max_v[8];
};
