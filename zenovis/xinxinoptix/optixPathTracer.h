// this part of code is modified from nvidia's optix example
#pragma once

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
};


struct CameraInfo
{
    float3 eye;
    float3 right, up, front;
    //float aspect;
    //float fov;
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
    unsigned short* lightMark;
    float* uniforms;//TODO
    cudaTextureObject_t textures[32];
};
