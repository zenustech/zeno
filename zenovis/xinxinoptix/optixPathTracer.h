// this part of code is modified from nvidia's optix example
#pragma once
#include <optix.h>
#include <Shape.h>

#include "LightBounds.h"
// #include <nanovdb/NanoVDB.h>
#include <zeno/types/LightObject.h>

enum RayType
{
    RAY_TYPE_RADIANCE  = 0,
    RAY_TYPE_OCCLUSION = 1,
    RAY_TYPE_COUNT
};

enum VisibilityMask {
    NothingMask    = 0u,
    DefaultMatMask = 1u << 0,
    VolumeMatMask  = 1u << 1,
    LightMatMask   = 1u << 2,
    EverythingMask = 255u
}; 

struct GenericLight
{
    float3 T, B, N;
    float3 emission;

    zeno::LightType type {};
    zeno::LightShape shape{};
    uint8_t config = zeno::LightConfigNull;

    unsigned long long ies=0u;
    cudaTextureObject_t tex{};
    union {
        RectShape rect;
        SphereShape sphere;

        ConeShape cone;
        PointShape point;
    };

    bool isDeltaLight() {
        if (type == zeno::LightType::Direction || type == zeno::LightType::IES)
            return true;
        else 
            return false;
    }

    pbrt::LightBounds bounds() {

        auto Phi = dot(emission, make_float3(1.0f/3.0f));
        bool doubleSided = config & zeno::LightConfigDoubleside;

        if (this->type == zeno::LightType::IES) {
            return  this->cone.BoundAsLight(Phi, false);
        }
        
        switch (this->shape) {
        case zeno::LightShape::Plane:
            return this->rect.BoundAsLight(Phi, doubleSided);
        case zeno::LightShape::Sphere:
            return this->sphere.BoundAsLight(Phi, false);
        case zeno::LightShape::Point:
            return this->point.BoundAsLight(Phi, false);
        }

        return pbrt::LightBounds();
    }

    void setConeData(const float3& p, const float3& dir, float range, float coneAngle) {
        this->cone.p = p;
        this->cone.dir = dir;
        this->cone.range = range;
        this->cone.cosFalloffStart = cosf(coneAngle);
        this->cone.cosFalloffEnd = cosf(coneAngle + __FLT_EPSILON__);
    }

    void setRectData(const float3& v0, const float3& v1, const float3& v2, const float3& normal) {
        this->rect.v0 = v0;
        this->rect.v1 = v1;
        this->rect.v2 = v2;

        this->rect.normal = normal;
        this->rect.area = length( cross(v1, v2) );
    }

    void setSphereData(const float3& center, float radius) {
        this->sphere.center = center;
        this->sphere.radius = radius;
        this->sphere.area = M_PIf * 4 * radius * radius;
    }
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
    float4*      accum_buffer_D;
    float4*      accum_buffer_S;
    float4*      accum_buffer_T;
    float4*      accum_buffer_B;
    uchar4*      frame_buffer;
    uchar4*      frame_buffer_D;
    uchar4*      frame_buffer_S;
    uchar4*      frame_buffer_T;
    uchar4*      frame_buffer_B;

    float3*      debug_buffer;
    float3*      albedo_buffer;
    float3*      normal_buffer;

    unsigned int width;
    unsigned int height;
    unsigned int samples_per_launch;

    CameraInfo cam;

    uint32_t num_lights;
    GenericLight *lights;
    uint32_t firstRectLightIdx;
    uint32_t firstSphereLightIdx;

    float skyLightProbablity() {

        static float DefaultSkyLightProbablity = 0.5f;

        if (sky_strength <= 0.0f)
            return -0.0f;

        if (sky_texture == 0llu || skycdf == nullptr) 
            return -0.0f;

        return this->num_lights>0? DefaultSkyLightProbablity : 1.0f;
    }

    unsigned long long lightTreeSampler;

    OptixTraversableHandle handle;

    int usingHdrSky;
    cudaTextureObject_t sky_texture;

    float* skycdf;
    int* sky_start;
    int2 windowCrop_min;
    int2 windowCrop_max;
    int2 windowSpace;

    uint32_t skynx;
    uint32_t skyny;
    float envavg;

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

#if defined (__cudacc__)
    const bool denoise;
#else
    bool denoise;
#endif

    bool show_background;

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
    //float4* vertices;
    float4* uv;
    float4* nrm;
    float4* clr;
    float4* tan;
    unsigned short* lightMark;
    int* meshIdxs;
    
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
