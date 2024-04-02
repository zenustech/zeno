#pragma once
#define USE_SHORT 1
#include <optix.h>
#include <Shape.h>

#include "LightBounds.h"
// #include <nanovdb/NanoVDB.h>
#include <zeno/types/LightObject.h>

#define TRI_PER_MESH (1<<29) //2^29

enum RayType
{
    RAY_TYPE_RADIANCE  = 0,
    RAY_TYPE_OCCLUSION = 1,
    RAY_TYPE_COUNT
};

enum VisibilityMask {
    NothingMask    = 0u,
    DefaultMatMask = 1u << 0,
    LightMatMask   = 1u << 1,
    VolumeMatMask  = 1u << 2,
    EverythingMask = 255u
}; 

struct GenericLight
{
    float3 T, B, N;
    float3 color;
    float intensity;
    float vIntensity;

    float spreadMajor;
    float spreadMinor;
    float spreadNormalize;
    float maxDistance;
    float falloffExponent;

    uint16_t mask = EverythingMask;

    zeno::LightType type {};
    zeno::LightShape shape{};
    uint8_t config = zeno::LightConfigNull;

    unsigned long long ies=0u;
    cudaTextureObject_t tex{};
    float texGamma = 1.0f;
    union {
        RectShape rect;
        SphereShape sphere;

        ConeShape cone;
        PointShape point;
        TriangleShape triangle;
    };

    bool isDeltaLight() {
        if (type == zeno::LightType::Direction || type == zeno::LightType::IES)
            return true;
        else 
            return false;
    }

    pbrt::LightBounds bounds() {

        auto Phi = intensity * dot(color, make_float3(1.f/3.f));
        bool doubleSided = config & zeno::LightConfigDoubleside;

        if (this->type == zeno::LightType::IES) {
            return this->cone.BoundAsLight(Phi, false);
        }

        if (this->type == zeno::LightType::Spot) {
            return this->cone.BoundAsLight(Phi, false);
        }
        
        switch (this->shape) {
        case zeno::LightShape::Plane:
        case zeno::LightShape::Ellipse:
            return this->rect.BoundAsLight(Phi, doubleSided);
        case zeno::LightShape::Sphere:
            return this->sphere.BoundAsLight(Phi, false);
        case zeno::LightShape::Point:
            return this->point.BoundAsLight(Phi, false);
        case zeno::LightShape::TriangleMesh:
            return this->triangle.BoundAsLight(Phi, false);
        }

        return pbrt::LightBounds();
    }

    void setConeData(const float3& p, const float3& dir, float range, float spreadAngle, float falloffAngle=0.0f) {
        this->cone.p = p;
        this->cone.range = range;

        this->cone.dir = dir;
        this->cone.cosFalloffStart = cosf(spreadAngle - falloffAngle);
        this->cone.cosFalloffEnd = cosf(spreadAngle);
    }

    void setRectData(const float3& v0, const float3& v1, const float3& v2, const float3& normal) {

        rect.v = v0;
        rect.lenX = length(v1);
        rect.axisX = v1 / rect.lenX;
        
        rect.lenY = length(v2);
        rect.axisY = v2 / rect.lenY;

        rect.normal = normal;
         //length( cross(v1, v2) );
        rect.area = rect.lenX * rect.lenY; 
    }

    void setSphereData(const float3& center, float radius) {
        this->sphere.center = center;
        this->sphere.radius = radius;
        this->sphere.area = M_PIf * 4 * radius * radius;
    }

    void setTriangleData(const float3& p0, const float3& p1, const float3& p2, const float3& normal, uint32_t coordsBufferOffset, uint32_t normalBufferOffset) {
        this->triangle.p0 = p0;
        this->triangle.p1 = p1;
        this->triangle.p2 = p2;

        this->triangle.coordsBufferOffset = coordsBufferOffset;
        this->triangle.normalBufferOffset = normalBufferOffset;
        
        this->triangle.faceNormal = normal;
        this->triangle.area = this->triangle.Area();
    }
};


struct CameraInfo
{
    //all distance in meters;
    float3 eye; //lens center position
    float3 right;   //lens right direction
    float3 front;   //lens front direction, so call optical axis
    float3 up;  //lens up direction
    float horizontal_shift;
    float vertical_shift;
    float pitch;
    float yaw;
    float focal_length;    //lens focal length
    float aperture;     //diameter of aperture
    float focal_distance;   //distance from focal plane center to lens plane
    float width;    //sensor physical width
    float height;   //sensor physical height
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
    float3*      frame_buffer_C;
    float3*      frame_buffer_D;
    float3*      frame_buffer_S;
    float3*      frame_buffer_T;
    float3*      frame_buffer_B;
    float3*      frame_buffer_M;

    float3*      debug_buffer;
    float3*      albedo_buffer;
    float3*      normal_buffer;

    unsigned int width;
    unsigned int height;
    unsigned int tile_i;
    unsigned int tile_j;
    unsigned int tile_w;
    unsigned int tile_h;
    unsigned int samples_per_launch;

    CameraInfo cam;

    uint32_t num_lights;
    GenericLight *lights;
    uint32_t firstRectLightIdx;
    uint32_t firstSphereLightIdx;
    uint32_t firstTriangleLightIdx;

    uint32_t maxInstanceID;

    unsigned long long lightTreeSampler;
    unsigned long long triangleLightCoordsBuffer;
    unsigned long long triangleLightNormalBuffer;

    float skyLightProbablity() {

        if (sky_strength <= 0.0f)
            return -0.0f;

        if (sky_texture == 0llu || skycdf == nullptr) 
            return -0.0f;

        static const float DefaultSkyLightProbablity = 0.5f;
        return this->num_lights>0? DefaultSkyLightProbablity : 1.0f;
    }

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

    int32_t outside_random_number;
    bool simpleRender     :1;
    bool show_background  :1;

    bool denoise : 1;

    float physical_camera_aperture;
    float physical_camera_shutter_speed;
    float physical_camera_iso;
    bool  physical_camera_aces;
    bool  physical_camera_exposure;
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
    uint16_t dc_index;
    uint16_t vol_depth=999;
    float vol_extinction=1.0f;
    
    bool equiangular  = false;
    bool multiscatter = false;

    //float4* vertices;
#ifdef USE_SHORT_COMPACT
    ushort2* uv;
    ushort2* nrm;
    ushort2* clr;
    ushort2* tan;
#else

  #ifdef USE_SHORT
      ushort3* uv;
      ushort3* nrm;
      ushort3* clr;
      ushort3* tan;
  #else
      float4* uv;
      float4* nrm;
      float4* clr;
      float4* tan;
  #endif

#endif
    unsigned short* lightMark;
    uint32_t* auxOffset;
#ifdef USE_SHORT
    ushort3* instPos;
    ushort3* instNrm;
    ushort3* instUv;
    ushort3* instClr;
    ushort3* instTang;
#else
    float3* instPos;
    float3* instNrm;
    float3* instUv;
    float3* instClr;
    float3* instTang;
#endif
    float4* uniforms;
    cudaTextureObject_t textures[32];

    unsigned long long vdb_grids[8];
    float vdb_max_v[8];
};
