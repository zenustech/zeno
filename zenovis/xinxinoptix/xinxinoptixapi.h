#pragma once
#include <functional>
#include <memory>
#include <string>
#include <vector>
#include <map>
#include <set>

#include <glm/glm.hpp>
#include "optixSphere.h"
#include "zeno/utils/vec.h"
#include "zeno/types/CurveType.h"
#include "zeno/types/LightObject.h"

#include "Portal.h"
#include "OptiXStuff.h"

enum ShaderMark {
    Mesh, Sphere, Volume,

    CURVE_QUADRATIC,
    CURVE_RIBBON,
    CURVE_CUBIC,

    CURVE_LINEAR,
    CURVE_BEZIER,
    CURVE_CATROM,
};

typedef std::tuple<std::string, ShaderMark> shader_key_t;
struct ByShaderKey
{
    bool operator()(const shader_key_t& a, const shader_key_t& b) const
    {
        auto& [a1, a2] = a;
        auto& [b1, b2] = b;
    
        if (a2 < b2)
            return true;
        if (a2 > b2)
            return false;

        // if (a1=="Default")
        //     return true;
        if (b1 == "Default")
            return false;
        if (a1 == "Light")
            return false;
        if (a1 < b1 || b1=="Light")
            return true;
        else
            return false;
    
        return a1 < b1;
    }
};

static const std::map<zeno::CurveType, ShaderMark> CURVE_SHADER_MARK {
    { zeno::CurveType::QUADRATIC_BSPLINE, ShaderMark::CURVE_QUADRATIC },
    { zeno::CurveType::RIBBON_BSPLINE,    ShaderMark::CURVE_RIBBON    },
    { zeno::CurveType::CUBIC_BSPLINE,     ShaderMark::CURVE_CUBIC     },
    
    { zeno::CurveType::LINEAR,    ShaderMark::CURVE_LINEAR    },
    { zeno::CurveType::BEZIER,    ShaderMark::CURVE_BEZIER    },
    { zeno::CurveType::CATROM,    ShaderMark::CURVE_CATROM    },
};

struct ShaderPrepared {
    bool dirty = true;

    ShaderMark mark;
    std::string matid;
    std::string filename;

    std::string callable;
    std::string parameters;

    std::map<std::string, std::string> macros;
    
    std::vector<std::shared_ptr<OptixUtil::cuTexture>> texs;
    std::vector<std::string>       vdb_keys;
};

namespace xinxinoptix {


void optixCleanup();
void optixDestroy();

void optixrender(int fbo = 0, int samples = 1, bool denoise = false, bool simpleRender = false);
void *optixgetimg(int &w, int &h);
void optixinit(int argc, char* argv[]);
void optixupdatebegin();

void prepareScene();
void updateShaders(std::vector<std::shared_ptr<ShaderPrepared>> &shaders, 
                    bool requireTriangObj, bool requireTriangLight, 
                    bool requireSphereObj, bool requireSphereLight, 
                    bool requireVolumeObj, uint usesCurveTypes, bool refresh=false);
                    
void updateRootIAS();
void buildLightTree();
void configPipeline(bool dirty);

void set_window_size(int nx, int ny);
void set_outside_random_number(int32_t outside_random_number);
void set_perspective(float const *U, float const *V, float const *W, float const *E, float aspect, float fov, float fpd, float aperture);
void set_physical_camera_param(float aperture, float shutter_speed, float iso, bool aces, bool exposure, bool panorama_camera, bool panorama_vr180, float pupillary_distance);
void set_perspective_by_fov(float const *U, float const *V, float const *W, float const *E, float aspect, float fov, int fov_type, float L, float focal_distance, float aperture, float pitch, float yaw, float h_shift, float v_shift);
void set_perspective_by_focal_length(float const *U, float const *V, float const *W, float const *E, float aspect, float focal_length, float w, float h, float focal_distance, float aperture, float pitch, float yaw, float h_shift, float v_shift);

glm::vec3 get_click_pos(int x, int y);

struct LightDat {
    std::vector<float> v0;
    std::vector<float> v1;
    std::vector<float> v2;
    std::vector<float> normal;
    std::vector<float> color;

    float spreadMajor;
    float spreadMinor;
    float intensity;
    float vIntensity;
    float fluxFixed;
    float maxDistance;
    float falloffExponent;

    bool visible, doubleside;
    uint8_t shape, type;
    uint16_t mask;

    uint32_t coordsBufferOffset = UINT_MAX;
    uint32_t normalBufferOffset = UINT_MAX;

    std::string profileKey;
    std::string textureKey;
    float textureGamma;
};

void load_triangle_light(std::string const &key, LightDat &ld,
                        const zeno::vec3f &v0,  const zeno::vec3f &v1,  const zeno::vec3f &v2, 
                        const zeno::vec3f *pn0, const zeno::vec3f *pn1, const zeno::vec3f *pn2,
                        const zeno::vec3f *uv0, const zeno::vec3f *uv1, const zeno::vec3f *uv2);
                        
void load_light(std::string const &key, LightDat &ld, float const*v0, float const*v1, float const*v2);
                
void unload_light();
void update_procedural_sky(zeno::vec2f sunLightDir, float sunLightSoftness, zeno::vec2f windDir, float timeStart, float timeSpeed,
                           float sunLightIntensity, float colorTemperatureMix, float colorTemperature);
void update_hdr_sky(float sky_rot, zeno::vec3f sky_rot3d, float sky_strength);
void using_hdr_sky(bool enable);
void show_background(bool enable);

void updatePortalLights(const std::vector<Portal>& portals);
void updateDistantLights(std::vector<zeno::DistantLightData>& dldl);
// void optixUpdateUniforms(std::vector<float4> & inConstants);
void optixUpdateUniforms(void *inConstants, std::size_t size);

const std::map<std::string, LightDat> &get_lightdats();

}
