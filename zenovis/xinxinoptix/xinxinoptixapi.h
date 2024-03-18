#pragma once
#include <functional>
#include <memory>
#include <string>
#include <vector>
#include <map>
#include <set>

#include "optixSphere.h"
#include "zeno/utils/vec.h"

enum ShaderMaker {
    Mesh = 0,
    Sphere = 1,
    Volume = 2,
};

struct ShaderPrepared {
    ShaderMaker mark;
    std::string matid;
    std::string filename;

    std::string callable;
    std::string parameters;
    
    std::vector<std::string> tex_names;
};

namespace xinxinoptix {

std::set<std::string> uniqueMatsForMesh();

void optixcleanup();
void optixrender(int fbo = 0, int samples = 1, bool denoise = false, bool simpleRender = false);
void *optixgetimg(int &w, int &h);
void optixinit(int argc, char* argv[]);
void optixupdatebegin();
void UpdateDynamicMesh(std::map<std::string, int> const &mtlidlut);
void UpdateStaticMesh(std::map<std::string, int> const &mtlidlut);
void UpdateInst();
void UpdateStaticInstMesh(const std::map<std::string, int> &mtlidlut);
void UpdateDynamicInstMesh(const std::map<std::string, int> &mtlidlut);
void CopyInstMeshToGlobalMesh();
void UpdateMeshGasAndIas(bool staticNeedUpdate);
void optixupdatematerial(std::vector<std::shared_ptr<ShaderPrepared>> &shaders);

void updateSphereXAS();

void updateVolume(uint32_t volume_shader_offset);
void updateRootIAS();
void buildLightTree();
void optixupdateend();

void set_window_size(int nx, int ny);
void set_outside_random_number(int32_t outside_random_number);
void set_perspective(float const *U, float const *V, float const *W, float const *E, float aspect, float fov, float fpd, float aperture);
void set_physical_camera_param(float aperture, float shutter_speed, float iso, bool aces, bool exposure);
void set_perspective_by_fov(float const *U, float const *V, float const *W, float const *E, float aspect, float fov, int fov_type, float L, float focal_distance, float aperture, float pitch, float yaw, float h_shift, float v_shift);
void set_perspective_by_focal_length(float const *U, float const *V, float const *W, float const *E, float aspect, float focal_length, float w, float h, float focal_distance, float aperture, float pitch, float yaw, float h_shift, float v_shift);
void load_object(std::string const &key, std::string const &mtlid, const std::string& instID, float const *verts, size_t numverts, int const *tris, size_t numtris, std::map<std::string, std::pair<float const *, size_t>> const &vtab,int const *matids, std::vector<std::string> const &matNameList);
void unload_object(std::string const &key);
void load_inst(const std::string &key, const std::string &instID, const std::string &onbType, std::size_t numInsts, const float *pos, const float *nrm, const float *uv, const float *clr, const float *tang);
void unload_inst(const std::string &key);

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
// void optixUpdateUniforms(std::vector<float4> & inConstants);
void optixUpdateUniforms(void *inConstants, std::size_t size);
}
