#pragma once
#include <functional>
#include <memory>
#include <string>
#include <vector>
#include <map>
#include <set>

#include "optixSphere.h"

enum ShaderMaker {
    Mesh = 0,
    Sphere = 1,
    Volume = 2,
};

struct ShaderPrepared {
    ShaderMaker mark;
    std::string matid;
    std::string source;
    std::vector<std::string> tex_names;

    std::shared_ptr<std::string> fallback;
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
void buildRootIAS();
void buildLightTree();
void optixupdateend();

void set_window_size(int nx, int ny);
void set_perspective(float const *U, float const *V, float const *W, float const *E, float aspect, float fov, float fpd, float aperture);

void load_object(std::string const &key, std::string const &mtlid, const std::string& instID, float const *verts, size_t numverts, int const *tris, size_t numtris, std::map<std::string, std::pair<float const *, size_t>> const &vtab,int const *matids, std::vector<std::string> const &matNameList);
void unload_object(std::string const &key);
void load_inst(const std::string &key, const std::string &instID, const std::string &onbType, std::size_t numInsts, const float *pos, const float *nrm, const float *uv, const float *clr, const float *tang);
void unload_inst(const std::string &key);

void load_triangle_light(std::string const &key, 
                        const float *v0, const float *v1, const float *v2, 
                        const float *n0, const float *n1, const float *n2,
                        const float *uv0, const float *uv1, const float *uv2,
                        float const *nor, float const *emi, float intensity,
                        bool visible, bool doubleside, float vIntensity, int shape, int type,
                        std::string& profileKey, std::string& textureKey, float gamma);
                        
void load_light(std::string const &key, float const*v0, float const*v1, float const*v2, 
                float const*nor, float const*emi, float intensity,
                bool visible, bool doubleside, float vIntensity, int shape, int type, 
                std::string& profileKey, std::string& textureKey, float gamma);
                
void unload_light();
void update_procedural_sky(zeno::vec2f sunLightDir, float sunLightSoftness, zeno::vec2f windDir, float timeStart, float timeSpeed,
                           float sunLightIntensity, float colorTemperatureMix, float colorTemperature);
void update_hdr_sky(float sky_rot, zeno::vec3f sky_rot3d, float sky_strength);
void using_hdr_sky(bool enable);
void show_background(bool enable);
// void optixUpdateUniforms(std::vector<float4> & inConstants);
void optixUpdateUniforms(void *inConstants, std::size_t size);
}
