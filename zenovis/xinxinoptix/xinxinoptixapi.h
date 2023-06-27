#pragma once
#include <functional>
#include <string>
#include <vector>
#include <map>
#include <set>

#include <zeno/utils/vec.h>
#include <glm/matrix.hpp>

enum ShaderMaker {
    Mesh = 0,
    Sphere = 1,
    Volume = 2,
};

struct ShaderPrepared {
    ShaderMaker mark;
    std::string material;
    std::string source;
    std::vector<std::string> tex_names;
};

namespace xinxinoptix {

struct InfoSphereTransformed {
    std::string materialID;
    std::string instanceID;
    
    glm::mat4 optix_transform;
    //Draw uniform sphere with transform
};

inline std::map<std::string, InfoSphereTransformed> LutSpheresTransformed;
void preload_sphere_transformed(std::string const &key, std::string const &mtlid, const std::string &instID, const glm::mat4& transform);

struct InfoSpheresCrowded {
    uint32_t sbt_count = 0;
    std::set<std::string> cached;
    std::set<std::string> mtlset;
    std::vector<std::string> mtlid_list{};
    std::vector<uint32_t>    sbtoffset_list{};

    std::vector<std::string> instid_list{};

    std::vector<float> radius_list{};
    std::vector<zeno::vec3f> center_list{};
}; 

inline InfoSpheresCrowded SpheresCrowded;

struct SphereInstanceGroupBase {
    std::string key;
    std::string instanceID;
    std::string materialID;

    zeno::vec3f center{};
    float radius{};
};

inline std::map<std::string, SphereInstanceGroupBase> SpheresInstanceGroupMap;

void preload_sphere_crowded(std::string const &key, std::string const &mtlid, const std::string &instID, const float &radius, const zeno::vec3f &center );
void foreach_sphere_crowded(std::function<void( const std::string &mtlid, std::vector<uint32_t> &sbtoffset_list)> func);

void cleanupSpheres();

std::set<std::string> uniqueMatsForMesh();
std::set<std::string> uniqueMatsForSphere();

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
void UpdateGasAndIas(bool staticNeedUpdate);
void optixupdatematerial(std::vector<ShaderPrepared>       &shaders);

void updateCrowdedSpheresGAS();
void updateUniformSphereGAS();
void updateInstancedSpheresGAS();

void updateVolume(uint32_t volume_shader_offset);
void optixupdatelight();
void optixupdateend();

void set_window_size(int nx, int ny);
void set_perspective(float const *U, float const *V, float const *W, float const *E, float aspect, float fov, float fpd, float aperture);

void load_object(std::string const &key, std::string const &mtlid, const std::string& instID, float const *verts, size_t numverts, int const *tris, size_t numtris, std::map<std::string, std::pair<float const *, size_t>> const &vtab,int const *matids, std::vector<std::string> const &matNameList);
void unload_object(std::string const &key);
void load_inst(const std::string &key, const std::string &instID, const std::string &onbType, std::size_t numInsts, const float *pos, const float *nrm, const float *uv, const float *clr, const float *tang);
void unload_inst(const std::string &key);
void load_light(std::string const &key, float const*v0,float const*v1,float const*v2, float const*nor,float const*emi );
void unload_light();
void update_procedural_sky(zeno::vec2f sunLightDir, float sunLightSoftness, zeno::vec2f windDir, float timeStart, float timeSpeed,
                           float sunLightIntensity, float colorTemperatureMix, float colorTemperature);
void update_hdr_sky(float sky_rot, zeno::vec3f sky_rot3d, float sky_strength);
void using_hdr_sky(bool enable);
void show_background(bool enable);
// void optixUpdateUniforms(std::vector<float4> & inConstants);
void optixUpdateUniforms(void *inConstants, std::size_t size);
}
