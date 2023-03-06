#pragma once
#include <string>
#include <vector>
#include <map>
#include <zeno/utils/vec.h>
#include <cuda_runtime.h>

namespace xinxinoptix {

void optixcleanup();
void optixrender(int fbo = 0, int samples = 1, bool simpleRender = false);
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
void optixupdatematerial(std::vector<std::string> const &shaders, std::vector<std::vector<std::string>> &texs);
void optixupdatelight();
void optixupdateend();

void set_window_size(int nx, int ny);
void set_perspective(float const *U, float const *V, float const *W, float const *E, float aspect, float fov, float fpd, float aperture);

void load_object(std::string const &key, std::string const &mtlid, const std::string& instID, float const *verts, size_t numverts, int const *tris, size_t numtris, std::map<std::string, std::pair<float const *, size_t>> const &vtab);
void unload_object(std::string const &key);
void load_inst(const std::string &key, const std::string &instID, const std::string &onbType, std::size_t numInsts, const float *pos, const float *nrm, const float *uv, const float *clr, const float *tang);
void unload_inst(const std::string &key);
void load_light(std::string const &key, float const*v0,float const*v1,float const*v2, float const*nor,float const*emi );
void unload_light();
void update_procedural_sky(zeno::vec2f sunLightDir, float sunLightSoftness, zeno::vec2f windDir, float timeStart, float timeSpeed,
                           float sunLightIntensity, float colorTemperatureMix, float colorTemperature);
void update_hdr_sky(float sky_rot, zeno::vec3f sky_rot3d, float sky_strength);
void using_hdr_sky(bool enable);
void optixUpdateUniforms(std::vector<float4> & inConstants);
}
