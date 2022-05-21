#pragma once

#include <string>
#include <vector>
#include <map>

namespace xinxinoptix {

void optixcleanup();
void optixrender(int fbo = 0);
void *optixgetimg(int &w, int &h);
void optixinit(int argc, char* argv[]);
void optixupdateend();
void optixupdatemesh(std::map<std::string, int> const &mtlidlut);
void optixupdatematerial(std::vector<std::string> const &shaders);

void set_window_size(int nx, int ny);
void set_perspective(float const *U, float const *V, float const *W, float const *E, float aspect, float fov);

void load_object(std::string const &key, std::string const &mtlid, float const *verts, size_t numverts, int const *tris, size_t numtris, std::map<std::string, std::pair<float const *, size_t>> const &vtab);
void unload_object(std::string const &key);

}
