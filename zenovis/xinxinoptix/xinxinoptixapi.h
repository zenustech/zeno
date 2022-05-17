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
void optixupdatemesh();
void optixupdatematerial(std::vector<const char *> const &shaders);

void set_window_size(int nx, int ny);
void set_view_matrix(float const *view);

void load_object(std::string const &key, float const *verts, size_t numverts, int const *tris, size_t numtris, std::map<std::string, std::pair<float const *, size_t>> const &vtab);
void unload_object(std::string const &key);

}
