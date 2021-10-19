#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

#include <blender/DNA_mesh_types.h>
#include <blender/DNA_meshdata_types.h>

#include <zs/zeno/ds/Mesh.h>

using namespace zs;

struct UpdateId {
    std::shared_ptr<zeno::ds::Mesh> zs_mesh;
    std::string bl_name;
};


std::vector<UpdateId> updates;


static auto get_updates() {
    return updates;
}


PYBIND11_MODULE(zs_blender, m) {
    m.def("get_updates", get_updates);
    m.def("get_updates", get_updates);
}
