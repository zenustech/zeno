#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

#include <blender/DNA_mesh_types.h>
#include <blender/DNA_meshdata_types.h>

#include <zs/zeno/ds/Mesh.h>

using namespace zs;

struct UpdateId {
    std::string bl_name;
    std::shared_ptr<zeno::ds::Mesh> zs_mesh;
};


std::vector<UpdateId> updates;


static std::vector<std::string> get_updated_names() {
    std::vector<std::string> bl_names;
    for (auto const &update: updates) {
        bl_names.push_back(update.bl_name);
    }
    return bl_names;
}


static void set_updated_names(std::vector<std::string> bl_names) {
    updates.clear();
    for (auto const &bl_name: bl_names) {
        updates.push_back({.bl_name = bl_name});
    }
}


static uintptr_t get_updated_mesh(int idx) {
    auto const &update = updates.at(idx);
    auto *zs_mesh = update.zs_mesh.get();
    return reinterpret_cast<uintptr_t>(zs_mesh);
}


static size_t get_mesh_verts(uintptr_t p_zs_mesh, uintptr_t p_bl_verts) {
    auto zs_mesh = reinterpret_cast<zeno::ds::Mesh *>(p_zs_mesh);

    if (auto bl_verts = reinterpret_cast<MVert *>(p_bl_verts)) {
        for (int i = 0; i < zs_mesh->vert.size(); i++) {
            bl_verts[i].co[0] = zs_mesh->vert[i][0];
            bl_verts[i].co[1] = zs_mesh->vert[i][1];
            bl_verts[i].co[2] = zs_mesh->vert[i][2];
        }
    }
    return zs_mesh->vert.size();
}


PYBIND11_MODULE(zs_blender, m) {
    m.def("get_updated_names", get_updated_names);
    m.def("set_updated_names", set_updated_names);
    m.def("get_updated_mesh", get_updated_mesh);
    m.def("get_mesh_verts", get_mesh_verts);
}
