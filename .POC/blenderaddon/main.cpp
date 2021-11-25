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


static size_t get_updates_count() {
    return updates.size();
}


static std::string get_update_bl_name(int idx) {
    auto const &update = updates.at(idx);
    return update.bl_name;
}


static uintptr_t get_update_zs_mesh(int idx) {
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
    m.def("get_update_zs_mesh", get_update_zs_mesh);
    m.def("set_update_bl_name", get_update_bl_name);
    m.def("get_updates_count", get_updates_count);
    m.def("get_mesh_verts", get_mesh_verts);
}
