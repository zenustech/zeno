#include <cstddef>
#include <zeno/zeno.h>
#include <zeno/PrimitiveObject.h>
#include <openvdb/tools/Morphology.h>
#include <openvdb/tools/VolumeToMesh.h>
#include <zeno/VDBGrid.h>
#include <omp.h>
#include <zeno/ZenoInc.h>


namespace zeno {

struct SDFToPoly : zeno::INode{
    virtual void apply() override {
    auto sdf = get_input("SDF")->as<VDBFloatGrid>();
    auto mesh = IObject::make<PrimitiveObject>();
    auto adaptivity = std::get<float>(get_param("adaptivity"));
    auto isoValue = std::get<float>(get_param("isoValue"));
    auto allowQuads = get_param<bool>("allowQuads");
    std::vector<openvdb::Vec3s> points(0);
    std::vector<openvdb::Vec3I> tris(0);
    std::vector<openvdb::Vec4I> quads(0);
    openvdb::tools::volumeToMesh(*(sdf->m_grid), points, tris, quads, isoValue, adaptivity, true);
    mesh->resize(points.size());
    auto &meshpos = mesh->add_attr<zeno::vec3f>("pos");
#pragma omp parallel for
    for(int i=0;i<points.size();i++)
    {
        meshpos[i] = zeno::vec3f(points[i][0],points[i][1],points[i][2]);
    }
    if (allowQuads) {
        mesh->tris.resize(tris.size());
        mesh->quads.resize(quads.size());
#pragma omp parallel for
        for(int i=0;i<tris.size();i++)
        {
            mesh->tris[i] = zeno::vec3i(tris[i][0],tris[i][1],tris[i][2]);
        }
#pragma omp parallel for
        for(int i=0;i<quads.size();i++)
        {
            mesh->quads[i] = zeno::vec4i(quads[i][0],quads[i][1],quads[i][2],quads[i][3]);
        }
    } else {
        mesh->tris.resize(tris.size() + 2*quads.size());
#pragma omp parallel for
        for(int i=0;i<tris.size();i++)
        {
            mesh->tris[i] = zeno::vec3i(tris[i][0],tris[i][1],tris[i][2]);
        }
#pragma omp parallel for
        for(int i=0;i<quads.size();i++)
        {
            mesh->tris[i*2+tris.size()] = zeno::vec3i(quads[i][0],quads[i][1],quads[i][2]);
            mesh->tris[i*2+1+tris.size()] = zeno::vec3i(quads[i][2],quads[i][3],quads[i][0]);
        }
    }

    set_output("Mesh", mesh);
  }
};

static int defSDFToPoly = zeno::defNodeClass<SDFToPoly>("SDFToPoly",
    { /* inputs: */ {
        "SDF",
    }, /* outputs: */ {
        "Mesh",
    }, /* params: */ {
        {"float", "isoValue", "0"},
        {"float", "adaptivity", "0"},
        {"bool", "allowQuads", "0"},
    }, /* category: */ {
    "openvdb",
    }});


struct ConvertTo_VDBFloatGrid_PrimitiveObject : SDFToPoly {
    virtual void apply() override {
        SDFToPoly::apply();
        get_input<PrimitiveObject>("prim")->move_assign(std::move(smart_any_cast<std::shared_ptr<IObject>>(outputs.at("prim"))).get());
    }
};

ZENO_DEFOVERLOADNODE(ConvertTo, _VDBFloatGrid_PrimitiveObject, typeid(VDBFloatGrid).name(), typeid(PrimitiveObject).name())({
        {"mesh", "prim"},
        {},
        {},
        {"primitive"},
});

}
