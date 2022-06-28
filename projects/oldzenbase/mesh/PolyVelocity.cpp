#include <zeno/zeno.h>
#include <zeno/MeshObject.h>
#include <zeno/ParticlesObject.h>
#include <omp.h>
#include <zeno/NumericObject.h>
//#include <tl/function_ref.hpp>
//openvdb::FloatGrid::Ptr grid = 
//openvdb::tools::meshToSignedDistanceField<openvdb::FloatGrid>
//(*openvdb::math::Transform::createLinearTransform(h), 
//points, triangles, quads, 4, 4);

namespace zeno {
static void subtractMesh(float dt, MeshObject* a, MeshObject* b,
                  ParticlesObject* c)
{
    size_t n = a->vertices.size();
    c->pos.resize(n);
    c->vel.resize(n);
#pragma omp parallel for
    for(int i=0;i<n;i++)
    {
        c->pos[i] = a->vertices[i];
        c->vel[i] = (a->vertices[i] - b->vertices[i])/dt;
    }
}
struct GeoVertexVel : zeno::INode{
    virtual void apply() override {
        auto dt = get_input("dt")->as<zeno::NumericObject>()->get<float>();
        auto mesh1 = get_input("TargetMesh")->as<MeshObject>();
        auto mesh2 = get_input("OriginMesh")->as<MeshObject>();
        auto result = zeno::IObject::make<ParticlesObject>();
        subtractMesh(dt, mesh1, mesh2, result.get());
        set_output("MeshVel", result);
  }
};

static int defGeoVertexVel = zeno::defNodeClass<GeoVertexVel>("GeoVertexVel",
    { /* inputs: */ {
        "dt", "TargetMesh", "OriginMesh", 
    }, /* outputs: */ {
        "MeshVel",
    }, /* params: */ {
    
    }, /* category: */ {
        "deprecated",
    }});

}
