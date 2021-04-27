#if 0
#include <zen/zen.h>
#include <zen/MeshObject.h>
#include <zen/ParticlesObject.h>
#include <omp.h>
//#include <tl/function_ref.hpp>
//openvdb::FloatGrid::Ptr grid = 
//openvdb::tools::meshToSignedDistanceField<openvdb::FloatGrid>
//(*openvdb::math::Transform::createLinearTransform(h), 
//points, triangles, quads, 4, 4);

namespace zenbase {
void subtractMesh(float dt, std::unique_ptr<MeshObject> &a, std::unique_ptr<MeshObject> &b,
                  std::unique_ptr<ParticlesObject> &c)
{
    size_t n = to->vertices.size();
    c->pos.resize(n);
    c->vel.resize(n);
#pragma omp parallel for
    for(int i=0;i<n;i++)
    {
        c->pos[i] = a->vertices[i];
        c->vel[i] = (a->vertices[i] - b->vertices[i])/dt;
    }
}
struct PolyVelocity : zen::INode{
    virtual void apply() override {
    auto dt = get_input("dt")->as<zenbase::NumericObject>()->get<float>();
    auto mesh1 = get_input("Mesh1")->as<MeshObject>();
    auto mesh2 = get_input("Mesh2")->as<MeshObject>();
    auto result = zen::IObject::make<ParticlesObject>();
    
    set_output("MeshVel", result);
  }
};

static int defPolyVelocity = zen::defNodeClass<PolyVelocity>("PolyVelocity",
    { /* inputs: */ {
        "dt", "Mesh1", "Mesh2", 
    }, /* outputs: */ {
        "MeshVel",
    }, /* params: */ {
    
    }, /* category: */ {
        "trimesh",
    }});

}
#endif