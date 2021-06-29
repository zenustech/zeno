#include <zeno/zeno.h>
#include <zeno/MeshObject.h>
#include <zeno/PrimitiveObject.h>
#include <zeno/NumericObject.h>
#include <zeno/vec.h>
#include <cstring>
#include <cstdlib>
#include <cassert>

namespace zen {

struct MeshToPrimitive : zen::INode{
    virtual void apply() override {
    auto mesh = get_input("mesh")->as<MeshObject>();
    auto result = zen::IObject::make<PrimitiveObject>();
    result->add_attr<zen::vec3f>("pos");
    result->add_attr<zen::vec3f>("tex");
    result->add_attr<zen::vec3f>("nrm");
    result->resize(mesh->vertices.size());
    result->tris.resize(mesh->vertices.size()/3);
    result->quads.resize(0);

#pragma omp parallel for
    for(int i=0;i<mesh->vertices.size();i++)
    {
        result->attr<zen::vec3f>("pos")[i] = zen::vec3f(mesh->vertices[i].x,
            mesh->vertices[i].y, mesh->vertices[i].z);

        if(mesh->uvs.size()>0)
        result->attr<zen::vec3f>("tex")[i] = zen::vec3f(mesh->uvs[i].x, mesh->uvs[i].y,
            0);
        if(mesh->normals.size()>0)
        result->attr<zen::vec3f>("nrm")[i] = zen::vec3f(mesh->normals[i].x, mesh->normals[i].y,mesh->normals[i].z);
    }
#pragma omp parallel for
    for(int i=0;i<mesh->vertices.size()/3;i++)
    {
        result->tris[i] = zen::vec3i(i*3, i*3+1, i*3+2);
    }
    
    set_output("prim", result);
  }
};

static int defMeshToPrimitive = zen::defNodeClass<MeshToPrimitive>("MeshToPrimitive",
    { /* inputs: */ {
        "mesh",
    }, /* outputs: */ {
        "prim",
    }, /* params: */ { 
    }, /* category: */ {
    "primitive",
    }});



struct PrimitiveToMesh : zen::INode{
    virtual void apply() override {
    auto mesh = get_input("prim")->as<PrimitiveObject>();
    auto result = zen::IObject::make<MeshObject>();
    auto pos = mesh->attr<zen::vec3f>("pos");
    auto uv = mesh->attr<zen::vec3f>("tex");
    auto nrm = mesh->attr<zen::vec3f>("nrm");
    result->vertices.resize(pos.size());
    result->uvs.resize(uv.size());
    result->normals.resize(nrm.size());

#pragma omp parallel for
    for(int i=0;i<result->vertices.size();i++)
    {
        result->vertices[i] = glm::vec3(mesh->attr<zen::vec3f>("pos")[i][0],
            mesh->attr<zen::vec3f>("pos")[i][1], mesh->attr<zen::vec3f>("pos")[i][2]);

        if(result->uvs.size()>0)
            result->uvs[i] = glm::vec2(mesh->attr<zen::vec3f>("tex")[i][0], 
            mesh->attr<zen::vec3f>("tex")[i][1]);

        if(result->normals.size()>0)
            result->normals[i] = glm::vec3(mesh->attr<zen::vec3f>("nrm")[i][0], mesh->attr<zen::vec3f>("nrm")[i][1],mesh->attr<zen::vec3f>("nrm")[i][2]);
    }

    
    set_output("mesh", result);
  }
};

static int defPrimitiveToMesh = zen::defNodeClass<PrimitiveToMesh>("PrimitiveToMesh",
    { /* inputs: */ {
        "prim",
    }, /* outputs: */ {
        "mesh",
    }, /* params: */ { 
    }, /* category: */ {
    "primitive",
    }});


}
