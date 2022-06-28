#include <zeno/zeno.h>
#include <zeno/MeshObject.h>
#include <zeno/PrimitiveObject.h>
#include <zeno/NumericObject.h>
#include <zeno/utils/vec.h>
#include <cstring>
#include <cstdlib>
#include <cassert>

namespace zeno {

struct MeshToPrimitive : zeno::INode{
    virtual void apply() override {
    auto mesh = get_input("mesh")->as<MeshObject>();
    auto result = zeno::IObject::make<PrimitiveObject>();
    result->add_attr<zeno::vec3f>("pos");
    result->add_attr<zeno::vec3f>("tex");
    result->add_attr<zeno::vec3f>("nrm");
    result->resize(mesh->vertices.size());
    result->tris.resize(mesh->vertices.size()/3);
    result->quads.resize(0);

#pragma omp parallel for
    for(int i=0;i<mesh->vertices.size();i++)
    {
        result->attr<zeno::vec3f>("pos")[i] = zeno::vec3f(mesh->vertices[i].x,
            mesh->vertices[i].y, mesh->vertices[i].z);

        if(mesh->uvs.size()>0)
        result->attr<zeno::vec3f>("tex")[i] = zeno::vec3f(mesh->uvs[i].x, mesh->uvs[i].y,
            0);
        if(mesh->normals.size()>0)
        result->attr<zeno::vec3f>("nrm")[i] = zeno::vec3f(mesh->normals[i].x, mesh->normals[i].y,mesh->normals[i].z);
    }
#pragma omp parallel for
    for(int i=0;i<mesh->vertices.size()/3;i++)
    {
        result->tris[i] = zeno::vec3i(i*3, i*3+1, i*3+2);
    }
    
    set_output("prim", result);
  }
};

static int defMeshToPrimitive = zeno::defNodeClass<MeshToPrimitive>("MeshToPrimitive",
    { /* inputs: */ {
        "mesh",
    }, /* outputs: */ {
        "prim",
    }, /* params: */ { 
    }, /* category: */ {
    "deprecated",
    }});



struct PrimitiveToMesh : zeno::INode{
    virtual void apply() override {
    auto mesh = get_input("prim")->as<PrimitiveObject>();
    auto result = zeno::IObject::make<MeshObject>();
    auto pos = mesh->attr<zeno::vec3f>("pos");
    auto uv = mesh->attr<zeno::vec3f>("tex");
    auto nrm = mesh->attr<zeno::vec3f>("nrm");
    result->vertices.resize(pos.size());
    result->uvs.resize(uv.size());
    result->normals.resize(nrm.size());

#pragma omp parallel for
    for(int i=0;i<result->vertices.size();i++)
    {
        result->vertices[i] = glm::vec3(mesh->attr<zeno::vec3f>("pos")[i][0],
            mesh->attr<zeno::vec3f>("pos")[i][1], mesh->attr<zeno::vec3f>("pos")[i][2]);

        if(result->uvs.size()>0)
            result->uvs[i] = glm::vec2(mesh->attr<zeno::vec3f>("tex")[i][0],
            mesh->attr<zeno::vec3f>("tex")[i][1]);

        if(result->normals.size()>0)
            result->normals[i] = glm::vec3(mesh->attr<zeno::vec3f>("nrm")[i][0], mesh->attr<zeno::vec3f>("nrm")[i][1],mesh->attr<zeno::vec3f>("nrm")[i][2]);
    }

    
    set_output("mesh", result);
  }
};

static int defPrimitiveToMesh = zeno::defNodeClass<PrimitiveToMesh>("PrimitiveToMesh",
    { /* inputs: */ {
        "prim",
    }, /* outputs: */ {
        "mesh",
    }, /* params: */ { 
    }, /* category: */ {
    "deprecated",
    }});


struct ConvertTo_MeshObject_PrimitiveObject : MeshToPrimitive {
    virtual void apply() override {
        MeshToPrimitive::apply();
        get_input<PrimitiveObject>("prim")->move_assign(std::move(smart_any_cast<std::shared_ptr<IObject>>(outputs.at("prim"))).get());
    }
};

ZENO_DEFOVERLOADNODE(ConvertTo, _MeshObject_PrimitiveObject, typeid(MeshObject).name(), typeid(PrimitiveObject).name())({
        {"mesh", "prim"},
        {},
        {},
        {"deprecated"},
});


struct ConvertTo_PrimitiveObject_MeshObject : PrimitiveToMesh {
    virtual void apply() override {
        PrimitiveToMesh::apply();
        get_input<MeshObject>("mesh")->move_assign(std::move(smart_any_cast<std::shared_ptr<IObject>>(outputs.at("mesh"))).get());
    }
};

ZENO_DEFOVERLOADNODE(ConvertTo, _PrimitiveObject_MeshObject, typeid(PrimitiveObject).name(), typeid(MeshObject).name())({
        {"prim", "mesh"},
        {},
        {},
        {"deprecated"},
});

}
