#include <zeno/zeno.h>
#include <zeno/MeshObject.h>
#include <zeno/ParticlesObject.h>
#include <omp.h>
//#include <tl/function_ref.hpp>
//openvdb::FloatGrid::Ptr grid = 
//openvdb::tools::meshToSignedDistanceField<openvdb::FloatGrid>
//(*openvdb::math::Transform::createLinearTransform(h), 
//points, triangles, quads, 4, 4);

namespace zeno {
static void MergeMesh(std::shared_ptr<MeshObject> &to, std::shared_ptr<MeshObject> &from)
{
    size_t n = to->vertices.size();
    to->vertices.resize(n+from->size());
    to->uvs.resize(n+from->size());
    to->normals.resize(n+from->size());
#pragma omp parallel for
    for(int i=0;i<from->size();i++)
    {
        to->vertices[n+i] = from->vertices[i];
        to->uvs[n+i] = from->uvs[i];
        to->normals[n+i] = from->normals[i];
    }
}
struct PolyDuplicate : zeno::INode{
    virtual void apply() override {
    auto inmesh = get_input("Mesh")->as<MeshObject>();
    auto posList = get_input("Particles")->as<ParticlesObject>();
    auto result = zeno::IObject::make<MeshObject>();
    //printf("%d\n",posList->pos.size());
    for(int i=0;i<posList->size();i++)
    {
        auto p = posList->pos[i];
        auto tempmesh = inmesh->Clone();
        tempmesh->translate(p);
        MergeMesh(result, tempmesh);
    }
    set_output("Meshes", result);
  }
};

static int defPolyDuplicate = zeno::defNodeClass<PolyDuplicate>("PolyDuplicate",
    { /* inputs: */ {
        "Mesh", "Particles", 
    }, /* outputs: */ {
        "Meshes",
    }, /* params: */ {
    
    }, /* category: */ {
        "trimesh",
    }});

struct MeshCopy : zeno::INode {
  virtual void apply() override {
    auto copyFromMesh = get_input("copyFrom")->as<MeshObject>();
    auto copyToMesh = get_input("copyTo")->as<MeshObject>();
    copyToMesh->vertices.resize(copyFromMesh->vertices.size());
    copyToMesh->uvs.resize(copyFromMesh->uvs.size());
    copyToMesh->normals.resize(copyFromMesh->normals.size());
#pragma omp parallel for
    for (int i = 0; i < copyFromMesh->vertices.size(); i++) {
      copyToMesh->vertices[i] = copyFromMesh->vertices[i];
      copyToMesh->uvs[i] = copyFromMesh->uvs[i];
      copyToMesh->normals[i] = copyFromMesh->normals[i];
    }
  }
};

static int defMeshCopy =
    zeno::defNodeClass<MeshCopy>("MeshCopy", {/* inputs: */
                                             {
                                                 "copyFrom",
                                                 "copyTo",
                                             },
                                             /* outputs: */
                                             {},
                                             /* params: */
                                             {},
                                             /* category: */
                                             {
                                                 "trimesh",
                                             }});
} // namespace zeno
