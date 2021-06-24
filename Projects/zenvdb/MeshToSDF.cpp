#include <zen/zen.h>
#include <zen/MeshObject.h>
#include <zen/PrimitiveObject.h>
#include <zen/VDBGrid.h>
#include <omp.h>
//#include <tl/function_ref.hpp>
//openvdb::FloatGrid::Ptr grid = 
//openvdb::tools::meshToSignedDistanceField<openvdb::FloatGrid>
//(*openvdb::math::Transform::createLinearTransform(h), 
//points, triangles, quads, 4, 4);

namespace zen {

struct MeshToSDF : zen::INode{
    virtual void apply() override {
    auto h = std::get<float>(get_param("voxel_size"));
    auto mesh = get_input("mesh")->as<MeshObject>();
    auto result = zen::IObject::make<VDBFloatGrid>();
    std::vector<openvdb::Vec3s> points;
    std::vector<openvdb::Vec3I> triangles;
    std::vector<openvdb::Vec4I> quads;
    points.resize(mesh->vertices.size());
    triangles.resize(mesh->vertices.size()/3);
    quads.resize(0);
#pragma omp parallel for
    for(int i=0;i<mesh->vertices.size();i++)
    {
        points[i] = openvdb::Vec3s(mesh->vertices[i].x, mesh->vertices[i].y, mesh->vertices[i].z);
    }
#pragma omp parallel for
    for(int i=0;i<mesh->vertices.size()/3;i++)
    {
        triangles[i] = openvdb::Vec3I(i*3, i*3+1, i*3+2);
    }
    auto vdbtransform = openvdb::math::Transform::createLinearTransform(h);
    if(std::get<std::string>(get_param("type"))==std::string("vertex"))
    {
        vdbtransform->postTranslate(openvdb::Vec3d{ -0.5,-0.5,-0.5 }*double(h));
    }
    result->m_grid = openvdb::tools::meshToSignedDistanceField<openvdb::FloatGrid>(*vdbtransform,points, triangles, quads, 4, 4);
    openvdb::tools::signedFloodFill(result->m_grid->tree());
    set_output("sdf", result);
  }
};

static int defMeshToSDF = zen::defNodeClass<MeshToSDF>("MeshToSDF",
    { /* inputs: */ {
        "mesh",
    }, /* outputs: */ {
        "sdf",
    }, /* params: */ {
    {"float", "voxel_size", "0.08 0"},
    {"string", "type", "vertex"},
    }, /* category: */ {
    "openvdb",
    }});



struct PrimitiveToSDF : zen::INode{
    virtual void apply() override {
    auto h = std::get<float>(get_param("voxel_size"));
    auto mesh = get_input("PrimitiveMesh")->as<PrimitiveObject>();
    auto result = zen::IObject::make<VDBFloatGrid>();
    std::vector<openvdb::Vec3s> points;
    std::vector<openvdb::Vec3I> triangles;
    std::vector<openvdb::Vec4I> quads;
    points.resize(mesh->attr<zen::vec3f>("pos").size());
    triangles.resize(mesh->tris.size());
    quads.resize(0);
#pragma omp parallel for
    for(int i=0;i<points.size();i++)
    {
        points[i] = openvdb::Vec3s(mesh->attr<zen::vec3f>("pos")[i][0], mesh->attr<zen::vec3f>("pos")[i][1], mesh->attr<zen::vec3f>("pos")[i][2]);
    }
#pragma omp parallel for
    for(int i=0;i<triangles.size();i++)
    {
        triangles[i] = openvdb::Vec3I(mesh->tris[i][0], mesh->tris[i][1], mesh->tris[i][2]);
    }
    auto vdbtransform = openvdb::math::Transform::createLinearTransform(h);
    if(std::get<std::string>(get_param("type"))==std::string("vertex"))
    {
        vdbtransform->postTranslate(openvdb::Vec3d{ -0.5,-0.5,-0.5 }*double(h));
    }
    result->m_grid = openvdb::tools::meshToSignedDistanceField<openvdb::FloatGrid>(*vdbtransform,points, triangles, quads, 4, 4);
    openvdb::tools::signedFloodFill(result->m_grid->tree());
    set_output("sdf", result);
  }
};

static int defPrimitiveToSDF = zen::defNodeClass<PrimitiveToSDF>("PrimitiveToSDF",
    { /* inputs: */ {
        "PrimitiveMesh",
    }, /* outputs: */ {
        "sdf",
    }, /* params: */ {
        {"float", "voxel_size", "0.08 0"},
        {"string", "type", "vertex"},
    }, /* category: */ {
    "openvdb",
    }});
}
