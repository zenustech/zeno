#include <cstddef>
#include <zeno/zeno.h>
#include <zeno/MeshObject.h>
#include <zeno/PrimitiveObject.h>
#include <openvdb/tools/Morphology.h>
#include <openvdb/tools/MeshToVolume.h>
#include <zeno/VDBGrid.h>
#include <omp.h>
#include <zeno/ZenoInc.h>
#include <openvdb/tools/LevelSetUtil.h> 
//#include <tl/function_ref.hpp>
//openvdb::FloatGrid::Ptr grid = 
//openvdb::tools::meshToSignedDistanceField<openvdb::FloatGrid>
//(*openvdb::math::Transform::createLinearTransform(h), 
//points, triangles, quads, 4, 4);

namespace zeno {

struct MeshToSDF : zeno::INode{
    virtual void apply() override {
    auto h = get_param<float>(("voxel_size"));
    if(has_input("Dx"))
    {
      h = get_input("Dx")->as<NumericObject>()->get<float>();
    }
    auto mesh = get_input("mesh")->as<MeshObject>();
    auto result = zeno::IObject::make<VDBFloatGrid>();
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
    if(get_param<std::string>(("type"))==std::string("vertex"))
    {
        vdbtransform->postTranslate(openvdb::Vec3d{ -0.5,-0.5,-0.5 }*double(h));
    }
    result->m_grid = openvdb::tools::meshToSignedDistanceField<openvdb::FloatGrid>(*vdbtransform,points, triangles, quads, 4, 4);
    openvdb::tools::signedFloodFill(result->m_grid->tree());
    set_output("sdf", result);
  }
};

static int defMeshToSDF = zeno::defNodeClass<MeshToSDF>("MeshToSDF",
    { /* inputs: */ {
        "mesh",{"float","Dx"},
    }, /* outputs: */ {
        "sdf",
    }, /* params: */ {
    {"float", "voxel_size", "0.08 0"},
    {"enum vertex cell", "type", "vertex"},
    }, /* category: */ {
    "openvdb",
    }});



struct PrimitiveToSDF : zeno::INode{
    virtual void apply() override {
    //auto h = get_param<float>(("voxel_size"));
    //if(has_input("Dx"))
    //{
      //h = get_input<NumericObject>("Dx")->get<float>();
    //}
    auto h = get_input2<float>("Dx");
    //auto h = get_input("Dx")->as<NumericObject>()->get<float>();
    auto mesh = get_input("PrimitiveMesh")->as<PrimitiveObject>();
    auto result = zeno::IObject::make<VDBFloatGrid>();
    std::vector<openvdb::Vec3s> points;
    std::vector<openvdb::Vec3I> triangles;
    std::vector<openvdb::Vec4I> quads;
    points.resize(mesh->attr<zeno::vec3f>("pos").size());
    triangles.resize(mesh->tris.size());
    quads.resize(0);
#pragma omp parallel for
    for(int i=0;i<points.size();i++)
    {
        points[i] = openvdb::Vec3s(mesh->attr<zeno::vec3f>("pos")[i][0], mesh->attr<zeno::vec3f>("pos")[i][1], mesh->attr<zeno::vec3f>("pos")[i][2]);
    }
#pragma omp parallel for
    for(int i=0;i<triangles.size();i++)
    {
        triangles[i] = openvdb::Vec3I(mesh->tris[i][0], mesh->tris[i][1], mesh->tris[i][2]);
    }
    auto vdbtransform = openvdb::math::Transform::createLinearTransform(h);
    if(get_param<std::string>(("type"))==std::string("vertex"))
    {
        vdbtransform->postTranslate(openvdb::Vec3d{ -0.5,-0.5,-0.5 }*double(h));
    }
    result->m_grid = openvdb::tools::meshToSignedDistanceField<openvdb::FloatGrid>(*vdbtransform,points, triangles, quads, 4, 4);
    openvdb::tools::signedFloodFill(result->m_grid->tree());
    set_output("sdf", result);
  }
};

static int defPrimitiveToSDF = zeno::defNodeClass<PrimitiveToSDF>("PrimitiveToSDF",
    { /* inputs: */ {
        "PrimitiveMesh", {"float","Dx","0.08"},
    }, /* outputs: */ {
        "sdf",
    }, /* params: */ {
        //{"float", "voxel_size", "0.08 0"},
        {"enum vertex cell", "type", "vertex"},
    }, /* category: */ {
    "openvdb",
    }});

struct SDFToFog : INode 
{
    virtual void apply() override {
        auto sdf = get_input("SDF")->as<VDBFloatGrid>();
        auto result = zeno::IObject::make<VDBFloatGrid>();
        auto dx = sdf->m_grid->voxelSize()[0];
        result->m_grid = sdf->m_grid->deepCopy();
        openvdb::tools::sdfToFogVolume(*(result->m_grid));
        set_output("oSDF", result);
    }
};
static int defSDFToFog = zeno::defNodeClass<SDFToFog>("SDFToFog",
    { /* inputs: */ {
        "SDF",
    }, /* outputs: */ {
        "oSDF",
    }, /* params: */ {
    }, /* category: */ {
    "openvdb",
    }});
}
