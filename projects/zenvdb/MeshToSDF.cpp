#include <cstddef>
#include <zeno/zeno.h>
#include <zeno/MeshObject.h>
#include <zeno/PrimitiveObject.h>
#include <openvdb/tools/Morphology.h>
#include <openvdb/tools/MeshToVolume.h>
#include <openvdb/tools/VolumeToMesh.h>
#include <zeno/VDBGrid.h>
#include <omp.h>
#include <zeno/ZenoInc.h>

//#include <tl/function_ref.hpp>
//openvdb::FloatGrid::Ptr grid = 
//openvdb::tools::meshToSignedDistanceField<openvdb::FloatGrid>
//(*openvdb::math::Transform::createLinearTransform(h), 
//points, triangles, quads, 4, 4);

namespace zeno {

struct MeshToSDF : zeno::INode{
    virtual void apply() override {
    auto h = std::get<float>(get_param("voxel_size"));
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
    if(std::get<std::string>(get_param("type"))==std::string("vertex"))
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
        "mesh","Dx",
    }, /* outputs: */ {
        "sdf",
    }, /* params: */ {
    {"float", "voxel_size", "0.08 0"},
    {"string", "type", "vertex"},
    }, /* category: */ {
    "openvdb",
    }});



struct PrimitiveToSDF : zeno::INode{
    virtual void apply() override {
    auto h = std::get<float>(get_param("voxel_size"));
    if(has_input("Dx"))
    {
      h = get_input("Dx")->as<NumericObject>()->get<float>();
    }
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
    if(std::get<std::string>(get_param("type"))==std::string("vertex"))
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
        "PrimitiveMesh","Dx",
    }, /* outputs: */ {
        "sdf",
    }, /* params: */ {
        {"float", "voxel_size", "0.08 0"},
        {"string", "type", "vertex"},
    }, /* category: */ {
    "openvdb",
    }});



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



}
