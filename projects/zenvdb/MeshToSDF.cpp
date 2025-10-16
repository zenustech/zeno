#include <cstddef>
#include <zeno/zeno.h>
#include <zeno/types/MeshObject.h>
#include <zeno/PrimitiveObject.h>
#include <zeno/types/IGeometryObject.h>
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
    auto h = get_param_float("voxel_size");
    if(has_input("Dx"))
    {
        h = get_input2_float("Dx");
    }
    auto mesh = safe_dynamic_cast<MeshObject>(get_input("mesh"));
    auto result = std::make_unique<VDBFloatGrid>();
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
    if (zsString2Std(get_param_string("type")) == "vertex")
    {
        vdbtransform->postTranslate(openvdb::Vec3d{ -0.5,-0.5,-0.5 }*double(h));
    }
    result->m_grid = openvdb::tools::meshToSignedDistanceField<openvdb::FloatGrid>(*vdbtransform,points, triangles, quads, 4, 4);
    openvdb::tools::signedFloodFill(result->m_grid->tree());
    set_output("sdf", std::move(result));
  }
};

static int defMeshToSDF = zeno::defNodeClass<MeshToSDF>("MeshToSDF",
    { /* inputs: */ {
        {gParamType_Mesh, "mesh", "", zeno::Socket_ReadOnly},
        {gParamType_Float,"Dx"},
    }, /* outputs: */ {
        {"object", "sdf"},
    }, /* params: */ {
    {gParamType_Float, "voxel_size", "0.08 0"},
    {"enum vertex cell", "type", "vertex"},
    }, /* category: */ {
    "deprecated",
    }});

struct GeometryToSDF : zeno::INode {
    virtual void apply() override {
        auto h = get_input2_float("Dx");
        auto mesh = safe_dynamic_cast<GeometryObject_Adapter>(get_input("Mesh"));
        auto result = std::make_unique<VDBFloatGrid>();
        int nfaces = mesh->nfaces();
        const auto& pos = mesh->points_pos();
        std::vector<openvdb::Vec3s> points(mesh->npoints());
        std::vector<openvdb::Vec3I> triangles;
        std::vector<openvdb::Vec4I> quads;

        for (int i = 0; i < pos.size(); i++) {
            points[i] = openvdb::Vec3s(pos[i][0], pos[i][1], pos[i][2]);
        }

        if (mesh->is_base_triangle()) {
            //triangles
        }
        else {
            quads.resize(nfaces);
            for (int iFace = 0; iFace < nfaces; iFace++) {
                const zeno::Vector<int>& indice = mesh->face_points(iFace);
                if (indice.size() != 4) {
                    throw makeError<UnimplError>("there is a face which is a not a Quadrilateral");
                }
                quads[iFace] = openvdb::Vec4I(indice[0], indice[1], indice[2], indice[3]);
            }
        }

        auto vdbtransform = openvdb::math::Transform::createLinearTransform(h);
        if (zsString2Std(get_param_string("type")) == "vertex")
        {
            vdbtransform->postTranslate(openvdb::Vec3d{ -0.5,-0.5,-0.5 }*double(h));
        }
        result->m_grid = openvdb::tools::meshToSignedDistanceField<openvdb::FloatGrid>(*vdbtransform, points, triangles, quads, 4, 4);
        openvdb::tools::signedFloodFill(result->m_grid->tree());
        set_output("sdf", std::move(result));
    }
};

static int defGeomToSDF = zeno::defNodeClass<GeometryToSDF>("GeometryToSDF",
    { /* inputs: */
        {
            {gParamType_Geometry, "Mesh"},
            {gParamType_Float,    "Dx","0.08"},
        }, /* outputs: */
    {
        {gParamType_VDBGrid, "sdf"},
    }, /* params: */ {
        //{"float", "voxel_size", "0.08 0"},
        {"enum vertex cell", "type", "vertex"},
    }, /* category: */ {
    "openvdb",
    } });



struct PrimitiveToSDF : zeno::INode{
    virtual void apply() override {
    //auto h = get_param_float("voxel_size");
    //if(has_input("Dx"))
    //{
      //h = get_input2_float("Dx");
    //}
    auto h = get_input2_float("Dx");
    //auto h = get_input2_float("Dx")();
    if (auto p = dynamic_cast<VDBFloatGrid *>(get_input("PrimitiveMesh"))) {
        set_output("sdf", clone_input("PrimitiveMesh"));
        return;
    }
    auto mesh = safe_dynamic_cast<PrimitiveObject>(get_input("PrimitiveMesh"));
    auto result = std::make_unique<VDBFloatGrid>();
    std::vector<openvdb::Vec3s> points;
    std::vector<openvdb::Vec3I> triangles;
    std::vector<openvdb::Vec4I> quads;
    points.resize(mesh->verts.size());
    triangles.resize(mesh->tris.size());
    quads.resize(mesh->quads.size());
#pragma omp parallel for
    for(int i=0;i<points.size();i++)
    {
        points[i] = openvdb::Vec3s(mesh->verts[i][0], mesh->verts[i][1], mesh->verts[i][2]);
    }
#pragma omp parallel for
    for(int i=0;i<triangles.size();i++)
    {
        triangles[i] = openvdb::Vec3I(mesh->tris[i][0], mesh->tris[i][1], mesh->tris[i][2]);
    }
#pragma omp parallel for
    for(int i=0;i<quads.size();i++)
    {
        quads[i] = openvdb::Vec4I(mesh->quads[i][0], mesh->quads[i][1], mesh->quads[i][2], mesh->quads[i][3]);
    }
    auto vdbtransform = openvdb::math::Transform::createLinearTransform(h);
    if (get_param_string("type") == "vertex")
    {
        vdbtransform->postTranslate(openvdb::Vec3d{ -0.5,-0.5,-0.5 }*double(h));
    }
    result->m_grid = openvdb::tools::meshToSignedDistanceField<openvdb::FloatGrid>(*vdbtransform,points, triangles, quads, 4, 4);
    openvdb::tools::signedFloodFill(result->m_grid->tree());
    set_output("sdf", std::move(result));
  }
};

static int defPrimitiveToSDF = zeno::defNodeClass<PrimitiveToSDF>("PrimitiveToSDF",
    { /* inputs: */ {
        {gParamType_Primitive, "PrimitiveMesh", "", zeno::Socket_ReadOnly},
        {gParamType_Float,"Dx","0.08"},
    }, /* outputs: */ {
        {gParamType_VDBGrid, "sdf"},
    }, /* params: */ {
        //{gParamType_Float, "voxel_size", "0.08 0"},
        {"enum vertex cell", "type", "vertex"},
    }, /* category: */ {
    "openvdb",
    }});

struct SDFToFog : INode 
{
    virtual void apply() override {
        auto sdf = safe_uniqueptr_cast<VDBFloatGrid>(clone_input("SDF"));
        if (!has_input("inplace") || !get_input2_bool("inplace")) {
            sdf = std::make_unique<VDBFloatGrid>(sdf->m_grid->deepCopy());
        }
        //auto dx = sdf->m_grid->voxelSize()[0];
        openvdb::tools::sdfToFogVolume(*(sdf->m_grid));
        set_output("oSDF", std::move(sdf));
    }
};
static int defSDFToFog = zeno::defNodeClass<SDFToFog>("SDFToFog",
    { /* inputs: */ {
        {gParamType_VDBGrid,"SDF", "", zeno::Socket_ReadOnly},
        {gParamType_Bool, "inplace", "0"},
    }, /* outputs: */ {
        {"object", "oSDF"},
    }, /* params: */ {
    }, /* category: */ {
    "openvdb",
    }});
}
