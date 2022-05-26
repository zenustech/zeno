// https://github.com/alembic/alembic/blob/master/lib/Alembic/AbcGeom/Tests/PolyMeshTest.cpp
// WHY THE FKING ALEMBIC OFFICIAL GIVES NO DOC BUT ONLY "TESTS" FOR ME TO LEARN THEIR FKING LIB
#include <zeno/zeno.h>
#include <zeno/utils/logger.h>
#include <zeno/types/StringObject.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/extra/GlobalState.h>
#include <Alembic/AbcGeom/All.h>
#include <Alembic/AbcCoreAbstract/All.h>
#include <Alembic/AbcCoreOgawa/All.h>
#include <Alembic/AbcCoreHDF5/All.h>
#include <Alembic/Abc/ErrorHandler.h>
#include "ABCTree.h"
#include <cstring>
#include <cstdio>
#include <spdlog/spdlog.h>


using namespace Alembic::AbcGeom;
namespace zeno {
namespace {

struct WriteAlembic : INode {
    OArchive archive;
    OPolyMesh meshyObj;
    virtual void apply() override {
        int frameid;
        if (has_input("frameid")) {
            frameid = get_param<int>("frameid");
        } else {
            frameid = zeno::state.frameid;
        }
        int frame_start = get_param<int>("frame_start");
        int frame_end = get_param<int>("frame_end");
        if (frameid == frame_start) {
            std::string path = get_param<std::string>("path");
            archive = {Alembic::AbcCoreOgawa::WriteArchive(), path};
            archive.addTimeSampling(TimeSampling(1.0/24, frame_start / 24.0));
            meshyObj = OPolyMesh( OObject( archive, 1 ), "mesh" );
        }
        auto prim = get_input<PrimitiveObject>("prim");
        if (frame_start <= frameid && frameid <= frame_end) {
            // Create a PolyMesh class.
            OPolyMeshSchema &mesh = meshyObj.getSchema();
            mesh.setTimeSampling(1);

            // some apps can arbitrarily name their primary UVs, this function allows
            // you to do that, and must be done before the first time you set UVs
            // on the schema
            mesh.setUVSourceName("main_uv");

            // Set a mesh sample.
            // We're creating the sample inline here,
            // but we could create a static sample and leave it around,
            // only modifying the parts that have changed.
            std::vector<int32_t> vertex_index_per_face;
            std::vector<int32_t> vertex_count_per_face;
            for (auto i = 0; i < prim->tris.size(); i++) {
                vertex_index_per_face.push_back(prim->tris[i][0]);
                vertex_index_per_face.push_back(prim->tris[i][1]);
                vertex_index_per_face.push_back(prim->tris[i][2]);
            }
            vertex_count_per_face.resize(prim->tris.size(), 3);
            if (prim->tris.has_attr("uv0")) {
                std::vector<zeno::vec2f> uv_data;
                std::vector<uint32_t> uv_indices;
                auto& uv0 = prim->tris.attr<zeno::vec3f>("uv0");
                auto& uv1 = prim->tris.attr<zeno::vec3f>("uv1");
                auto& uv2 = prim->tris.attr<zeno::vec3f>("uv2");
                for (auto i = 0; i < prim->tris.size(); i++) {
                    uv_data.emplace_back(uv0[i][0], uv0[i][1]);
                    uv_data.emplace_back(uv1[i][0], uv1[i][1]);
                    uv_data.emplace_back(uv2[i][0], uv2[i][1]);
                    uv_indices.push_back(uv_indices.size());
                    uv_indices.push_back(uv_indices.size());
                    uv_indices.push_back(uv_indices.size());
                }

                // UVs and Normals use GeomParams, which can be written or read
                // as indexed or not, as you'd like.
                OV2fGeomParam::Sample uvsamp;
                uvsamp.setVals(V2fArraySample( (const V2f *)uv_data.data(), uv_data.size()));
                uvsamp.setIndices(UInt32ArraySample( uv_indices.data(), uv_indices.size() ));
                uvsamp.setScope(kFacevaryingScope);
                OPolyMeshSchema::Sample mesh_samp(
                    V3fArraySample( ( const V3f * )prim->verts.data(), prim->verts.size() ),
                    Int32ArraySample( vertex_index_per_face.data(), vertex_index_per_face.size() ),
                    Int32ArraySample( vertex_count_per_face.data(), vertex_count_per_face.size() ),
                    uvsamp);

                mesh.set( mesh_samp );
            } else {
                OPolyMeshSchema::Sample mesh_samp(
                    V3fArraySample( ( const V3f * )prim->verts.data(), prim->verts.size() ),
                    Int32ArraySample( vertex_index_per_face.data(), vertex_index_per_face.size() ),
                    Int32ArraySample( vertex_count_per_face.data(), vertex_count_per_face.size() ));
                mesh.set( mesh_samp );
            }
        }
    }
};

ZENDEFNODE(WriteAlembic, {
    {
        {"prim"},
        {"frameid"},
    },
    {},
    {
        {"writepath", "path", ""},
        {"int", "frame_start", "0"},
        {"int", "frame_end", "100"},
    },
    {"alembic"},
});

} // namespace
} // namespace zeno
