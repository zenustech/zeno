// https://github.com/alembic/alembic/blob/master/lib/Alembic/AbcGeom/Tests/PolyMeshTest.cpp
// WHY THE FKING ALEMBIC OFFICIAL GIVES NO DOC BUT ONLY "TESTS" FOR ME TO LEARN THEIR FKING LIB
#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/UserData.h>
#include <zeno/extra/GlobalState.h>
#include <Alembic/AbcGeom/All.h>
#include <Alembic/AbcCoreAbstract/All.h>
#include <Alembic/AbcCoreOgawa/All.h>
#include <Alembic/Abc/ErrorHandler.h>
#include "ABCTree.h"
#include "ABCCommon.h"
#include "zeno/utils/format.h"
#include "zeno/utils/string.h"
#include "zeno/types/ListObject.h"
#include "zeno/utils/log.h"
#include "zeno/funcs/PrimitiveUtils.h"
#include "zeno/extra/TempNode.h"
#include <numeric>
#include <filesystem>

namespace fs = std::filesystem;

using namespace Alembic::AbcGeom;
namespace zeno {
namespace {

template<typename T>
void write_velocity(std::shared_ptr<PrimitiveObject> prim, T& mesh_samp) {
    if (prim->verts.has_attr("v")) {
        auto &vel = prim->verts.attr<vec3f>("v");
        mesh_samp.setVelocities(V3fArraySample( ( const V3f * )vel.data(), vel.size() ));
    }
}

static void write_normal(std::shared_ptr<PrimitiveObject> prim, OPolyMeshSchema::Sample& mesh_samp) {
    if (prim->verts.has_attr("nrm")) {
        auto &nrm = (std::vector<N3f>&)prim->verts.attr<vec3f>("nrm");
        ON3fGeomParam::Sample oNormalsSample(nrm, kVaryingScope);
        mesh_samp.setNormals(oNormalsSample);
    }
}

struct WriteAlembic : INode {
    OArchive archive;
    OPolyMesh meshyObj;
    virtual void apply() override {
        bool flipFrontBack = get_param<int>("flipFrontBack");
        int frameid;
        if (has_input("frameid")) {
            frameid = get_param<int>("frameid");
        } else {
            frameid = getGlobalState()->frameid;
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
            if (flipFrontBack) {
                primFlipFaces(prim.get());
            }

            if (prim->loops.size()) {
                for (const auto& [start, size]: prim->polys) {
                    for (auto i = 0; i < size; i++) {
                        vertex_index_per_face.push_back(prim->loops[start + i]);
                    }
                    auto base = vertex_index_per_face.size() - size;
                    vertex_count_per_face.push_back(size);
                }
                if (prim->loops.has_attr("uvs")) {
                    std::vector<zeno::vec2f> uv_data;
                    for (const auto& uv: prim->uvs) {
                        uv_data.push_back(uv);
                    }
                    std::vector<uint32_t> uv_indices;
                    for (const auto& [start, size]: prim->polys) {
                        for (auto i = 0; i < size; i++) {
                            auto uv_index = prim->loops.attr<int>("uvs")[start + i];
                            uv_indices.push_back(uv_index);
                        }
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
                }
                else {
                    OPolyMeshSchema::Sample mesh_samp(
                    V3fArraySample( ( const V3f * )prim->verts.data(), prim->verts.size() ),
                            Int32ArraySample( vertex_index_per_face.data(), vertex_index_per_face.size() ),
                            Int32ArraySample( vertex_count_per_face.data(), vertex_count_per_face.size() ));
                    mesh.set( mesh_samp );
                }
            }
            else {
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
        {"bool", "flipFrontBack", "1"},
    },
    {"deprecated"},
});

template<typename T1>
void write_attrs(
        std::map<std::string, std::any> &verts_attrs
        , std::map<std::string, std::any> &loops_attrs
        , std::map<std::string, std::any> &polys_attrs
        , std::string path
        , std::shared_ptr<PrimitiveObject> prim
        , T1& schema
        , int frameid
        , int real_frame_start
        , std::map<int, vec3i> &prim_size_per_frame
) {
    OCompoundProperty arbAttrs = schema.getArbGeomParams();
    prim->verts.foreach_attr<std::variant<vec3f, float, int>>([&](auto const &key, auto &arr) {
        if (key == "v" || key == "nrm") {
            return;
        }
        std::string full_key = path + '/' + key;
        using T = std::decay_t<decltype(arr[0])>;
        if constexpr (std::is_same_v<T, zeno::vec3f>) {
            if (verts_attrs.count(full_key) == 0) {
                verts_attrs[full_key] = OFloatGeomParam(arbAttrs.getPtr(), key, false, kVaryingScope, 3);
                for (auto i = real_frame_start; i < frameid; i++) {
                    auto samp = OFloatGeomParam::Sample();
                    std::vector<float> v(std::get<0>(prim_size_per_frame[i]) * 3);
                    samp.setVals(v);
                    std::any_cast<OFloatGeomParam>(verts_attrs[full_key]).set(samp);
                }
            }
            auto samp = OFloatGeomParam::Sample();
            std::vector<float> v(arr.size() * 3);
            for (auto i = 0; i < arr.size(); i++) {
                v[i * 3 + 0] = arr[i][0];
                v[i * 3 + 1] = arr[i][1];
                v[i * 3 + 2] = arr[i][2];
            }
            samp.setVals(v);
            std::any_cast<OFloatGeomParam>(verts_attrs[full_key]).set(samp);
        } else if constexpr (std::is_same_v<T, float>) {
            if (verts_attrs.count(full_key) == 0) {
                verts_attrs[full_key] = OFloatGeomParam(arbAttrs.getPtr(), key, false, kVaryingScope, 1);
                for (auto i = real_frame_start; i < frameid; i++) {
                    auto samp = OFloatGeomParam::Sample();
                    std::vector<float> v(std::get<0>(prim_size_per_frame[i]));
                    samp.setVals(v);
                    std::any_cast<OFloatGeomParam>(verts_attrs[full_key]).set(samp);
                }
            }
            auto samp = OFloatGeomParam::Sample();
            samp.setVals(arr);
            std::any_cast<OFloatGeomParam>(verts_attrs[full_key]).set(samp);
        } else if constexpr (std::is_same_v<T, int>) {
            if (verts_attrs.count(full_key) == 0) {
                verts_attrs[full_key] = OInt32GeomParam (arbAttrs.getPtr(), key, false, kVaryingScope, 1);
                for (auto i = real_frame_start; i < frameid; i++) {
                    auto samp = OInt32GeomParam::Sample();
                    std::vector<int> v(std::get<0>(prim_size_per_frame[i]));
                    samp.setVals(v);
                    std::any_cast<OInt32GeomParam >(verts_attrs[full_key]).set(samp);
                }
            }
            auto samp = OInt32GeomParam::Sample();
            samp.setVals(arr);
            std::any_cast<OInt32GeomParam>(verts_attrs[full_key]).set(samp);
        }
    });
    if (prim->loops.size() > 0) {
        prim->loops.foreach_attr<std::variant<vec3f, float, int>>([&](auto const &key, auto &arr) {
            if (key == "uvs") {
                return;
            }
            std::string full_key = path + '/' + key;
            using T = std::decay_t<decltype(arr[0])>;
            if constexpr (std::is_same_v<T, zeno::vec3f>) {
                if (loops_attrs.count(full_key) == 0) {
                    loops_attrs[full_key] = OFloatGeomParam(arbAttrs.getPtr(), key, false, kFacevaryingScope, 3);
                    for (auto i = real_frame_start; i < frameid; i++) {
                        auto samp = OFloatGeomParam::Sample();
                        std::vector<float> v(std::get<1>(prim_size_per_frame[i]) * 3);
                        samp.setVals(v);
                        std::any_cast<OFloatGeomParam>(loops_attrs[full_key]).set(samp);
                    }
                }
                auto samp = OFloatGeomParam::Sample();
                std::vector<float> v(arr.size() * 3);
                for (auto i = 0; i < arr.size(); i++) {
                    v[i * 3 + 0] = arr[i][0];
                    v[i * 3 + 1] = arr[i][1];
                    v[i * 3 + 2] = arr[i][2];
                }
                samp.setVals(v);
                std::any_cast<OFloatGeomParam>(loops_attrs[full_key]).set(samp);
            } else if constexpr (std::is_same_v<T, float>) {
                if (loops_attrs.count(full_key) == 0) {
                    loops_attrs[full_key] = OFloatGeomParam(arbAttrs.getPtr(), key, false, kFacevaryingScope, 1);
                    for (auto i = real_frame_start; i < frameid; i++) {
                        auto samp = OFloatGeomParam::Sample();
                        std::vector<float> v(std::get<1>(prim_size_per_frame[i]));
                        samp.setVals(v);
                        std::any_cast<OFloatGeomParam>(loops_attrs[full_key]).set(samp);
                    }
                }
                auto samp = OFloatGeomParam::Sample();
                samp.setVals(arr);
                std::any_cast<OFloatGeomParam>(loops_attrs[full_key]).set(samp);
            } else if constexpr (std::is_same_v<T, int>) {
                if (loops_attrs.count(full_key) == 0) {
                    loops_attrs[full_key] = OInt32GeomParam (arbAttrs.getPtr(), key, false, kFacevaryingScope, 1);
                    for (auto i = real_frame_start; i < frameid; i++) {
                        auto samp = OInt32GeomParam::Sample();
                        std::vector<int> v(std::get<1>(prim_size_per_frame[i]));
                        samp.setVals(v);
                        std::any_cast<OInt32GeomParam>(loops_attrs[full_key]).set(samp);
                    }
                }
                auto samp = OInt32GeomParam::Sample();
                samp.setVals(arr);
                std::any_cast<OInt32GeomParam>(loops_attrs[full_key]).set(samp);
            }
        });
    }
    if (prim->polys.size() > 0) {
        prim->polys.foreach_attr<std::variant<vec3f, float, int>>([&](auto const &key, auto &arr) {
            if (key == "faceset" || key == "matid" || key == "abcpath") {
                return;
            }
            std::string full_key = path + '/' + key;
            using T = std::decay_t<decltype(arr[0])>;
            if constexpr (std::is_same_v<T, zeno::vec3f>) {
                if (polys_attrs.count(full_key) == 0) {
                    polys_attrs[full_key] = OFloatGeomParam(arbAttrs.getPtr(), key, false, kUniformScope, 3);
                    for (auto i = real_frame_start; i < frameid; i++) {
                        auto samp = OFloatGeomParam::Sample();
                        std::vector<float> v(std::get<2>(prim_size_per_frame[i]) * 3);
                        samp.setVals(v);
                        std::any_cast<OFloatGeomParam>(polys_attrs[full_key]).set(samp);
                    }
                }
                auto samp = OFloatGeomParam::Sample();
                std::vector<float> v(arr.size() * 3);
                for (auto i = 0; i < arr.size(); i++) {
                    v[i * 3 + 0] = arr[i][0];
                    v[i * 3 + 1] = arr[i][1];
                    v[i * 3 + 2] = arr[i][2];
                }
                samp.setVals(v);
                std::any_cast<OFloatGeomParam>(polys_attrs[full_key]).set(samp);
            } else if constexpr (std::is_same_v<T, float>) {
                if (polys_attrs.count(full_key) == 0) {
                    polys_attrs[full_key] = OFloatGeomParam(arbAttrs.getPtr(), key, false, kUniformScope, 1);
                    for (auto i = real_frame_start; i < frameid; i++) {
                        auto samp = OFloatGeomParam::Sample();
                        std::vector<float> v(std::get<2>(prim_size_per_frame[i]));
                        samp.setVals(v);
                        std::any_cast<OFloatGeomParam>(polys_attrs[full_key]).set(samp);
                    }
                }
                auto samp = OFloatGeomParam::Sample();
                samp.setVals(arr);
                std::any_cast<OFloatGeomParam>(polys_attrs[full_key]).set(samp);
            } else if constexpr (std::is_same_v<T, int>) {
                if (polys_attrs.count(full_key) == 0) {
                    polys_attrs[full_key] = OInt32GeomParam (arbAttrs.getPtr(), key, false, kUniformScope, 1);
                    for (auto i = real_frame_start; i < frameid; i++) {
                        auto samp = OInt32GeomParam::Sample();
                        std::vector<int> v(std::get<2>(prim_size_per_frame[i]));
                        samp.setVals(v);
                        std::any_cast<OInt32GeomParam>(polys_attrs[full_key]).set(samp);
                    }
                }
                auto samp = OInt32GeomParam::Sample();
                samp.setVals(arr);
                std::any_cast<OInt32GeomParam>(polys_attrs[full_key]).set(samp);
            }
        });
    }
}
void write_user_data(
        std::map<std::string, std::any> &user_attrs
        , std::string path
        , std::shared_ptr<PrimitiveObject> prim
        , OCompoundProperty& user
        , int frameid
        , int real_frame_start
) {
    auto &ud = prim->userData();
    for (const auto& [key, value] : ud.m_data) {
        std::string full_key = path + '/' + key;
        if (key == "faceset_count" || zeno::starts_with(key, "faceset_")) {
            continue;
        }
        if (key == "abcpath_count" || zeno::starts_with(key, "abcpath_")) {
            continue;
        }
        if (key == "matNum" || zeno::starts_with(key, "Material_") || key == "mtlid") {
            continue;
        }
        if (ud.has<int>(key)) {
            if (user_attrs.count(full_key) == 0) {
                auto p = OInt32Property(user, key);
                p.setTimeSampling(1);
                user_attrs[full_key] = p;
                if (real_frame_start != frameid) {
                    for (auto i = real_frame_start; i < frameid; i++) {
                        p.set({});
                    }
                }
            }
            std::any_cast<OInt32Property>(user_attrs[full_key]).set(ud.get2<int>(key));
        }
        else if (ud.has<float>(key)) {
            if (user_attrs.count(full_key) == 0) {
                auto p = OFloatProperty(user, key);
                p.setTimeSampling(1);
                user_attrs[full_key] = p;
                if (real_frame_start != frameid) {
                    for (auto i = real_frame_start; i < frameid; i++) {
                        p.set({});
                    }
                }
            }
            std::any_cast<OFloatProperty>(user_attrs[full_key]).set(ud.get2<float>(key));
        }
        else if (ud.has<vec2i>(key)) {
            if (user_attrs.count(full_key) == 0) {
                auto p = OV2iProperty(user, key);
                p.setTimeSampling(1);
                user_attrs[full_key] = p;
                if (real_frame_start != frameid) {
                    for (auto i = real_frame_start; i < frameid; i++) {
                        p.set({});
                    }
                }
            }
            auto v = ud.get2<vec2i>(key);
            std::any_cast<OV2iProperty>(user_attrs[full_key]).set(Imath::V2i(v[0], v[1]));
        }
        else if (ud.has<vec3i>(key)) {
            if (user_attrs.count(full_key) == 0) {
                auto p = OV3iProperty(user, key);
                p.setTimeSampling(1);
                user_attrs[full_key] = p;
                if (real_frame_start != frameid) {
                    for (auto i = real_frame_start; i < frameid; i++) {
                        p.set({});
                    }
                }
            }
            auto v = ud.get2<vec3i>(key);
            std::any_cast<OV3iProperty>(user_attrs[full_key]).set(Imath::V3i(v[0], v[1], v[2]));
        }
        else if (ud.has<vec2f>(key)) {
            if (user_attrs.count(full_key) == 0) {
                auto p = OV2fProperty(user, key);
                p.setTimeSampling(1);
                user_attrs[full_key] = p;
                if (real_frame_start != frameid) {
                    for (auto i = real_frame_start; i < frameid; i++) {
                        p.set({});
                    }
                }
            }
            auto v = ud.get2<vec2f>(key);
            std::any_cast<OV2fProperty>(user_attrs[full_key]).set(Imath::V2f(v[0], v[1]));
        }
        else if (ud.has<vec3f>(key)) {
            if (user_attrs.count(full_key) == 0) {
                auto p = OV3fProperty(user, key);
                p.setTimeSampling(1);
                user_attrs[full_key] = p;
                if (real_frame_start != frameid) {
                    for (auto i = real_frame_start; i < frameid; i++) {
                        p.set({});
                    }
                }
            }
            auto v = ud.get2<vec3f>(key);
            std::any_cast<OV3fProperty>(user_attrs[full_key]).set(Imath::V3f(v[0], v[1], v[2]));
        }
        else if (ud.has<std::string>(key)) {
            if (user_attrs.count(full_key) == 0) {
                auto p = OStringProperty(user, key);
                p.setTimeSampling(1);
                user_attrs[full_key] = p;
                if (real_frame_start != frameid) {
                    for (auto i = real_frame_start; i < frameid; i++) {
                        p.set({});
                    }
                }
            }
            std::any_cast<OStringProperty>(user_attrs[full_key]).set(ud.get2<std::string>(key));
        }
    }
}

static void write_faceset(
        std::shared_ptr<PrimitiveObject> prim
        , OPolyMeshSchema &mesh
        , std::map<std::string, OFaceSet> &o_faceset
        , std::map<std::string, OFaceSetSchema> &o_faceset_schema
        ) {
    auto &ud = prim->userData();
    std::vector<std::string> faceSetNames;
    std::vector<std::vector<int>> faceset_idxs;
    if (ud.has<int>("faceset_count")) {
        int faceset_count = ud.get2<int>("faceset_count");
        for (auto i = 0; i < faceset_count; i++) {
            faceSetNames.emplace_back(ud.get2<std::string>(zeno::format("faceset_{}", i)));
        }
        faceset_idxs.resize(faceset_count);
        std::vector<int> faceset;
        if (prim->polys.size() && prim->polys.attr_is<int>("faceset")) {
            faceset = prim->polys.attr<int>("faceset");
        }
        else if (prim->tris.size() && prim->tris.attr_is<int>("faceset")) {
            faceset = prim->tris.attr<int>("faceset");
        }
        for (auto i = 0; i < faceset.size(); i++) {
            if (faceset[i] >= 0) {
                faceset_idxs[faceset[i]].push_back(i);
            }
        }
        for (auto i = 0; i < faceset_count; i++) {
            if (o_faceset_schema.count(faceSetNames[i]) == 0) {
                o_faceset[faceSetNames[i]] = mesh.createFaceSet(faceSetNames[i]);
                o_faceset_schema[faceSetNames[i]] = o_faceset[faceSetNames[i]].getSchema ();
            }
            OFaceSetSchema::Sample my_face_set_samp ( faceset_idxs[i] );
            // faceset is visible, doesn't change.
            o_faceset_schema[faceSetNames[i]].set ( my_face_set_samp );
            o_faceset_schema[faceSetNames[i]].setFaceExclusivity ( kFaceSetExclusive );
        }
    }
}

void prim_to_poly_if_only_vertex(PrimitiveObject* p) {
    if (p->polys.size() || p->tris.size()) {
        return;
    }
    p->loops.resize(p->verts.size());
    std::iota(p->loops.begin(), p->loops.end(), 0);
    p->polys.resize(p->verts.size());
    for (auto i = 0; i < p->polys.size(); i++) {
        p->polys[i] = {i, 1};
    }
}

struct WriteAlembic2 : INode {
    OArchive archive;
    OPolyMesh meshyObj;
    std::string usedPath;
    std::map<std::string, std::any> verts_attrs;
    std::map<std::string, std::any> loops_attrs;
    std::map<std::string, std::any> polys_attrs;
    std::map<std::string, std::any> user_attrs;
    std::map<std::string, OFaceSet> o_faceset;
    std::map<std::string, OFaceSetSchema> o_faceset_schema;
    std::map<int, vec3i> prim_size_per_frame;
    int real_frame_start = -1;

    virtual void apply() override {
        auto prim = get_input<PrimitiveObject>("prim");
        bool flipFrontBack = get_input2<int>("flipFrontBack");
        float fps = get_input2<float>("fps");
        int frameid;
        if (has_input("frameid")) {
            frameid = std::lround(get_input2<float>("frameid"));
        } else {
            frameid = getGlobalState()->frameid;
        }
        int frame_start = get_input2<int>("frame_start");
        int frame_end = get_input2<int>("frame_end");
        std::string path = get_input2<std::string>("path");
        auto folderPath = fs::path(path).parent_path();

        if (!fs::exists(folderPath)) {
            fs::create_directories(folderPath);
        }
        if (usedPath != path) {
            usedPath = path;
            archive = CreateArchiveWithInfo(
                Alembic::AbcCoreOgawa::WriteArchive(),
                path,
                fps,
                "Zeno : " + getGlobalState()->zeno_version,
                "None"
            );
            real_frame_start = -1;
            meshyObj = OPolyMesh( OObject( archive, 1 ), "mesh" );
            verts_attrs.clear();
            loops_attrs.clear();
            polys_attrs.clear();
            user_attrs.clear();
            prim_size_per_frame.clear();
        }
        if (!(frame_start <= frameid && frameid <= frame_end)) {
            return;
        }
        if (real_frame_start == -1) {
            real_frame_start = frameid;
            archive.addTimeSampling(TimeSampling(1.0/fps, real_frame_start / fps));
        }
        if (archive.valid() == false) {
            zeno::makeError("Not init. Check whether in correct correct frame range.");
        }
        if (flipFrontBack) {
            primFlipFaces(prim.get());
        }
        {
            prim_to_poly_if_only_vertex(prim.get());
            // Create a PolyMesh class.
            OPolyMeshSchema &mesh = meshyObj.getSchema();
            write_faceset(prim, mesh, o_faceset, o_faceset_schema);

            OCompoundProperty user = mesh.getUserProperties();
            write_user_data(user_attrs, "", prim, user, frameid, real_frame_start);

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

            if (prim->tris.size()) {
                zeno::primPolygonate(prim.get(), true);
            }
            {
                {
                    prim_size_per_frame[frameid] = {
                        int(prim->verts.size()),
                        int(prim->loops.size()),
                        int(prim->polys.size()),
                    };
                }
                for (const auto& [start, size]: prim->polys) {
                    for (auto i = 0; i < size; i++) {
                        vertex_index_per_face.push_back(prim->loops[start + i]);
                    }
                    auto base = vertex_index_per_face.size() - size;
                    vertex_count_per_face.push_back(size);
                }
                if (prim->loops.has_attr("uvs")) {
                    std::vector<zeno::vec2f> uv_data;
                    for (const auto& uv: prim->uvs) {
                        uv_data.push_back(uv);
                    }
                    std::vector<uint32_t> uv_indices;
                    for (const auto& [start, size]: prim->polys) {
                        for (auto i = 0; i < size; i++) {
                            auto uv_index = prim->loops.attr<int>("uvs")[start + i];
                            uv_indices.push_back(uv_index);
                        }
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
                    write_velocity(prim, mesh_samp);
                    write_normal(prim, mesh_samp);
                    write_attrs(verts_attrs, loops_attrs, polys_attrs, "", prim, mesh, frameid, real_frame_start, prim_size_per_frame);
                    mesh.set( mesh_samp );
                }
                else {
                    OPolyMeshSchema::Sample mesh_samp(
                    V3fArraySample( ( const V3f * )prim->verts.data(), prim->verts.size() ),
                            Int32ArraySample( vertex_index_per_face.data(), vertex_index_per_face.size() ),
                            Int32ArraySample( vertex_count_per_face.data(), vertex_count_per_face.size() ));
                    write_velocity(prim, mesh_samp);
                    write_normal(prim, mesh_samp);
                    write_attrs(verts_attrs, loops_attrs, polys_attrs, "", prim, mesh, frameid, real_frame_start, prim_size_per_frame);
                    mesh.set( mesh_samp );
                }
            }
        }
    }
};

ZENDEFNODE(WriteAlembic2, {
    {
        {"prim"},
        {"frameid"},
        {"writepath", "path", ""},
        {"int", "frame_start", "0"},
        {"int", "frame_end", "100"},
        {"float", "fps", "25"},
        {"bool", "flipFrontBack", "1"},
    },
    {
    },
    {},
    {"alembic"},
});

struct WriteAlembicPrims : INode {
    OArchive archive;
    std::string usedPath;
    std::map<std::string, OPolyMesh> meshyObjs;
    std::map<std::string, std::any> verts_attrs;
    std::map<std::string, std::any> loops_attrs;
    std::map<std::string, std::any> polys_attrs;
    std::map<std::string, std::any> user_attrs;
    std::map<std::string, std::map<std::string, OFaceSet>> o_faceset;
    std::map<std::string, std::map<std::string, OFaceSetSchema>> o_faceset_schema;
    std::map<std::string, std::map<int, vec3i>> prim_size_per_frame;
    int real_frame_start = -1;

    virtual void apply() override {
        std::vector<std::shared_ptr<PrimitiveObject>> prims;

        if (has_input("prim")) {
            auto prim = get_input2<PrimitiveObject>("prim");
            prims = primUnmergeFaces(prim.get(), "abcpath");
        }
        else {
            prims = get_input<ListObject>("prims")->get<PrimitiveObject>();
        }
        bool flipFrontBack = get_input2<int>("flipFrontBack");
        float fps = get_input2<float>("fps");
        int frameid;
        if (has_input("frameid")) {
            frameid = std::lround(get_input2<float>("frameid"));
        } else {
            frameid = getGlobalState()->frameid;
        }
        int frame_start = get_input2<int>("frame_start");
        int frame_end = get_input2<int>("frame_end");
        std::string path = get_input2<std::string>("path");
        auto folderPath = fs::path(path).parent_path();

        if (!fs::exists(folderPath)) {
            fs::create_directories(folderPath);
        }
        std::vector<std::shared_ptr<PrimitiveObject>> new_prims;

        {
            // unmerged prim when abcpath_count of a prim in list more than 1
            std::vector<std::shared_ptr<PrimitiveObject>> temp_prims;
            int counter = 0;
            for (auto prim: prims) {
                counter += 1;
                prim_to_poly_if_only_vertex(prim.get());
                if (prim->userData().get2<int>("abcpath_count", 0) == 0) {
                    prim_set_abcpath(prim.get(), "/ABC/unassigned");
                }
                if (prim->userData().get2<int>("abcpath_count") == 1) {
                    if (prim->userData().get2<int>("faceset_count", 0) == 0) {
                        prim_set_faceset(prim.get(), "defFS");
                    }
                    temp_prims.push_back(prim);
                }
                else {
                    auto unmerged_prims = primUnmergeFaces(prim.get(), "abcpath");
                    for (const auto& unmerged_prim: unmerged_prims) {
                        if (unmerged_prim->userData().get2<int>("faceset_count", 0) == 0) {
                            prim_set_faceset(unmerged_prim.get(), "defFS");
                        }
                        temp_prims.push_back(unmerged_prim);
                    }
                }
            }
            prims = temp_prims;

            // merge by abcpath
            std::vector<std::string> paths;
            std::map<std::string, std::vector<std::shared_ptr<PrimitiveObject>>> path_to_prims;

            for (auto prim: prims) {
                auto path = prim->userData().get2<std::string>("abcpath_0");
                if (path_to_prims.count(path) == 0) {
                    paths.push_back(path);
                    path_to_prims[path] = {};
                }
                path_to_prims[path].push_back(prim);
            }
            for (auto path : paths) {
                if (path_to_prims[path].size() > 1) {
                    std::vector<zeno::PrimitiveObject *> primList;
                    for (auto prim: path_to_prims[path]) {
                        primList.push_back(prim.get());
                    }
                    auto prim = primMergeWithFacesetMatid(primList);
                    new_prims.push_back(prim);
                }
                else {
                    new_prims.push_back(path_to_prims[path][0]);
                }
            }
        }
        if (usedPath != path) {
            usedPath = path;
            archive = CreateArchiveWithInfo(
                Alembic::AbcCoreOgawa::WriteArchive(),
                path,
                fps,
                "Zeno : " + getGlobalState()->zeno_version,
                "None"
            );
            meshyObjs.clear();
            verts_attrs.clear();
            loops_attrs.clear();
            polys_attrs.clear();
            user_attrs.clear();
            o_faceset.clear();
            o_faceset_schema.clear();
            prim_size_per_frame.clear();
            real_frame_start = -1;
            for (auto prim: new_prims) {
                auto path = prim->userData().get2<std::string>("abcpath_0");
                if (!starts_with(path, "/ABC/")) {
                    log_error("abcpath_0 must start with /ABC/");
                }
                auto n_path = path.substr(5);
                auto subnames = split_str(n_path, '/');
                OObject oObject = OObject( archive, 1 );
                for (auto i = 0; i < subnames.size() - 1; i++) {
                    auto child = oObject.getChild(subnames[i]);
                    if (child.valid()) {
                        oObject = child;
                    }
                    else {
                        oObject = OObject( oObject, subnames[i] );
                    }
                }
                meshyObjs[path] = OPolyMesh (oObject, subnames[subnames.size() - 1]);
            }
        }
        if (!(frame_start <= frameid && frameid <= frame_end)) {
            return;
        }
        if (real_frame_start == -1) {
            real_frame_start = frameid;
            archive.addTimeSampling(TimeSampling(1.0/fps, real_frame_start / fps));
        }
        if (archive.valid() == false) {
            zeno::makeError("Not init. Check whether in correct correct frame range.");
        }
        for (auto prim: new_prims) {
            if (flipFrontBack) {
                primFlipFaces(prim.get());
            }
            auto path = prim->userData().get2<std::string>("abcpath_0");

            {
                // Create a PolyMesh class.
                OPolyMeshSchema &mesh = meshyObjs[path].getSchema();
                auto &ud = prim->userData();
                std::vector<std::string> faceSetNames;
                std::vector<std::vector<int>> faceset_idxs;
                write_faceset(prim, mesh, o_faceset[path], o_faceset_schema[path]);

                OCompoundProperty user = mesh.getUserProperties();
                write_user_data(user_attrs, path, prim, user, frameid, real_frame_start);

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

                if (prim->tris.size()) {
                    zeno::primPolygonate(prim.get(), true);
                }
                {
                    {
                        prim_size_per_frame[path][frameid] = {
                            int(prim->verts.size()),
                            int(prim->loops.size()),
                            int(prim->polys.size()),
                        };
                    }
                    for (const auto& [start, size]: prim->polys) {
                        for (auto i = 0; i < size; i++) {
                            vertex_index_per_face.push_back(prim->loops[start + i]);
                        }
                        auto base = vertex_index_per_face.size() - size;
                        vertex_count_per_face.push_back(size);
                    }
                    if (prim->loops.has_attr("uvs")) {
                        std::vector<zeno::vec2f> uv_data;
                        for (const auto& uv: prim->uvs) {
                            uv_data.push_back(uv);
                        }
                        std::vector<uint32_t> uv_indices;
                        for (const auto& [start, size]: prim->polys) {
                            for (auto i = 0; i < size; i++) {
                                auto uv_index = prim->loops.attr<int>("uvs")[start + i];
                                uv_indices.push_back(uv_index);
                            }
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
                        write_velocity(prim, mesh_samp);
                        write_normal(prim, mesh_samp);
                        write_attrs(verts_attrs, loops_attrs, polys_attrs, path, prim, mesh, frameid, real_frame_start, prim_size_per_frame[path]);
                        mesh.set( mesh_samp );
                    }
                    else {
                        OPolyMeshSchema::Sample mesh_samp(
                        V3fArraySample( ( const V3f * )prim->verts.data(), prim->verts.size() ),
                                Int32ArraySample( vertex_index_per_face.data(), vertex_index_per_face.size() ),
                                Int32ArraySample( vertex_count_per_face.data(), vertex_count_per_face.size() ));
                        write_velocity(prim, mesh_samp);
                        write_normal(prim, mesh_samp);
                        write_attrs(verts_attrs, loops_attrs, polys_attrs, path, prim, mesh, frameid, real_frame_start, prim_size_per_frame[path]);
                        mesh.set( mesh_samp );
                    }
                }
            }
        }
    }
};

ZENDEFNODE(WriteAlembicPrims, {
    {
        {"prim"},
        {"list", "prims"},
        {"frameid"},
        {"writepath", "path", ""},
        {"int", "frame_start", "0"},
        {"int", "frame_end", "100"},
        {"float", "fps", "25"},
        {"bool", "flipFrontBack", "1"},
    },
    {
    },
    {},
    {"alembic"},
});

} // namespace
} // namespace zeno
