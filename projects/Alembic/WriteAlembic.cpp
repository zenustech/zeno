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
#include "zeno/utils/format.h"
#include "zeno/utils/string.h"
#include "zeno/types/ListObject.h"
#include "zeno/utils/log.h"
#include "Alembic/AbcCoreHDF5/ReadWrite.h"
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
        ON3fGeomParam::Sample oNormalsSample(nrm, kFacevaryingScope);
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

            if (prim->loops.size()) {
                for (const auto& [start, size]: prim->polys) {
                    for (auto i = 0; i < size; i++) {
                        vertex_index_per_face.push_back(prim->loops[start + i]);
                    }
                    auto base = vertex_index_per_face.size() - size;
                    if (flipFrontBack) {
                        for (int j = 0; j < (size / 2); j++) {
                            std::swap(vertex_index_per_face[base + j], vertex_index_per_face[base + size - 1 - j]);
                        }
                    }
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
                        auto base = uv_indices.size() - size;
                        if (flipFrontBack) {
                            for (int j = 0; j < (size / 2); j++) {
                                std::swap(uv_indices[base + j], uv_indices[base + size - 1 - j]);
                            }
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
                    if (flipFrontBack) {
                        vertex_index_per_face.push_back(prim->tris[i][2]);
                        vertex_index_per_face.push_back(prim->tris[i][1]);
                    }
                    else {
                        vertex_index_per_face.push_back(prim->tris[i][1]);
                        vertex_index_per_face.push_back(prim->tris[i][2]);
                    }
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
                        if (flipFrontBack) {
                            uv_data.emplace_back(uv2[i][0], uv2[i][1]);
                            uv_data.emplace_back(uv1[i][0], uv1[i][1]);
                        }
                        else {
                            uv_data.emplace_back(uv1[i][0], uv1[i][1]);
                            uv_data.emplace_back(uv2[i][0], uv2[i][1]);
                        }
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

template<typename T1, typename T2>
void write_attrs(std::map<std::string, std::any> &attrs, std::string path, std::shared_ptr<PrimitiveObject> prim, T1& schema, T2& samp) {
    OCompoundProperty arbAttrs = schema.getArbGeomParams();
    prim->verts.foreach_attr<std::variant<vec3f, float, int>>([&](auto const &key, auto &arr) {
        if (key == "v" || key == "nrm") {
            return;
        }
        std::string full_key = path + '/' + key;
        using T = std::decay_t<decltype(arr[0])>;
        if constexpr (std::is_same_v<T, zeno::vec3f>) {
            if (attrs.count(full_key) == 0) {
                attrs[full_key] = OFloatGeomParam(arbAttrs.getPtr(), key, false, kVaryingScope, 3);
            }
            auto samp = OFloatGeomParam::Sample();
            std::vector<float> v(arr.size() * 3);
            for (auto i = 0; i < arr.size(); i++) {
                v[i * 3 + 0] = arr[i][0];
                v[i * 3 + 1] = arr[i][1];
                v[i * 3 + 2] = arr[i][2];
            }
            samp.setVals(v);
            std::any_cast<OFloatGeomParam>(attrs[full_key]).set(samp);
        } else if constexpr (std::is_same_v<T, float>) {
            if (attrs.count(full_key) == 0) {
                attrs[full_key] = OFloatGeomParam(arbAttrs.getPtr(), key, false, kVaryingScope, 1);
            }
            auto samp = OFloatGeomParam::Sample();
            samp.setVals(arr);
            std::any_cast<OFloatGeomParam>(attrs[full_key]).set(samp);
        } else if constexpr (std::is_same_v<T, int>) {
            if (attrs.count(full_key) == 0) {
                attrs[full_key] = OInt32GeomParam (arbAttrs.getPtr(), key, false, kVaryingScope, 1);
            }
            auto samp = OInt32GeomParam::Sample();
            samp.setVals(arr);
            std::any_cast<OInt32GeomParam>(attrs[full_key]).set(samp);
        }
    });
    if (prim->loops.size() > 0) {
        prim->loops.foreach_attr<std::variant<vec3f, float, int>>([&](auto const &key, auto &arr) {
            std::string full_key = path + '/' + key;
            using T = std::decay_t<decltype(arr[0])>;
            if constexpr (std::is_same_v<T, zeno::vec3f>) {
                if (attrs.count(full_key) == 0) {
                    attrs[full_key] = OFloatGeomParam(arbAttrs.getPtr(), key, false, kFacevaryingScope, 3);
                }
                auto samp = OFloatGeomParam::Sample();
                std::vector<float> v(arr.size() * 3);
                for (auto i = 0; i < arr.size(); i++) {
                    v[i * 3 + 0] = arr[i][0];
                    v[i * 3 + 1] = arr[i][1];
                    v[i * 3 + 2] = arr[i][2];
                }
                samp.setVals(v);
                std::any_cast<OFloatGeomParam>(attrs[full_key]).set(samp);
            } else if constexpr (std::is_same_v<T, float>) {
                if (attrs.count(full_key) == 0) {
                    attrs[full_key] = OFloatGeomParam(arbAttrs.getPtr(), key, false, kFacevaryingScope, 1);
                }
                auto samp = OFloatGeomParam::Sample();
                samp.setVals(arr);
                std::any_cast<OFloatGeomParam>(attrs[full_key]).set(samp);
            } else if constexpr (std::is_same_v<T, int>) {
                if (attrs.count(full_key) == 0) {
                    attrs[full_key] = OInt32GeomParam (arbAttrs.getPtr(), key, false, kFacevaryingScope, 1);
                }
                auto samp = OInt32GeomParam::Sample();
                samp.setVals(arr);
                std::any_cast<OInt32GeomParam>(attrs[full_key]).set(samp);
            }
        });
    }
    if (prim->polys.size() > 0) {
        prim->polys.foreach_attr<std::variant<vec3f, float, int>>([&](auto const &key, auto &arr) {
            std::string full_key = path + '/' + key;
            using T = std::decay_t<decltype(arr[0])>;
            if constexpr (std::is_same_v<T, zeno::vec3f>) {
                if (attrs.count(full_key) == 0) {
                    attrs[full_key] = OFloatGeomParam(arbAttrs.getPtr(), key, false, kUniformScope, 3);
                }
                auto samp = OFloatGeomParam::Sample();
                std::vector<float> v(arr.size() * 3);
                for (auto i = 0; i < arr.size(); i++) {
                    v[i * 3 + 0] = arr[i][0];
                    v[i * 3 + 1] = arr[i][1];
                    v[i * 3 + 2] = arr[i][2];
                }
                samp.setVals(v);
                std::any_cast<OFloatGeomParam>(attrs[full_key]).set(samp);
            } else if constexpr (std::is_same_v<T, float>) {
                if (attrs.count(full_key) == 0) {
                    attrs[full_key] = OFloatGeomParam(arbAttrs.getPtr(), key, false, kUniformScope, 1);
                }
                auto samp = OFloatGeomParam::Sample();
                samp.setVals(arr);
                std::any_cast<OFloatGeomParam>(attrs[full_key]).set(samp);
            } else if constexpr (std::is_same_v<T, int>) {
                if (attrs.count(full_key) == 0) {
                    attrs[full_key] = OInt32GeomParam (arbAttrs.getPtr(), key, false, kUniformScope, 1);
                }
                auto samp = OInt32GeomParam::Sample();
                samp.setVals(arr);
                std::any_cast<OInt32GeomParam>(attrs[full_key]).set(samp);
            }
        });
    }
    if (prim->tris.size() > 0) {
        prim->tris.foreach_attr<std::variant<vec3f, float, int>>([&](auto const &key, auto &arr) {
            zeno::log_info("{} {}", key, int(arr.size()));
            std::string full_key = path + '/' + key;
            using T = std::decay_t<decltype(arr[0])>;
            if constexpr (std::is_same_v<T, zeno::vec3f>) {
                if (attrs.count(full_key) == 0) {
                    attrs[full_key] = OFloatGeomParam(arbAttrs.getPtr(), key, false, kUniformScope, 3);
                }
                auto samp = OFloatGeomParam::Sample();
                std::vector<float> v(arr.size() * 3);
                for (auto i = 0; i < arr.size(); i++) {
                    v[i * 3 + 0] = arr[i][0];
                    v[i * 3 + 1] = arr[i][1];
                    v[i * 3 + 2] = arr[i][2];
                }
                samp.setVals(v);
                std::any_cast<OFloatGeomParam>(attrs[full_key]).set(samp);
            } else if constexpr (std::is_same_v<T, float>) {
                zeno::log_info("std::is_same_v<T, float>");
                if (attrs.count(full_key) == 0) {
                    attrs[full_key] = OFloatGeomParam(arbAttrs.getPtr(), key, false, kUniformScope, 1);
                }
                auto samp = OFloatGeomParam::Sample();
                samp.setVals(arr);
                std::any_cast<OFloatGeomParam>(attrs[full_key]).set(samp);
            } else if constexpr (std::is_same_v<T, int>) {
                if (attrs.count(full_key) == 0) {
                    attrs[full_key] = OInt32GeomParam (arbAttrs.getPtr(), key, false, kUniformScope, 1);
                }
                auto samp = OInt32GeomParam::Sample();
                samp.setVals(arr);
                std::any_cast<OInt32GeomParam>(attrs[full_key]).set(samp);
            }
        });
    }
}
void write_user_data(std::map<std::string, std::any> &user_attrs, std::string path, std::shared_ptr<PrimitiveObject> prim, OCompoundProperty& user) {
    auto &ud = prim->userData();
    for (const auto& [key, value] : ud.m_data) {
        std::string full_key = path + '/' + key;
        if (key == "faceset_count" || zeno::starts_with(key, "faceset_")) {
            continue;
        }
        if (ud.has<int>(key)) {
            if (user_attrs.count(full_key) == 0) {
                auto p = OInt32Property(user, key);
                p.setTimeSampling(1);
                user_attrs[full_key] = p;
            }
            std::any_cast<OInt32Property>(user_attrs[full_key]).set(ud.get2<int>(key));
        }
        else if (ud.has<float>(key)) {
            if (user_attrs.count(full_key) == 0) {
                auto p = OFloatProperty(user, key);
                p.setTimeSampling(1);
                user_attrs[full_key] = p;
            }
            std::any_cast<OFloatProperty>(user_attrs[full_key]).set(ud.get2<float>(key));
        }
        else if (ud.has<vec2i>(key)) {
            if (user_attrs.count(full_key) == 0) {
                auto p = OV2iProperty(user, key);
                p.setTimeSampling(1);
                user_attrs[full_key] = p;
            }
            auto v = ud.get2<vec2i>(key);
            std::any_cast<OV2iProperty>(user_attrs[full_key]).set(Imath_3_2::V2i(v[0], v[1]));
        }
        else if (ud.has<vec3i>(key)) {
            if (user_attrs.count(full_key) == 0) {
                auto p = OV3iProperty(user, key);
                p.setTimeSampling(1);
                user_attrs[full_key] = p;
            }
            auto v = ud.get2<vec3i>(key);
            std::any_cast<OV3iProperty>(user_attrs[full_key]).set(Imath_3_2::V3i(v[0], v[1], v[2]));
        }
        else if (ud.has<vec2f>(key)) {
            if (user_attrs.count(full_key) == 0) {
                auto p = OV2fProperty(user, key);
                p.setTimeSampling(1);
                user_attrs[full_key] = p;
            }
            auto v = ud.get2<vec2f>(key);
            std::any_cast<OV2fProperty>(user_attrs[full_key]).set(Imath_3_2::V2f(v[0], v[1]));
        }
        else if (ud.has<vec3f>(key)) {
            if (user_attrs.count(full_key) == 0) {
                auto p = OV3fProperty(user, key);
                p.setTimeSampling(1);
                user_attrs[full_key] = p;
            }
            auto v = ud.get2<vec3f>(key);
            std::any_cast<OV3fProperty>(user_attrs[full_key]).set(Imath_3_2::V3f(v[0], v[1], v[2]));
        }
        else if (ud.has<std::string>(key)) {
            if (user_attrs.count(full_key) == 0) {
                auto p = OStringProperty(user, key);
                p.setTimeSampling(1);
                user_attrs[full_key] = p;
            }
            std::any_cast<OStringProperty>(user_attrs[full_key]).set(ud.get2<std::string>(key));
        }
    }
}

static void write_faceset(std::shared_ptr<PrimitiveObject> prim, OPolyMeshSchema &mesh) {
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
            OFaceSet faceset = mesh.createFaceSet(faceSetNames[i]);
            OFaceSetSchema facesetSchema = faceset.getSchema ();
            OFaceSetSchema::Sample my_face_set_samp ( faceset_idxs[i] );
            // faceset is visible, doesn't change.
            facesetSchema.set ( my_face_set_samp );
            facesetSchema.setFaceExclusivity ( kFaceSetExclusive );
        }
    }
}
struct WriteAlembic2 : INode {
    OArchive archive;
    OPolyMesh meshyObj;
    OPoints pointsObj;
    std::string usedPath;
    std::map<std::string, std::any> attrs;
    std::map<std::string, std::any> user_attrs;

    virtual void apply() override {
        auto prim = get_input<PrimitiveObject>("prim");
        bool flipFrontBack = get_input2<int>("flipFrontBack");
        float fps = has_input("fps")? get_input2<float>("fps") : 24.0f;
        int frameid;
        if (has_input("frameid")) {
            frameid = get_input2<int>("frameid");
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
                "Zeno",
                "None"
            );
            archive.addTimeSampling(TimeSampling(1.0/fps, frame_start / fps));
            if (prim->polys.size() || prim->tris.size()) {
                meshyObj = OPolyMesh( OObject( archive, 1 ), "mesh" );
            }
            else {
                pointsObj = OPoints (OObject( archive, 1 ), "points");
            }
            attrs.clear();
            user_attrs.clear();
        }
        if (!(frame_start <= frameid && frameid <= frame_end)) {
            return;
        }
        if (archive.valid() == false) {
            zeno::makeError("Not init. Check whether in correct correct frame range.");
        }
        if (prim->polys.size() || prim->tris.size()) {
            // Create a PolyMesh class.
            OPolyMeshSchema &mesh = meshyObj.getSchema();
            write_faceset(prim, mesh);

            OCompoundProperty user = mesh.getUserProperties();
            write_user_data(user_attrs, "", prim, user);

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

            if (prim->loops.size()) {
                for (const auto& [start, size]: prim->polys) {
                    for (auto i = 0; i < size; i++) {
                        vertex_index_per_face.push_back(prim->loops[start + i]);
                    }
                    auto base = vertex_index_per_face.size() - size;
                    if (flipFrontBack) {
                        for (int j = 0; j < (size / 2); j++) {
                            std::swap(vertex_index_per_face[base + j], vertex_index_per_face[base + size - 1 - j]);
                        }
                    }
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
                        auto base = uv_indices.size() - size;
                        if (flipFrontBack) {
                            for (int j = 0; j < (size / 2); j++) {
                                std::swap(uv_indices[base + j], uv_indices[base + size - 1 - j]);
                            }
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
                    write_attrs(attrs, "", prim, mesh, mesh_samp);
                    mesh.set( mesh_samp );
                }
                else {
                    OPolyMeshSchema::Sample mesh_samp(
                    V3fArraySample( ( const V3f * )prim->verts.data(), prim->verts.size() ),
                            Int32ArraySample( vertex_index_per_face.data(), vertex_index_per_face.size() ),
                            Int32ArraySample( vertex_count_per_face.data(), vertex_count_per_face.size() ));
                    write_velocity(prim, mesh_samp);
                    write_normal(prim, mesh_samp);
                    write_attrs(attrs, "", prim, mesh, mesh_samp);
                    mesh.set( mesh_samp );
                }
            }
            else {
                for (auto i = 0; i < prim->tris.size(); i++) {
                    vertex_index_per_face.push_back(prim->tris[i][0]);
                    if (flipFrontBack) {
                        vertex_index_per_face.push_back(prim->tris[i][2]);
                        vertex_index_per_face.push_back(prim->tris[i][1]);
                    }
                    else {
                        vertex_index_per_face.push_back(prim->tris[i][1]);
                        vertex_index_per_face.push_back(prim->tris[i][2]);
                    }
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
                        if (flipFrontBack) {
                            uv_data.emplace_back(uv2[i][0], uv2[i][1]);
                            uv_data.emplace_back(uv1[i][0], uv1[i][1]);
                        }
                        else {
                            uv_data.emplace_back(uv1[i][0], uv1[i][1]);
                            uv_data.emplace_back(uv2[i][0], uv2[i][1]);
                        }
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
                    write_velocity(prim, mesh_samp);
                    write_normal(prim, mesh_samp);
                    write_attrs(attrs, "", prim, mesh, mesh_samp);
                    mesh.set( mesh_samp );
                } else {
                    OPolyMeshSchema::Sample mesh_samp(
                    V3fArraySample( ( const V3f * )prim->verts.data(), prim->verts.size() ),
                            Int32ArraySample( vertex_index_per_face.data(), vertex_index_per_face.size() ),
                            Int32ArraySample( vertex_count_per_face.data(), vertex_count_per_face.size() ));
                    write_velocity(prim, mesh_samp);
                    write_normal(prim, mesh_samp);
                    write_attrs(attrs, "", prim, mesh, mesh_samp);
                    mesh.set( mesh_samp );
                }
            }
        }
        else {
            OPointsSchema &points = pointsObj.getSchema();
            OCompoundProperty user = points.getUserProperties();
            write_user_data(user_attrs, "", prim, user);
            points.setTimeSampling(1);
            OPointsSchema::Sample samp(V3fArraySample( ( const V3f * )prim->verts.data(), prim->verts.size() ));
            std::vector<uint64_t> ids(prim->verts.size());
            if (prim->verts.attr_is<int>("id")) {
                auto &ids_ = prim->verts.attr<int>("id");
                for (auto i = 0; i < prim->verts.size(); i++) {
                    ids[i] = ids_[i];
                }
            }
            else {
                std::iota(ids.begin(), ids.end(), 0);
            }
            samp.setIds(Alembic::Abc::UInt64ArraySample(ids.data(), ids.size()));
            write_velocity(prim, samp);
            write_attrs(attrs, "", prim, points, samp);
            points.set( samp );
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
        {"fps"},
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
    std::map<std::string, OPoints> pointsObjs;
    std::map<std::string, std::any> attrs;
    std::map<std::string, std::any> user_attrs;

    virtual void apply() override {
        auto prims = get_input<ListObject>("prims")->get<PrimitiveObject>();
        bool flipFrontBack = get_input2<int>("flipFrontBack");
        float fps = has_input("fps")? get_input2<float>("fps") : 24.0f;
        int frameid;
        if (has_input("frameid")) {
            frameid = get_input2<int>("frameid");
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
                "Zeno",
                "None"
            );
            archive.addTimeSampling(TimeSampling(1.0/fps, frame_start / fps));
            meshyObjs.clear();
            pointsObjs.clear();
            attrs.clear();
            user_attrs.clear();
            for (auto prim: prims) {
                auto path = prim->userData().get2<std::string>("path");
                auto n_path = path.substr(5);
                auto subnames = split_str(n_path, '/');
                OObject oObject = OObject( archive, 1 );
                for (auto i = 0; i < subnames.size() - 1; i++) {
                    oObject = OObject( archive, subnames[i] );
                }
                if (prim->polys.size() || prim->tris.size()) {
                    meshyObjs[path] = OPolyMesh (oObject, subnames[subnames.size() - 1]);
                }
                else {
                    pointsObjs[path] = OPoints (oObject, subnames[subnames.size() - 1]);
                }
            }
        }
        if (!(frame_start <= frameid && frameid <= frame_end)) {
            return;
        }
        if (archive.valid() == false) {
            zeno::makeError("Not init. Check whether in correct correct frame range.");
        }
        for (auto prim: prims) {
            auto path = prim->userData().get2<std::string>("path");

            if (prim->polys.size() || prim->tris.size()) {
                // Create a PolyMesh class.
                OPolyMeshSchema &mesh = meshyObjs[path].getSchema();
                auto &ud = prim->userData();
                std::vector<std::string> faceSetNames;
                std::vector<std::vector<int>> faceset_idxs;
                write_faceset(prim, mesh);

                OCompoundProperty user = mesh.getUserProperties();
                write_user_data(user_attrs, path, prim, user);

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

                if (prim->loops.size()) {
                    for (const auto& [start, size]: prim->polys) {
                        for (auto i = 0; i < size; i++) {
                            vertex_index_per_face.push_back(prim->loops[start + i]);
                        }
                        auto base = vertex_index_per_face.size() - size;
                        if (flipFrontBack) {
                            for (int j = 0; j < (size / 2); j++) {
                                std::swap(vertex_index_per_face[base + j], vertex_index_per_face[base + size - 1 - j]);
                            }
                        }
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
                            auto base = uv_indices.size() - size;
                            if (flipFrontBack) {
                                for (int j = 0; j < (size / 2); j++) {
                                    std::swap(uv_indices[base + j], uv_indices[base + size - 1 - j]);
                                }
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
                        write_attrs(attrs, path, prim, mesh, mesh_samp);
                        mesh.set( mesh_samp );
                    }
                    else {
                        OPolyMeshSchema::Sample mesh_samp(
                        V3fArraySample( ( const V3f * )prim->verts.data(), prim->verts.size() ),
                                Int32ArraySample( vertex_index_per_face.data(), vertex_index_per_face.size() ),
                                Int32ArraySample( vertex_count_per_face.data(), vertex_count_per_face.size() ));
                        write_velocity(prim, mesh_samp);
                        write_normal(prim, mesh_samp);
                        write_attrs(attrs, path, prim, mesh, mesh_samp);
                        mesh.set( mesh_samp );
                    }
                }
                else {
                    for (auto i = 0; i < prim->tris.size(); i++) {
                        vertex_index_per_face.push_back(prim->tris[i][0]);
                        if (flipFrontBack) {
                            vertex_index_per_face.push_back(prim->tris[i][2]);
                            vertex_index_per_face.push_back(prim->tris[i][1]);
                        }
                        else {
                            vertex_index_per_face.push_back(prim->tris[i][1]);
                            vertex_index_per_face.push_back(prim->tris[i][2]);
                        }
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
                            if (flipFrontBack) {
                                uv_data.emplace_back(uv2[i][0], uv2[i][1]);
                                uv_data.emplace_back(uv1[i][0], uv1[i][1]);
                            }
                            else {
                                uv_data.emplace_back(uv1[i][0], uv1[i][1]);
                                uv_data.emplace_back(uv2[i][0], uv2[i][1]);
                            }
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
                        write_velocity(prim, mesh_samp);
                        write_normal(prim, mesh_samp);
                        write_attrs(attrs, path, prim, mesh, mesh_samp);
                        mesh.set( mesh_samp );
                    } else {
                        OPolyMeshSchema::Sample mesh_samp(
                        V3fArraySample( ( const V3f * )prim->verts.data(), prim->verts.size() ),
                                Int32ArraySample( vertex_index_per_face.data(), vertex_index_per_face.size() ),
                                Int32ArraySample( vertex_count_per_face.data(), vertex_count_per_face.size() ));
                        write_velocity(prim, mesh_samp);
                        write_normal(prim, mesh_samp);
                        write_attrs(attrs, path, prim, mesh, mesh_samp);
                        mesh.set( mesh_samp );
                    }
                }
            }
            else {
                OPointsSchema &points = pointsObjs[path].getSchema();
                OCompoundProperty user = points.getUserProperties();
                write_user_data(user_attrs, path, prim, user);
                points.setTimeSampling(1);
                OPointsSchema::Sample samp(V3fArraySample( ( const V3f * )prim->verts.data(), prim->verts.size() ));
                std::vector<uint64_t> ids(prim->verts.size());
                if (prim->verts.attr_is<int>("id")) {
                    auto &ids_ = prim->verts.attr<int>("id");
                    for (auto i = 0; i < prim->verts.size(); i++) {
                        ids[i] = ids_[i];
                    }
                }
                else {
                    std::iota(ids.begin(), ids.end(), 0);
                }
                samp.setIds(Alembic::Abc::UInt64ArraySample(ids.data(), ids.size()));
                write_velocity(prim, samp);
                write_attrs(attrs, path, prim, points, samp);
                points.set( samp );
            }
        }
    }
};

ZENDEFNODE(WriteAlembicPrims, {
    {
        {"list", "prims"},
        {"frameid"},
        {"writepath", "path", ""},
        {"int", "frame_start", "0"},
        {"int", "frame_end", "100"},
        {"fps"},
        {"bool", "flipFrontBack", "1"},
    },
    {
    },
    {},
    {"alembic"},
});

} // namespace
} // namespace zeno
