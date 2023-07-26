#include <zeno/zeno.h>
#include <zeno/utils/logger.h>
#include <zeno/utils/fileio.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/PrimitiveTools.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/UserData.h>
#include <zeno/extra/GlobalState.h>
#include <zeno/utils/string.h>
#include "rapidjson/document.h"

#include "draco/mesh/mesh.h"
#include "draco/core/decoder_buffer.h"
#include "draco/compression/decode.h"

namespace zeno {
namespace zeno_gltf {
    enum class ComponentType {
        GL_BYTE = 0x1400,
        GL_UNSIGNED_BYTE = 0x1401,
        GL_SHORT = 0x1402,
        GL_UNSIGNED_SHORT = 0x1403,
        GL_INT = 0x1404,
        GL_UNSIGNED_INT = 0x1405,
        GL_FLOAT = 0x1406,
        GL_DOUBLE = 0x140A,
    };
    enum class Type {
        SCALAR,
        VEC2,
        VEC3,
        VEC4,
    };
    struct Accessor {
        std::optional<int> bufferView;
        int count;
        ComponentType componentType;
        Type type;
        int byteOffset;
    };
    struct BufferView {
        int buffer;
        int byteOffset;
        int byteLength;
        int byteStride;
    };
    struct Material {
        std::string name;
    };
namespace fs = std::filesystem;

static std::shared_ptr<PrimitiveObject> read_gltf_model(std::string path) {
    rapidjson::Document doc;
    std::vector<std::vector<char>> buffers;
    if (zeno::ends_with(path, ".glb", false)) {
        auto data = zeno::file_get_binary(path);
        auto reader = BinaryReader(data);
        reader.seek_from_begin(8);
        auto total_len = reader.read_LE<int>();
        auto json_len = reader.read_LE<int>();
        reader.skip(4);
        std::string json = reader.read_string(json_len);
        zeno::file_put_content(path + ".json", json);
        doc.Parse(json.c_str());
        while (!reader.is_eof()) {
            auto len = reader.read_LE<int>();
            reader.skip(4);
            std::vector<char> buffer = reader.read_chunk(len);
            buffers.push_back(buffer);
        }
    }
    else {
        auto json = zeno::file_get_content(path);
        doc.Parse(json.c_str());

        zeno::log_info("buffers {}", doc["buffers"].Size());
        for (auto i = 0; i < doc["buffers"].Size(); i++) {
            fs::path p = path;
            auto parent = p.parent_path().string();
            std::string bin_path = parent + '/' + doc["buffers"][i]["uri"].GetString();
            auto buffer = zeno::file_get_binary(bin_path);
            zeno::log_info("{}", bin_path);
            buffers.push_back(buffer);
        }
    }
    std::vector<Material> materials;
    {
        for (auto i = 0; i < doc["materials"].Size(); i++) {
            const auto &m = doc["materials"][i];
            Material material;
            material.name = m["name"].GetString();
            materials.push_back(material);
        }
    }
    std::vector<Accessor> accessors;
    {
        for (auto i = 0; i < doc["accessors"].Size(); i++) {
            const auto & a = doc["accessors"][i];
            Accessor accessor;
            accessor.bufferView = std::nullopt;
            if (a.HasMember("bufferView")) {
                accessor.bufferView = a["bufferView"].GetInt();
            }
            accessor.byteOffset = a.HasMember("byteOffset")? a["byteOffset"].GetInt() : 0;
            accessor.count = a["count"].GetInt();
            accessor.componentType = ComponentType(a["componentType"].GetInt());
            std::string str_type = a["type"].GetString();
            if (str_type == "SCALAR") {
                accessor.type = Type::SCALAR;
            }
            else if (str_type == "VEC2") {
                accessor.type = Type::VEC2;
            }
            else if (str_type == "VEC3") {
                accessor.type = Type::VEC3;
            }
            else if (str_type == "VEC4") {
                accessor.type = Type::VEC4;
            }
            accessors.push_back(accessor);
        }
    }

    std::vector<BufferView> bufferViews;
    {
        for (auto i = 0; i < doc["bufferViews"].Size(); i++) {
            const auto & v = doc["bufferViews"][i];
            BufferView bufferView;
            bufferView.buffer = v["buffer"].GetInt();
            bufferView.byteOffset = v.HasMember("byteOffset")? v["byteOffset"].GetInt() : 0;
            bufferView.byteStride = v.HasMember("byteStride")? v["byteStride"].GetInt() : 0;
            bufferView.byteLength = v["byteLength"].GetInt();
            bufferViews.push_back(bufferView);
        }
    }
    auto prims = std::make_shared<zeno::ListObject>();
    for (auto mi = 0; mi < doc["meshes"].Size(); mi++) {
        auto prim = std::make_shared<zeno::PrimitiveObject>();
        const auto &mesh = doc["meshes"][mi];
        const auto &primitive = mesh["primitives"][0];
        if (primitive.HasMember("extensions") && primitive["extensions"].HasMember("KHR_draco_mesh_compression")) {
            const auto &draco_info = primitive["extensions"]["KHR_draco_mesh_compression"];
            auto bufferView = draco_info["bufferView"].GetInt();
            const auto &bv = bufferViews[bufferView];
            draco::Decoder dracoDecoder;
            draco::DecoderBuffer dracoDecoderBuffer;
            auto pData = &buffers[bv.buffer][bv.byteOffset];
            dracoDecoderBuffer.Init(reinterpret_cast<char *>(pData), bv.byteLength);
            auto type_statusor = draco::Decoder::GetEncodedGeometryType(&dracoDecoderBuffer);
            const draco::EncodedGeometryType geom_type = type_statusor.value();
            if (geom_type != draco::TRIANGULAR_MESH) {
                zeno::log_error("not draco::TRIANGULAR_MESH");
                throw std::runtime_error("not draco::TRIANGULAR_MESH");
            }

            draco::StatusOr<std::unique_ptr<draco::Mesh>> decoderStatus = dracoDecoder.DecodeMeshFromBuffer(&dracoDecoderBuffer);
            auto mesh = std::move(decoderStatus).value();
            auto vertexCount = mesh->num_points();
            prim->resize(vertexCount);
            auto faceCount = mesh->num_faces();
            prim->tris.resize(faceCount);

            // from https://github.com/google/draco/blob/master/src/draco/io/obj_encoder.cc
            const draco::PointAttribute *const att = mesh->GetNamedAttribute(draco::GeometryAttribute::POSITION);
            for (draco::AttributeValueIndex i(0); i < static_cast<uint32_t>(att->size()); ++i) {
                att->ConvertValue<float, 3>(i, prim->verts[i.value()].data());
            }
            for (draco::FaceIndex i(0); i < faceCount; ++i) {
                int _0 = att->mapped_index(mesh->face(i)[0]).value();
                int _1 = att->mapped_index(mesh->face(i)[1]).value();
                int _2 = att->mapped_index(mesh->face(i)[2]).value();
                prim->tris[i.value()] = {_0, _1, _2};
            }
        }
        else {
            {
                const auto &position = primitive["attributes"]["POSITION"].GetInt();
                const auto &acc = accessors[position];
                const auto &bv = bufferViews[acc.bufferView.value()];
                auto reader = BinaryReader(buffers[bv.buffer]);
                reader.seek_from_begin(bv.byteOffset + acc.byteOffset);
                prim->resize(acc.count);
                for (auto i = 0; i < acc.count; i++) {
                    prim->verts[i] = reader.read_LE<vec3f>();
                }
            }
            {
                auto index = primitive["indices"].GetInt();
                const auto &acc = accessors[index];
                const auto &bv = bufferViews[acc.bufferView.value()];
                auto reader = BinaryReader(buffers[bv.buffer]);
                reader.seek_from_begin(bv.byteOffset + acc.byteOffset);
                auto count = acc.count / 3;
                prim->tris.resize(count);
                if (acc.componentType == ComponentType::GL_UNSIGNED_SHORT) {
                    for (auto i = 0; i < count; i++) {
                        auto f0 = reader.read_LE<uint16_t>();
                        auto f1 = reader.read_LE<uint16_t>();
                        auto f2 = reader.read_LE<uint16_t>();
                        prim->tris[i] = {f0, f1, f2};
                    }
                }
                else if (acc.componentType == ComponentType::GL_UNSIGNED_INT) {
                    for (auto i = 0; i < count; i++) {
                        prim->tris[i] = reader.read_LE<vec3i>();
                    }
                }
                else {
                    zeno::log_info("not support componentType: {}", int(acc.componentType));
                }
            }
        }
        prims->arr.push_back(prim);
    }
    auto prim = primMerge(prims->getRaw<PrimitiveObject>());
    auto &ud = prim->userData();
    for (auto i = 0; i < materials.size(); i++) {
        ud.set2(zeno::format("Material_{}", i), materials[i].name);
    }
    return prim;
}

struct LoadGLTFModel : INode {
    virtual void apply() override {
        auto path = get_input2<std::string>("path");
        auto prim = read_gltf_model(path);
        if (get_input2<bool>("cm unit")) {
            for (auto & vert : prim->verts) {
                vert *= 0.01f;
            }
        }
        set_output("prim", std::move(prim));
    }
};

ZENDEFNODE(LoadGLTFModel, {
    {
        {"readpath", "path"},
        {"bool", "cm unit", "0"},
    },
    {
        "prim"
    },
    {},
    {"alembic"},
});

struct ReadTile : INode {
    virtual void apply() override {
        auto path = get_input2<std::string>("path");

        fs::path p = path;
        auto parent = p.parent_path().string();
        auto json = zeno::file_get_content(path);
        rapidjson::Document doc;
        doc.Parse(json.c_str());
        const auto& root = doc["root"];
        const auto& children = root["children"];

        zeno::log_info("count {}", children.Size());
        auto list = std::make_shared<ListObject>();

        for (auto i = 0; i < children.Size(); i++) {
//        for (auto i = 0; i < 10; i++) {
            const auto &c = children[i];
            std::string uri = c["content"]["uri"].GetString();
            uri = parent + "/" + uri.substr(0, uri.size() - 4) + "glb";
            const auto &box = c["boundingVolume"]["box"];
            vec3f ct = {
                    box[0].GetFloat(),
                    box[1].GetFloat(),
                    box[2].GetFloat(),
            };
            auto prim = read_gltf_model(uri);
            vec3f bmin, bmax;
            std::tie(bmin, bmax) = primBoundingBox(prim.get());
            vec3f bc = (bmin + bmax) / 2;
            for (auto i = 0; i < prim->verts.size(); i++) {
                prim->verts[i] += - bc + vec3f(ct[0], ct[2], -ct[1]);
            }
            list->arr.push_back(prim);
        }
        auto pPrims = list->getRaw<PrimitiveObject>();
        auto output = primMerge(pPrims);
        set_output("prim", std::move(output));
    }
};

ZENDEFNODE(ReadTile, {
    {
        {"readpath", "path"},
        {"frame"},
    },
    {
        "prim",
    },
    {},
    {"alembic"},
});
}


} // namespace zeno
