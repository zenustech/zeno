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
    int bufferView;
    int count;
    ComponentType componentType;
    Type type;
};
struct BufferView {
    int buffer;
    int byteOffset;
    int byteLength;
    int byteStride;
};

struct GLBHeader {
    uint32_t magic;
    uint32_t version;
    uint32_t length;
};

struct ReadGLTF : zeno::INode {
    virtual void apply() override {
        auto path = get_input2<std::string>("path");
        std::vector<std::vector<char>> buffers;
        rapidjson::Document root;

        if (zeno::ends_with(path, ".glb", false)) {
            std::ifstream file(path.c_str(), std::ios::binary);
            // Read 12-byte header
            GLBHeader header;
            file.read(reinterpret_cast<char *>(&header), sizeof(GLBHeader));
            if (header.magic != 0x46546C67) {
                return;
            }
            // Read JSON chunk
            uint32_t jsonChunkLength;
            file.read(reinterpret_cast<char *>(&jsonChunkLength), sizeof(uint32_t));
            uint32_t jsonChunkType;
            file.read(reinterpret_cast<char *>(&jsonChunkType), sizeof(uint32_t));

            std::vector<char> jsonBuffer(jsonChunkLength);
            file.read(jsonBuffer.data(), jsonChunkLength);

            // Parse JSON
            std::string json(jsonBuffer.begin(), jsonBuffer.end());
            root.Parse(json.c_str());
            //        std::ofstream outputFile("d:\\1.json");
            //        outputFile << json;

            // Read binary chunk
            uint32_t binaryChunkLength;
            file.read(reinterpret_cast<char *>(&binaryChunkLength), sizeof(uint32_t));
            uint32_t binaryChunkType;
            file.read(reinterpret_cast<char *>(&binaryChunkType), sizeof(uint32_t));

            std::vector<char> binaryBuffer(binaryChunkLength);
            file.read(binaryBuffer.data(), binaryChunkLength);

            // Parse Bin
            buffers.push_back(binaryBuffer);
        }
        else {
            auto json = zeno::file_get_content(path);
            root.Parse(json.c_str());

            zeno::log_info("buffers {}", root["buffers"].Size());
            for (auto i = 0; i < root["buffers"].Size(); i++) {
                std::filesystem::path p = path;
                auto parent = p.parent_path().string();
                std::string bin_path = parent + '/' + root["buffers"][i]["uri"].GetString();
                auto buffer = zeno::file_get_binary(bin_path);
                zeno::log_info("{}", bin_path);
                buffers.push_back(buffer);
            }
        }
        std::vector<Accessor> accessors;
        {
            for (auto i = 0; i < root["accessors"].Size(); i++) {
                const auto & a = root["accessors"][i];
                Accessor accessor;
                accessor.bufferView = a.HasMember("bufferView")? a["bufferView"].GetInt():0;
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
            for (auto i = 0; i < root["bufferViews"].Size(); i++) {
                const auto & v = root["bufferViews"][i];
                BufferView bufferView;
                bufferView.buffer = v["buffer"].GetInt();
                bufferView.byteOffset = v.HasMember("byteOffset")? v["byteOffset"].GetInt() : 0;
                bufferView.byteStride = v.HasMember("byteStride")? v["byteStride"].GetInt() : 0;
                bufferView.byteLength = v["byteLength"].GetInt();
                bufferViews.push_back(bufferView);
            }
        }
        auto prim = std::make_shared<zeno::PrimitiveObject>();
        {
            int pvert = 0;
            int pnrm = 0;
            int ptris = 0;
            int puv = 0;
            for(size_t mi = 0;mi < root["meshes"].Size();mi++){
                const auto &mesh = root["meshes"][mi];
                const auto &primitive = mesh["primitives"][0];
                {
                    const auto &position = primitive["attributes"]["POSITION"].GetInt();
                    const auto &acc = accessors[position];
                    const auto &bv = bufferViews[acc.bufferView];
                    auto reader = BinaryReader(buffers[bv.buffer]);
                    reader.seek_from_begin(bv.byteOffset);
                    prim->resize(pvert + acc.count);
                    for (auto i = pvert; i < pvert + acc.count; i++) {
                        prim->verts[i] = reader.read_LE<vec3f>();
                    }
                    pvert += acc.count;
                }
                {
                    const auto &normal = primitive["attributes"]["NORMAL"].GetInt();
                    const auto &acc = accessors[normal];
                    const auto &bv = bufferViews[acc.bufferView];
                    auto reader = BinaryReader(buffers[bv.buffer]);
                    reader.seek_from_begin(bv.byteOffset);
                    auto count = acc.count/3;
                    auto &n = prim->verts.add_attr<vec3f>("nrm");
                    if (acc.componentType == ComponentType::GL_FLOAT) {
                        for (auto i = pnrm; i < pnrm + count; i++) {
                            n[i] = reader.read_LE<vec3f>();
                        }
                        pnrm += count;
                    }
                    else if (acc.componentType == ComponentType::GL_DOUBLE) {
                        for (auto i = pnrm; i < pnrm + count; i++) {
                            auto n0 = float(reader.read_LE<double>());
                            auto n1 = float(reader.read_LE<double>());
                            auto n2 = float(reader.read_LE<double>());
                            n[i] = {n0, n1, n2};
                        }
                        pnrm += count;
                    }
                    else {
                        zeno::log_info("no support componentType for normal: {}", int(acc.componentType));
                    }
                }
                {
                    auto index = primitive["indices"].GetInt();
                    const auto &acc = accessors[index];
                    const auto &bv = bufferViews[acc.bufferView];
                    auto reader = BinaryReader(buffers[bv.buffer]);
                    reader.seek_from_begin(bv.byteOffset);
                    auto count = acc.count / 3;
                    prim->tris.resize(ptris + count);
                    if (acc.componentType == ComponentType::GL_SHORT) {
                        for (auto i = ptris; i < ptris + count; i++) {
                            auto f0 = reader.read_LE<int16_t>();
                            auto f1 = reader.read_LE<int16_t>();
                            auto f2 = reader.read_LE<int16_t>();
                            prim->tris[i] = {f0, f1, f2};
                        }
                    }
                    else if (acc.componentType == ComponentType::GL_UNSIGNED_SHORT) {
                        for (auto i = ptris; i < ptris + count; i++) {
                            auto f0 = reader.read_LE<uint16_t>();
                            auto f1 = reader.read_LE<uint16_t>();
                            auto f2 = reader.read_LE<uint16_t>();
                            prim->tris[i] = {f0, f1, f2};
                        }
                    }
                    else if (acc.componentType == ComponentType::GL_INT) {
                        for (auto i = ptris; i < ptris + count; i++) {
                            prim->tris[i] = reader.read_LE<vec3i>();
                        }
                    }
                    else if (acc.componentType == ComponentType::GL_UNSIGNED_INT) {
                        for (auto i = ptris; i < ptris + count; i++) {
                            prim->tris[i] = reader.read_LE<vec3i>();
                        }
                    }
                    else {
                        zeno::log_info("not support componentType for face: {}", int(acc.componentType));
                    }
                    ptris += count;
                }
                {
                    int T0 = 0;
                    if (primitive["attributes"].HasMember("TEXCOORD_0")){
                        T0 = primitive["attributes"]["TEXCOORD_0"].GetInt();
                    }
                    const auto &acc = accessors[T0];
                    const auto &bv = bufferViews[acc.bufferView];
                    auto reader = BinaryReader(buffers[bv.buffer]);
                    reader.seek_from_begin(bv.byteOffset);
                    auto count = acc.count/4;
                    prim->uvs.resize(puv + count);
                    auto &uv = prim->uvs;
                    if (acc.componentType == ComponentType::GL_FLOAT) {
                        for (auto i = puv; i < puv + count; i++) {
                            uv[i] = reader.read_LE<vec2f>();
                        }
                    }
                    else if (acc.componentType == ComponentType::GL_DOUBLE) {
                        for (auto i = puv; i < puv + count; i++) {
                            auto n0 = float(reader.read_LE<double>());
                            auto n1 = float(reader.read_LE<double>());
                            uv[i] = {n0,n1};
                        }
                    }
                    else {
                        zeno::log_info("no support componentType for uv: {}", int(acc.componentType));
                    }
                    puv += count;
                }
            }
        }
        set_output("prim", std::move(prim));
    }
};
ZENDEFNODE(ReadGLTF, {
    {
        { "readpath", "path" },
    },
    {
        { "prim" },
    },
    {},
    { "primitive" },
});
}
}