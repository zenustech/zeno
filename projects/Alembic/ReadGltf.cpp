#include <zeno/zeno.h>
#include <zeno/utils/logger.h>
#include <zeno/types/StringObject.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/PrimitiveTools.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/UserData.h>
#include <zeno/extra/GlobalState.h>
#include <cstring>
#include <cstdio>
#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <tinygltf/tiny_gltf.h>

using namespace tinygltf;

namespace zeno {


static vec3f read_vec3f(unsigned char *data) {
    float x = *(float*)data;
    float y = *(float*)(data + 4);
    float z = *(float*)(data + 8);
    return vec3f(x, y, z);
}

struct ReadGltf : INode {
    virtual void apply() override {
        auto path = get_input2<std::string>("path");
        auto prim = std::make_shared<PrimitiveObject>();
        tinygltf::Model model;
        tinygltf::TinyGLTF loader;
        std::string err;
        std::string warn;

        bool ret = loader.LoadASCIIFromFile(&model, &err, &warn, path);
//bool ret = loader.LoadBinaryFromFile(&model, &err, &warn, argv[1]); // for binary glTF(.glb)

        if (!warn.empty()) {
            printf("Warn: %s\n", warn.c_str());
        }

        if (!err.empty()) {
            printf("Err: %s\n", err.c_str());
        }

        if (!ret) {
            printf("Failed to parse glTF\n");
        }

        {
            for (size_t i = 0; i < model.bufferViews.size(); ++i) {
                zeno::log_info("======= {} ==========", i);
                const tinygltf::BufferView &bufferView = model.bufferViews[i];
                if (bufferView.target == 0) {
                    zeno::log_info("WARN: bufferView.target is zero");
                    continue;
                }

                const tinygltf::Buffer &buffer = model.buffers[bufferView.buffer];
                zeno::log_info("bufferview.target {}", bufferView.target);
                zeno::log_info("buffer.data.size = {}", buffer.data.size());
                zeno::log_info("bufferview.byteOffset = {}", bufferView.byteOffset);
            }
        }

        zeno::log_info("model.nodes.size: {}", (int)model.nodes.size());
        const tinygltf::Scene &scene = model.scenes[model.defaultScene];
        zeno::log_info("scene.nodes.size: {}", (int)scene.nodes.size());
        auto node_index = scene.nodes[0];
        auto mesh_index = model.nodes[node_index].mesh;
        zeno::log_info("mesh_index {}", mesh_index);
        zeno::log_info("mesh size {}", (int)model.meshes.size());
        Mesh &mesh = model.meshes[mesh_index];
        zeno::log_info("mesh.primitives.size: {}", (int)mesh.primitives.size());

        for (size_t i = 0; i < mesh.primitives.size(); ++i) {
            Primitive primitive = mesh.primitives[i];
            zeno::log_info("{}", primitive.indices);
//            Accessor indexAccessor = model.accessors[primitive.indices];
            for (auto &attrib : primitive.attributes) {
                tinygltf::Accessor accessor = model.accessors[attrib.second];
                int vertex_count = accessor.count;
                zeno::log_info("vertex_count: {}", vertex_count);
                int byteStride = accessor.ByteStride(model.bufferViews[accessor.bufferView]);
//                glBindBuffer(GL_ARRAY_BUFFER, vbos[accessor.bufferView]);

                int size = 0;
                if (accessor.type == TINYGLTF_TYPE_SCALAR) {
                    size = 1;
                } else if (accessor.type == TINYGLTF_TYPE_VEC2) {
                    size = 2;
                } else if (accessor.type == TINYGLTF_TYPE_VEC3) {
                    size = 3;
                } else if (accessor.type == TINYGLTF_TYPE_VEC4) {
                    size = 4;
                } else if (accessor.type == TINYGLTF_TYPE_VECTOR) {
                    size = 4;
                } else if (accessor.type == TINYGLTF_TYPE_MATRIX) {
                    size = 16;
                } else if (accessor.type == TINYGLTF_TYPE_MAT2) {
                    size = 4;
                } else if (accessor.type == TINYGLTF_TYPE_MAT3) {
                    size = 9;
                } else if (accessor.type == TINYGLTF_TYPE_MAT4) {
                    size = 16;
                }
                zeno::log_info("{} {}, size {}, byteStride {}", attrib.first, attrib.second, (int)size, byteStride);
                // attrib.first: attr name
                // attrib.second: attr accessor index
                // size: component count in tuple
                // byteStride: byte per tuple, Specifies the byte offset between consecutive generic vertex attributes
                // accessor.componentType : int, float and so on

                int vaa = -1;
                if (attrib.first.compare("POSITION") == 0) vaa = 0;
                if (attrib.first.compare("NORMAL") == 0) vaa = 1;
                if (attrib.first.compare("TEXCOORD_0") == 0) vaa = 2;

                if (attrib.first.compare("POSITION") == 0) {
                    for (auto j = 0; j < vertex_count; j++) {
                        auto& bufferView = model.bufferViews[accessor.bufferView];
                        auto& buffer = model.buffers[bufferView.buffer];
                        vec3f pos = read_vec3f(buffer.data.data() + bufferView.byteOffset + j * 12);
                        zeno::log_info("pos {}", pos);
                    }
                }

//                if (vaa > -1) {
//                    glEnableVertexAttribArray(vaa);
//                    glVertexAttribPointer(vaa, size, accessor.componentType,
//                                          accessor.normalized ? GL_TRUE : GL_FALSE,
//                                          byteStride, BUFFER_OFFSET(accessor.byteOffset));
//                } else
//                    std::cout << "vaa missing: " << attrib.first << std::endl;
            }
        }



        set_output("prim", std::move(prim));
    }
};

ZENDEFNODE(ReadGltf, {
    {
        {"readpath", "path"},
    },
    {"prim"},
    {},
    {"gltf"},
});
}