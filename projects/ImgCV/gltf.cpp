#include "json.hpp"
#include <iostream>
#include <fstream>
#include <zeno/zeno.h>
#include <zeno/NumericObject.h>
#include <zeno/MeshObject.h>
#include <zeno/StringObject.h>
#include <cstring>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/DictObject.h>
#include <zeno/types/StringObject.h>
#include <zeno/types/PrimitiveTools.h>
#include <zeno/utils/string.h>
#include <zeno/utils/logger.h>
#include <zeno/utils/vec.h>
#include <cstdlib>
#include <cassert>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <rapidjson\document.h>
//#include <json/json.h>
#include "rapidjson/writer.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/document.h"
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/NumericObject.h>
#include <string>
#include <vector>

namespace zeno {

struct Vertex1 {
    float position[3];
    float normal[3];
    float texCoord[2];
};

struct ReadGLTF : zeno::INode {
    virtual void apply() override {
        auto path = get_input2<std::string>("path");
        auto prim = std::make_shared<zeno::PrimitiveObject>();
        auto &pos = prim->verts;
        std::ifstream gltfFile(path.c_str());
        if (!gltfFile.is_open()) {
            zeno::log_info( "Failed to open .gltf file:{}" , path.c_str());
        }
        std::string gltfContent((std::istreambuf_iterator<char>(gltfFile)),
                                std::istreambuf_iterator<char>());
        gltfFile.close();
        rapidjson::Document root;
        root.Parse(gltfContent.c_str());

        if (root.HasParseError()) {
            zeno::log_info("Failed to parse .gltf file: {}",path.c_str());
        }
        const rapidjson::Value& accessors = root["accessors"];
        const rapidjson::Value& bufferViews = root["bufferViews"];
        const rapidjson::Value& buffers = root["buffers"];

        const int verticesAccessorIndex = 0;
        const int verticesBufferViewIndex = accessors[verticesAccessorIndex]["bufferView"].GetInt();
        const int verticesBufferIndex = bufferViews[verticesBufferViewIndex]["buffer"].GetInt();
        const int byteOffset = bufferViews[verticesBufferViewIndex]["byteOffset"].GetInt();
        const int vertexCount = accessors[verticesAccessorIndex]["count"].GetInt();
        const int bufferLength = buffers[verticesBufferIndex]["byteLength"].GetInt();
        const std::string& uri = buffers[verticesBufferIndex]["uri"].GetString();

        size_t lastBackslashPos = path.find_last_of("/\\");
        if (lastBackslashPos != std::string::npos) {
            path.erase(lastBackslashPos + 1);
        }
//        std::string bpath = path.substr(0, path.length() - 5);
//        std::string str = ".bin";
        path.append(uri);
        std::ifstream bufferFile(path.c_str(), std::ios::binary);
        if (!bufferFile.is_open()) {
            zeno::log_info("Failed to open buffer file");
        }
        bufferFile.seekg(byteOffset, std::ios::beg);
        std::vector<Vertex1> vertices(vertexCount);
        bufferFile.read(reinterpret_cast<char*>(vertices.data()), vertexCount * sizeof(Vertex1));
        bufferFile.close();
        prim->verts.resize(vertexCount);
        auto &nrm = prim->verts.add_attr<vec3f>("nrm");
        for(size_t i = 0;i < vertexCount;i++){
            zeno::vec3f v = {vertices[i].position[0],vertices[i].position[1],vertices[i].position[2]};
            pos[i] = v;
            zeno::vec3f n = {vertices[i].normal[0],vertices[i].normal[1],vertices[i].normal[2]};
            nrm[i] = n;
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
