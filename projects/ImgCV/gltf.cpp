#include "json.hpp"
#include <iostream>
#include <fstream>
#include <zeno/zeno.h>
#include <zeno/NumericObject.h>
#include <zeno/MeshObject.h>
#include <zeno/StringObject.h>
#include <cstring>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/StringObject.h>
#include <zeno/utils/logger.h>
#include <zeno/utils/vec.h>
#include <cassert>
#include <fstream>
#include <rapidjson\document.h>
#include "rapidjson/writer.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/document.h"
#include <string>
#include <vector>

namespace zeno {

struct ReadGLTF : zeno::INode {
    virtual void apply() override {
        auto path = get_input2<std::string>("path");
        auto prim = std::make_shared<zeno::PrimitiveObject>();
        auto &pos = prim->verts;
        std::ifstream gltfFile(path.c_str());
        if (!gltfFile.is_open()) {
            zeno::log_info("Failed to open .gltf file:{}", path.c_str());
        }
        std::string gltfContent((std::istreambuf_iterator<char>(gltfFile)),
                                std::istreambuf_iterator<char>());
        gltfFile.close();
        rapidjson::Document root;
        root.Parse(gltfContent.c_str());

        if (root.HasParseError()) {
            zeno::log_info("Failed to parse .gltf file: {}", path.c_str());
        }
//        int PrimitiveIdx = 0;
        int PrimitiveCount = root["meshes"].Size();
        zeno::log_info("PrimitiveCount:{}",PrimitiveCount);
        for(size_t PrimitiveIdx = 0; PrimitiveIdx < PrimitiveCount; PrimitiveIdx++){
// verts
            const rapidjson::Value &primitives = root["meshes"][PrimitiveIdx]["primitives"];
            const rapidjson::Value &attributes = primitives[PrimitiveIdx]["attributes"];
            int BufferIdxOfPosition = attributes["POSITION"].GetInt();
            const rapidjson::Value &accessor = root["accessors"][BufferIdxOfPosition];
            const rapidjson::Value &bufferViews = root["bufferViews"][BufferIdxOfPosition];
            int VertByteOffset = bufferViews["byteOffset"].GetInt();
            int VertByteLength = bufferViews["byteLength"].GetInt();
            int VertBufferIndex = bufferViews["buffer"].GetInt();
            int VertexCount = accessor["count"].GetInt();
    /////////////////////

            const rapidjson::Value &buffers = root["buffers"][VertBufferIndex];
            std::string uri = buffers["uri"].GetString();

            size_t lastBackslashPos = path.find_last_of("/\\");
            if (lastBackslashPos != std::string::npos) {
                path.erase(lastBackslashPos + 1);
            }
            auto vpath = path;
            vpath.append(uri);
            std::ifstream bufferFile(vpath.c_str(), std::ios::binary);
            if (!bufferFile.is_open()) {
                zeno::log_info("Failed to open buffer file");
            }
            bufferFile.seekg(VertByteOffset, std::ios::beg);
            std::vector<vec3f> vertices1(VertexCount);
            bufferFile.read(reinterpret_cast<char *>(vertices1.data()), VertByteLength);

            prim->verts.resize(VertexCount);
            prim->uvs.resize(VertexCount);
            auto &uvs = prim->uvs;
            auto &nrm = prim->verts.add_attr<vec3f>("nrm");
            for (size_t i = 0; i < VertexCount; i++) {
                zeno::vec3f v = {static_cast<float>(vertices1[i][0]),
                                 static_cast<float>(vertices1[i][1]),
                                 static_cast<float>(vertices1[i][2])};
                pos[i] = v;
            }
    //normal
            int BufferIdxOfNormal = attributes["NORMAL"].GetInt();
            const rapidjson::Value &NormalAccessor = root["accessors"][BufferIdxOfNormal];
            const rapidjson::Value &NormalBufferViews = root["bufferViews"][BufferIdxOfNormal];
            int NormalByteOffset = NormalBufferViews["byteOffset"].GetInt();
            int NormalByteLength = NormalBufferViews["byteLength"].GetInt();
            int NormalBufferIndex = NormalBufferViews["buffer"].GetInt();
            int NormalCount = NormalAccessor["count"].GetInt();
    /////////////////////
            const rapidjson::Value &NormalBuffers = root["buffers"][NormalBufferIndex];
            bufferFile.seekg(NormalByteOffset, std::ios::beg);
            std::vector<vec3f> normalo(NormalCount);
            bufferFile.read(reinterpret_cast<char *>(normalo.data()), NormalByteLength);
            auto &n = prim->verts.add_attr<vec3f>("nrm");
            for (size_t i = 0; i < NormalCount; i++) {
                zeno::vec3f no = {static_cast<float>(normalo[i][0]),
                                  static_cast<float>(normalo[i][1]),
                                  static_cast<float>(normalo[i][2])};
                n[i] = no;
            }
    //tris
            int BufferIdxOfIndices = primitives[PrimitiveIdx]["indices"].GetInt();
            const rapidjson::Value &IndexBufferViews = root["bufferViews"][BufferIdxOfIndices];
            int IndexByteOffset = IndexBufferViews["byteOffset"].GetInt();
            int IndexByteLength = IndexBufferViews["byteLength"].GetInt();
            int IndexBufferIndex = IndexBufferViews["buffer"].GetInt();
            const rapidjson::Value &indexBuffers = root["buffers"][IndexBufferIndex];
    ////////////////////////
            int indexSizeInBytes = 2;
            int indexCount = IndexByteLength / indexSizeInBytes / 3;
            bufferFile.seekg(IndexByteOffset, std::ios::beg);
            std::vector<unsigned short> indiceso;
//            indiceso.resize(indexCount);
//            bufferFile.read(reinterpret_cast<char *>(indiceso.data()), IndexByteLength);
//            bufferFile.close();
//            auto &tri = prim->tris;
//            prim->tris.resize(indexCount+1);
//            for (size_t i = 0; i < indexCount * 3; i += 3) {
//                zeno::vec3i tr = {indiceso[i], indiceso[i + 1], indiceso[i + 2]};
//                tri[(int)(i/3)] = tr;
//            }
            if (bufferFile) {
                bufferFile.seekg(0, std::ios::end);
                std::streampos fileSize = bufferFile.tellg();
                bufferFile.seekg(0, std::ios::beg);
                indiceso.resize(fileSize / sizeof(unsigned short));
                bufferFile.read(reinterpret_cast<char*>(indiceso.data()), fileSize);
            }
            auto &tri = prim->tris;
            prim->tris.resize(indexCount);
            for(size_t i = 0;i < indexCount*3;i+=3){
                zeno::vec3i tr = {indiceso[i],indiceso[i+1],indiceso[i+2]};
                tri[(int)(i/3)] = tr;
            }
            bufferFile.close();

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
