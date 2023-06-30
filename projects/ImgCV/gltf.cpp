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
#include <rapidjson/document.h>
#include "rapidjson/writer.h"
#include "rapidjson/stringbuffer.h"
#include <string>
#include <vector>

namespace zeno {

struct ReadGLTF : zeno::INode {
    virtual void apply() override {
        auto path = get_input2<std::string>("path");
        auto prim = std::make_shared<zeno::PrimitiveObject>();
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
        int vcount = 0;
        int nrmcount = 0;
        int tricount = 0;

//        int PrimitiveCount = 1;
        int PrimitiveCount = root["meshes"].Size();
        zeno::log_info("PrimitiveCount:{}",PrimitiveCount);
        for(size_t PrimitiveIdx = 0; PrimitiveIdx < PrimitiveCount; PrimitiveIdx++){
            int VertexCount = 0;
            std::vector<vec3f> vertices1;
            int NormalCount = 0;
            std::vector<vec3f> normalo;
            int IndexCount = 0;
            std::vector<unsigned short> indiceso;
// verts
            const rapidjson::Value &primitives = root["meshes"][PrimitiveIdx]["primitives"];
            const rapidjson::Value &attributes = primitives[PrimitiveIdx]["attributes"];
            int AccessorsIdxOfPosition = attributes["POSITION"].GetInt();
            const rapidjson::Value &PositionAccessor = root["accessors"][AccessorsIdxOfPosition];
            int BufferViewsIdxOfPosition = PositionAccessor["bufferView"].GetInt();
            const rapidjson::Value &VertBufferView = root["bufferViews"][BufferViewsIdxOfPosition];
            int VertByteOffset = 0;
            if(VertBufferView.HasMember("byteOffset")){
                VertByteOffset = VertBufferView["byteOffset"].GetInt();
            }
            int VertByteLength = VertBufferView["byteLength"].GetInt();
            VertexCount = PositionAccessor["count"].GetInt();

            int VertBufferIdx = VertBufferView["buffer"].GetInt();
/////////////////////
//            zeno::log_info("buffer.count:{},buffers.count:{}",static_cast<int>(root["bufferViews"].MemberCount()),static_cast<int>(root["buffers"].MemberCount()));
            const rapidjson::Value &buffers = root["buffers"][VertBufferIdx];
            std::string uri = buffers["uri"].GetString();

            size_t lastBackslashPos = path.find_last_of("/\\");
            if (lastBackslashPos != std::string::npos) {
                path.erase(lastBackslashPos + 1);
            }
            auto vpath = path;
            vpath.append(uri);
            std::ifstream bufferFile(vpath.c_str(), std::ios::binary);
            zeno::log_info("vpath.c_str():{}",vpath.c_str());
            if (!bufferFile.is_open()) {
                zeno::log_info("Failed to open buffer file");
            }
            bufferFile.seekg(VertByteOffset, std::ios::beg);
            vertices1.resize(VertexCount);
            normalo.resize(50000);
            bufferFile.read(reinterpret_cast<char *>(vertices1.data()), VertByteLength);
    //normal
            int AccessorsIdxOfNormal = attributes["NORMAL"].GetInt();
            const rapidjson::Value &NormalAccessor = root["accessors"][AccessorsIdxOfNormal];
            int BufferViewIdxOfNormal = NormalAccessor["bufferView"].GetInt();
            const rapidjson::Value &NormalBufferViews = root["bufferViews"][BufferViewIdxOfNormal];
            int NormalByteOffset = 0;
            if(NormalBufferViews.HasMember("byteOffset")){
                NormalByteOffset = NormalBufferViews["byteOffset"].GetInt();
            }
            int NormalByteLength = NormalBufferViews["byteLength"].GetInt();
            NormalCount = NormalAccessor["count"].GetInt();
// ///////////////////
            normalo.resize(NormalCount);
            bufferFile.seekg(NormalByteOffset, std::ios::beg);
            bufferFile.read(reinterpret_cast<char *>(normalo.data()), NormalByteLength);
//tris
            int AccessorsIdxOfIndices = primitives[PrimitiveIdx]["indices"].GetInt();
            const rapidjson::Value &IndexAccessor = root["accessors"][AccessorsIdxOfIndices];
            int BufferViewIdxOfIndices = IndexAccessor["bufferView"].GetInt();
            const rapidjson::Value &IndexBufferViews = root["bufferViews"][BufferViewIdxOfIndices];
            int IndexByteOffset = 0;
            if(IndexBufferViews.HasMember("byteOffset")){
                IndexByteOffset = IndexBufferViews["byteOffset"].GetInt();
            }
            int IndexByteLength = IndexBufferViews["byteLength"].GetInt();
            IndexCount = IndexAccessor["count"].GetInt();
            indiceso.resize(IndexCount);

    ////////////////////////

            bufferFile.seekg(IndexByteOffset, std::ios::beg);
            bufferFile.read(reinterpret_cast<char*>(indiceso.data()), IndexByteLength);
            bufferFile.close();
            zeno::log_info("VertexCount:{}",VertexCount);
            prim->resize(VertexCount);
            auto &pos = prim->verts;
            for (size_t i = vcount; i < vcount + VertexCount; i++) {
                zeno::vec3f v = {static_cast<float>(vertices1[i][0]),
                                 static_cast<float>(vertices1[i][1]),
                                 static_cast<float>(vertices1[i][2])};
                prim->verts[i] = v;
            }
            vcount += VertexCount;
            auto &n = prim->verts.add_attr<vec3f>("nrm");
            for (size_t i = nrmcount; i < nrmcount + NormalCount; i++) {
                zeno::vec3f no = {static_cast<float>(normalo[i][0]),
                                  static_cast<float>(normalo[i][1]),
                                  static_cast<float>(normalo[i][2])};
                n[i] = no;
            }
            nrmcount += NormalCount;
            auto &tri = prim->tris;
            prim->tris.resize(IndexCount);
            for(size_t i = tricount;i < tricount + IndexCount;i+=3){
                zeno::vec3i tr = {indiceso[i],indiceso[i+1],indiceso[i+2]};
                tri[(int)(i/3)] = tr;
            }
            tricount += IndexCount;
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

struct ReadGLB : zeno::INode {
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
            zeno::log_info("vpath.c_str():{}",vpath.c_str());
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
//            if (bufferFile) {
            bufferFile.seekg(0, std::ios::end);
            std::streampos fileSize = bufferFile.tellg();
            bufferFile.seekg(0, std::ios::beg);
            indiceso.resize(fileSize / sizeof(unsigned short));
            bufferFile.read(reinterpret_cast<char*>(indiceso.data()), fileSize);
//            }
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

ZENDEFNODE(ReadGLB, {
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
