#pragma once
#include "ClusterCache.h"
#include "ClusterUri.h"
#include <algorithm>
#include <cstring>
#include <fstream>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <meshoptimizer.h>
#include <unordered_map>

namespace zeno::cluster {
using Json = nlohmann::json;
namespace fs = std::filesystem;
struct Source {
    const Json &scene;
    fs::path root;
    std::unordered_map<size_t, std::vector<uint8_t>> decoded;
    const std::vector<uint8_t> &view(size_t id) {
        auto found = decoded.find(id);
        if (found != decoded.end())
            return found->second;
        const auto &v = scene.at("bufferViews").at(id);
        const auto ext = v.value("extensions", Json::object());
        const auto compressed = ext.find("EXT_meshopt_compression");
        const auto &range = compressed == ext.end() ? v : *compressed;
        const auto buffer = range.at("buffer").get<size_t>();
        const auto path = assetUriPath(root,scene.at("buffers").at(buffer).at("uri").get<std::string>());
        const auto offset = range.value("byteOffset", uint64_t(0));
        const auto bytes = range.at("byteLength").get<size_t>();
        const auto fileBytes = fs::file_size(path);
        if (offset > fileBytes || bytes > fileBytes - offset)
            throw std::runtime_error("glTF buffer range outside source file");
        std::vector<uint8_t> input(bytes);
        std::ifstream stream(path, std::ios::binary);
        stream.seekg(offset);
        stream.read(reinterpret_cast<char *>(input.data()), bytes);
        if (!stream)
            throw std::runtime_error("Source buffer read failed");
        if (compressed == ext.end())
            return decoded.emplace(id, std::move(input)).first->second;
        const auto count = range.at("count").get<size_t>(), stride = range.at("byteStride").get<size_t>();
        if (!stride || stride > 256 || count > SIZE_MAX / stride)
            throw std::runtime_error("Invalid meshopt decode size");
        std::vector<uint8_t> output(count * stride);
        const auto mode = range.at("mode").get<std::string>();
        int status = -1;
        if (mode == "ATTRIBUTES")
            status = meshopt_decodeVertexBuffer(output.data(), count, stride, input.data(), input.size());
        else if (mode == "TRIANGLES")
            status = meshopt_decodeIndexBuffer(output.data(), count, stride, input.data(), input.size());
        else if (mode == "INDICES")
            status = meshopt_decodeIndexSequence(output.data(), count, stride, input.data(), input.size());
        if (status)
            throw std::runtime_error("meshopt source decode failed: " + mode);
        const auto filter = range.value("filter", "NONE");
        if (filter == "OCTAHEDRAL")
            meshopt_decodeFilterOct(output.data(), count, stride);
        else if (filter == "QUATERNION")
            meshopt_decodeFilterQuat(output.data(), count, stride);
        else if (filter == "EXPONENTIAL")
            meshopt_decodeFilterExp(output.data(), count, stride);
        else if (filter == "COLOR")
            meshopt_decodeFilterColor(output.data(), count, stride);
        else if (filter != "NONE")
            throw std::runtime_error("Unsupported meshopt filter");
        return decoded.emplace(id, std::move(output)).first->second;
    }
    float component(const uint8_t *p, int type, bool normalized) {
        if (type == 5126) {
            float v;
            std::memcpy(&v, p, 4);
            return v;
        }
        if (type == 5120) {
            int8_t v;
            std::memcpy(&v, p, 1);
            return normalized ? std::max(-1.f, v / 127.f) : float(v);
        }
        if (type == 5121)
            return normalized ? *p / 255.f : float(*p);
        if (type == 5122) {
            int16_t v;
            std::memcpy(&v, p, 2);
            return normalized ? std::max(-1.f, v / 32767.f) : float(v);
        }
        if (type == 5123) {
            uint16_t v;
            std::memcpy(&v, p, 2);
            return normalized ? v / 65535.f : float(v);
        }
        if (type == 5125) {
            uint32_t v;
            std::memcpy(&v, p, 4);
            return float(v);
        }
        throw std::runtime_error("Unsupported glTF component type");
    }
    std::vector<std::array<float,4>> attributes(size_t id,size_t components) {
        const auto& a=scene.at("accessors").at(id);
        if(components<2||components>4||a.at("type")!=("VEC"+std::to_string(components)))
            throw std::runtime_error("Unexpected glTF attribute type");
        const auto& viewMeta=scene.at("bufferViews").at(a.at("bufferView").get<size_t>());
        const auto& data=view(a.at("bufferView").get<size_t>());
        const int type=a.at("componentType");
        const size_t width=(type==5120||type==5121)?1:(type==5122||type==5123)?2:4;
        const size_t stride=viewMeta.value("byteStride",width*components),offset=a.value("byteOffset",size_t(0)),count=a.at("count");
        if(!count||stride<width*components||offset>data.size()||(count-1)>(data.size()-offset)/stride||(count-1)*stride+width*components>data.size()-offset)
            throw std::runtime_error("glTF attribute outside decoded buffer");
        std::vector<std::array<float,4>> result(count);
        for(size_t i=0;i<count;++i)for(size_t j=0;j<components;++j) {
            result[i][j]=component(data.data()+offset+i*stride+j*width,type,a.value("normalized",false));
            if(!std::isfinite(result[i][j]))throw std::runtime_error("Non-finite glTF attribute");
        }
        return result;
    }
    std::vector<glm::vec3> vectors(size_t id) {
        const auto &a = scene.at("accessors").at(id);
        if (a.at("type") != "VEC3")
            throw std::runtime_error("Expected glTF VEC3 geometry accessor");
        const auto &v = scene.at("bufferViews").at(a.at("bufferView").get<size_t>());
        const auto &data = view(a.at("bufferView").get<size_t>());
        const int type = a.at("componentType");
        const size_t width = (type == 5120 || type == 5121) ? 1 : (type == 5122 || type == 5123) ? 2 : 4;
        const size_t stride = v.value("byteStride", width * 3), offset = a.value("byteOffset", size_t(0)),
                     count = a.at("count");
        if (!count || offset > data.size() || (count - 1) > (data.size() - offset) / stride ||
            (count - 1) * stride + width * 3 > data.size() - offset)
            throw std::runtime_error("glTF geometry accessor outside decoded buffer");
        std::vector<glm::vec3> result(count);
        for (size_t i = 0; i < count; ++i)
            for (size_t j = 0; j < 3; ++j)
                result[i][j] = component(data.data() + offset + i * stride + j * width, type,
                                         a.value("normalized", false));
        return result;
    }
    std::vector<uint32_t> indices(size_t id) {
        const auto &a = scene.at("accessors").at(id);
        const auto &data = view(a.at("bufferView").get<size_t>());
        const int type = a.at("componentType");
        const size_t width = type == 5121 ? 1 : type == 5123 ? 2 : type == 5125 ? 4 : 0;
        const size_t count = a.at("count"), offset = a.value("byteOffset", size_t(0));
        if (!width || offset > data.size() || count > (data.size() - offset) / width || count % 3)
            throw std::runtime_error("Invalid triangle index accessor");
        std::vector<uint32_t> result(count);
        for (size_t i = 0; i < count; ++i)
            std::memcpy(&result[i], data.data() + offset + i * width, width);
        return result;
    }
    std::vector<glm::quat> rotations(size_t id) {
        const auto &a = scene.at("accessors").at(id);
        if (a.at("type") != "VEC4" || a.at("componentType") != 5126)
            throw std::runtime_error("Expected float quaternion instance accessor");
        const auto &v = scene.at("bufferViews").at(a.at("bufferView").get<size_t>());
        const auto &data = view(a.at("bufferView"));
        const size_t stride = v.value("byteStride", size_t(16)), offset = a.value("byteOffset", size_t(0)),
                     count = a.at("count");
        if (stride < 16 || offset > data.size() || !count || (count - 1) > (data.size() - offset) / stride ||
            (count - 1) * stride + 16 > data.size() - offset)
            throw std::runtime_error("Instance quaternion accessor outside buffer");
        std::vector<glm::quat> result(count);
        for (size_t i = 0; i < count; ++i) {
            float q[4];
            std::memcpy(q, data.data() + offset + i * stride, 16);
            result[i] = glm::quat(q[3], q[0], q[1], q[2]);
        }
        return result;
    }
    std::string contentKey(size_t id,bool shading=false) const {
        uint64_t hash=14695981039346656037ull;
        auto append=[&](const char* p,size_t n){for(size_t i=0;i<n;++i){hash^=static_cast<unsigned char>(p[i]);hash*=1099511628211ull;}};
        const auto layout=geometryKey(id,shading);append(layout.data(),layout.size());
        std::vector<char> bytes(1u<<20);
        for(const auto& p:scene.at("meshes").at(id).at("primitives")) {
            std::vector<size_t> ids{p.at("indices").get<size_t>(),p.at("attributes").at("POSITION").get<size_t>()};
            for(const char* name:{"NORMAL","TEXCOORD_0","TANGENT"})
                if((shading||std::string(name)=="NORMAL")&&p.at("attributes").contains(name))ids.push_back(p.at("attributes").at(name).get<size_t>());
            for(auto accessor:ids) {
                const auto& view=scene.at("bufferViews").at(scene.at("accessors").at(accessor).at("bufferView").get<size_t>());
                const auto ext=view.value("extensions",Json::object());
                const auto& range=ext.contains("EXT_meshopt_compression")?ext.at("EXT_meshopt_compression"):view;
                const auto file=assetUriPath(root,scene.at("buffers").at(range.at("buffer").get<size_t>()).at("uri").get<std::string>());
                const uint64_t offset=range.value("byteOffset",uint64_t(0)),count=range.at("byteLength");
                const auto size=fs::file_size(file);
                if(offset>size||count>size-offset)throw std::runtime_error("glTF signature range outside source buffer");
                std::ifstream input(file,std::ios::binary);input.seekg(offset);
                for(uint64_t left=count;left;) {
                    const auto n=size_t(std::min<uint64_t>(left,bytes.size()));input.read(bytes.data(),n);
                    if(!input)throw std::runtime_error("Cannot read glTF source signature: "+file.u8string());
                    append(bytes.data(),n);left-=n;
                }
            }
        }
        return std::to_string(hash);
    }
    std::string geometryKey(size_t id,bool shading=false) const {
        auto accessor = [&](size_t id) {
            const auto &a = scene.at("accessors").at(id);
            const auto &v = scene.at("bufferViews").at(a.at("bufferView").get<size_t>());
            const auto ext = v.value("extensions", Json::object());
            const auto found = ext.find("EXT_meshopt_compression");
            const auto &range = found == ext.end() ? v : *found;
            return Json{{"uri", scene.at("buffers").at(range.at("buffer").get<size_t>()).at("uri")},
                        {"offset", range.value("byteOffset", uint64_t(0))},
                        {"bytes", range.at("byteLength")},
                        {"stride", v.value("byteStride", size_t(0))},
                        {"codec", found == ext.end() ? Json() : *found},
                        {"accessorOffset", a.value("byteOffset", size_t(0))},
                        {"component", a.at("componentType")},
                        {"type", a.at("type")},
                        {"count", a.at("count")},
                        {"normalized", a.value("normalized", false)}};
        };
        Json key = Json::array();
        for (const auto &p : scene.at("meshes").at(id).at("primitives")) {
            const auto &a = p.at("attributes");
            auto position = accessor(a.at("POSITION"));
            auto index = accessor(p.at("indices"));
            auto normal = a.contains("NORMAL") ? accessor(a.at("NORMAL")) : Json();
            auto uv = shading && a.contains("TEXCOORD_0") ? accessor(a.at("TEXCOORD_0")) : Json();
            auto tangent = shading && a.contains("TANGENT") ? accessor(a.at("TANGENT")) : Json();
            // Buffer indices vary between aliases; the URI/range identifies bytes.
            for (auto *value : {&position, &index, &normal, &uv, &tangent})
                if (!value->is_null() && !(*value)["codec"].is_null())
                    (*value)["codec"].erase("buffer");
            key.push_back({{"mode", p.value("mode", 4)},
                           {"position", position},
                           {"normal", normal},
                           {"indices", index}});
            if(shading) {key.back()["uv"]=uv;key.back()["tangent"]=tangent;}
        }
        return key.dump();
    }
};

} // namespace zeno::cluster
