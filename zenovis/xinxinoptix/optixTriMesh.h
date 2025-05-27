#pragma once 

#include "optixCommon.h"

#include <map>
#include <vector>

struct MeshDat {
    bool dirty = true;

    std::vector<std::string> mtlidList;
    std::string mtlid;
    std::vector<float> verts;
    std::vector<uint> tris;
    std::vector<int> triMats;

    std::map<std::string, std::vector<float>> vertattrs;
    auto const &getAttr(std::string const &s) const
    {
        //if(vertattrs.find(s)!=vertattrs.end())
        //{
            return vertattrs.at(s);//->second;
        //}
    }
};

struct MeshObject {
    bool dirty = true;
    std::string matid;

    std::vector<float3> vertices{};
    std::vector<uint3>  indices {};
    std::vector<uint16_t> mat_idx{};

    std::vector<float2> g_uv;
    std::vector<ushort3> g_clr, g_nrm, g_tan;

template<typename T>
using raii = xinxinoptix::raii<T>;

    raii<CUdeviceptr> d_uv, d_clr, d_nrm, d_tan;
    raii<CUdeviceptr> d_idx;
    raii<CUdeviceptr> d_mat;

    std::shared_ptr<SceneNode> node = std::make_shared<SceneNode>();

    MeshObject() = default;

    void resize(size_t tri_num, size_t vert_num) {

        indices.resize(tri_num);
        //mat_idx.resize(tri_num);

        vertices.resize(vert_num);
        g_nrm.resize(vert_num);
        g_tan.resize(vert_num);
        g_clr.resize(vert_num);
        g_uv.resize(vert_num);
    }

    template<class T>
    static void upload(std::vector<T>& vector, raii<CUdeviceptr>& buffer) {

        if (!vector.empty()) {

            auto byte_size = sizeof(vector[0]) * vector.size();
            buffer.resize(byte_size);
            cudaMemcpy((void*)buffer.handle, vector.data(), byte_size, cudaMemcpyHostToDevice);

        } else { buffer.reset(); }
    }

    void upload(size_t extra_size) {

        upload(g_uv, d_uv);
        upload(g_clr, d_clr);
        upload(g_nrm, d_nrm);
        upload(g_tan, d_tan);
        upload(indices, d_idx);

        auto offset = roundUp<size_t>(extra_size, 128u);
        auto gas_ptr = node->buffer.handle + offset;

        auto buffers = this->aux();
        if (0) {
            upload(mat_idx, d_mat);
            buffers.push_back(d_mat.handle);
        } else {
            d_mat.reset();
            buffers.push_back(0);
        }
        std::reverse(buffers.begin(), buffers.end());

        auto byte_size = sizeof(buffers[0]) * buffers.size();
        cudaMemcpy((void*)(gas_ptr-byte_size), buffers.data(), byte_size, cudaMemcpyHostToDevice);
    }

    std::vector<CUdeviceptr> aux() {
        return std::vector { d_idx.handle, d_uv.handle, d_clr.handle, d_nrm.handle, d_tan.handle };
    }

    void buildGas(OptixDeviceContext context, uint16_t sbt_count) {

        auto buffers = this->aux();
        std::reverse(buffers.begin(), buffers.end());
        auto extra_size = sizeof(buffers[0]) * buffers.size();

        xinxinoptix::buildMeshGAS(context, vertices, indices, mat_idx, sbt_count, node->buffer, node->handle, extra_size);
        upload(extra_size);
    }

    ~MeshObject() {
        node = nullptr;
    }
};