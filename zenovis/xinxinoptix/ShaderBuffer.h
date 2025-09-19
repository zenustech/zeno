#pragma once

#include <map>
#include <unordered_map>
#include <vector>
#include <string>
#include <memory>
#include <optix.h>
#include <sutil/sutil.h>
#include <sutil/Exception.h>

#include <raiicuda.h>

using ShaderBufferHostRef = std::shared_ptr< xinxinoptix::raii<CUdeviceptr> >;

struct ShaderBufferGroup {

    std::map<std::string, ShaderBufferHostRef> buffers;
    xinxinoptix::raii<CUdeviceptr> indexing;

    void reset() {
        indexing.reset();
        buffers.clear();
    }

    CUdeviceptr upload() {

        auto count = buffers.size();
        auto byte_size = sizeof(CUdeviceptr) * count;
        if (byte_size == 0) return 0;

        indexing.resize(byte_size);
        std::vector<CUdeviceptr> tmp(count);

        size_t i =0;
        for (auto& [k, v] : buffers) {
            tmp[i] = buffers[k]->handle; ++i;
        }

        cudaMemcpy((void*)indexing.handle, tmp.data(), byte_size, cudaMemcpyHostToDevice);
        return indexing.handle; 
    }

    auto code(std::string& source) {
        std::map<std::string, std::string> macro {};

        size_t i =0;
        for (const auto& [name, ref] : buffers) {

            auto found = source.find(name) != std::string::npos;
            if (!found) {

                continue;
            }

            macro[name+"_buffer"] = "buffers["+  std::to_string(i) +"]"; 
            macro[name+"_bfsize"] = std::to_string(ref->size);

        }
        return macro;
    }
};

inline ShaderBufferGroup globalShaderBufferGroup;

inline std::unordered_map<std::string, ShaderBufferGroup> LocalShaderBufferMap;

static void load_buffer_group(const std::string buffer_name, ShaderBufferHostRef buffer_ref, std::string key="") {

    if (key=="") {
        globalShaderBufferGroup.buffers[buffer_name] = buffer_ref;
        return;
    }

    LocalShaderBufferMap[key].buffers[buffer_name] = buffer_ref;
}