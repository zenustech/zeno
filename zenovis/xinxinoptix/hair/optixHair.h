//
// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
#pragma once

#include <cuda/GeometryData.h>
#include <cuda/BufferView.h>
#include <cuda_runtime.h>
#include <sutil/Aabb.h>

#include <optix.h>
#include <optix_stubs.h>

#include <set>
#include <map>
#include <memory>

#include <glm/common.hpp>
#include <glm/mat4x4.hpp>

#include <XAS.h>
#include <raiicuda.h>

#include "Util.h"
#include "Hair.h"

struct CurveGroup {
    std::string mtlid;
    zeno::CurveType curveType;

    std::vector<uint> strands;
    std::vector<float3> points;
    std::vector<float>  widths;
    std::vector<float3> normals;
};

struct HairState
{
    std::shared_ptr<Hair> pHair;
    sutil::Aabb aabb;

    zeno::CurveType curveType;
    std::string mtid;

    OptixTraversableHandle gasHandle = 0;
    xinxinoptix::raii<CUdeviceptr> gasBuffer {};
    // Aux data
    GeometryData::Curves curves {};

public:

void makeHairGAS(OptixDeviceContext contex);
void makeAuxData();

void makeCurveGroupGas(OptixDeviceContext context, 
                        const std::vector<float3>& points, 
                        const std::vector<float>& widths,
                        const std::vector<float3>& normals, 
                        const std::vector<uint>& strands);
};

inline std::map< std::string, std::shared_ptr<Hair> > hair_cache;
inline std::map< std::tuple<std::string, uint>, std::shared_ptr<HairState> > geo_hair_map;

using hair_state_key = std::tuple<std::string, uint, std::string>;

inline std::set<hair_state_key> hair_xxx_cache;
inline std::set<hair_state_key> hair_yyy_cache;

inline void loadHair(const std::string& filePath, const std::string& mtlid, uint mode, glm::mat4 transform=glm::mat4(1.0f)) {

    auto hair = [&]() -> std::shared_ptr<Hair> {
        if (hair_cache.count(filePath) == 0) 
        {
            auto tmp = std::make_shared<Hair>( filePath );
            tmp->prepareWidths();
            hair_cache[filePath] = tmp;
        }
        return hair_cache[filePath];
    } ();

    auto state = std::make_shared<HairState>();
    state->curveType = (zeno::CurveType)mode;
    state->pHair = hair;

    geo_hair_map[ std::tuple{filePath, mode} ] = state;

    auto key = std::tuple{ filePath, mode, mtlid};
    hair_xxx_cache.insert(key );
}

inline void prepareHairs(OptixDeviceContext context) {

    std::vector< std::tuple<std::string, uint> > garbage;
    for (auto& [key, _] : geo_hair_map) {
        auto& [filePath, mode] = key;
        if (hair_cache.count(filePath)) {
            continue;
        }
        garbage.push_back(key);
    }

    for (auto& key : garbage) {
        geo_hair_map.erase(key);
    }

    hair_yyy_cache = hair_xxx_cache;
    hair_xxx_cache.clear();
    hair_cache.clear();

    for (auto& [key, state] : geo_hair_map) {

        state->makeHairGAS(context);
        state->makeAuxData();
    }
}

inline std::vector<CurveGroup> curveGroupCache;
inline std::vector<std::shared_ptr<HairState>> curveGroupStateCache;

inline void loadCurveGroup(const std::vector<float3>& points, const std::vector<float>& widths, const std::vector<float3>& normals, const std::vector<uint>& strands, 
                           zeno::CurveType curveType, std::string mtlid) {

    CurveGroup cg;
    cg.mtlid = mtlid; 
    cg.curveType = curveType;

    cg.points = points;
    cg.widths = widths;
    cg.normals = normals;
    cg.strands = strands;

    curveGroupCache.push_back(cg);
}

inline void prepareCurveGroup(OptixDeviceContext context) {

    curveGroupStateCache.clear();

    for (auto& ele : curveGroupCache) {
        auto state = std::make_shared<HairState>();
        state->curveType = ele.curveType;
        state->mtid = ele.mtlid;
        state->makeCurveGroupGas(context, ele.points, ele.widths, ele.normals, ele.strands);

        curveGroupStateCache.push_back(state);
    }

    curveGroupCache.clear();
}


inline void cleanupHairs() {
    hair_cache.clear();
    geo_hair_map.clear();

    hair_xxx_cache.clear();
    hair_yyy_cache.clear();  

    curveGroupCache.clear();
    curveGroupStateCache.clear();  
}

namespace xinxinoptix {

void updateCurves();

}