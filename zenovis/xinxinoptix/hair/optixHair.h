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

#include <cuda_runtime.h>
#include <sutil/Aabb.h>

#include <optix.h>
#include <optix_stubs.h>

#include <set>
#include <map>
#include <memory>
#include <filesystem>

#include <glm/common.hpp>
#include <glm/mat4x4.hpp>

#include <XAS.h>
#include <raiicuda.h>

#include "Util.h"
#include "Hair.h"
#include "GeometryAux.h"

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
    std::shared_ptr<CurveGroup> curveGroup;
    sutil::Aabb aabb;

    zeno::CurveType curveType;
    std::string mtid;

    OptixTraversableHandle gasHandle = 0;
    xinxinoptix::raii<CUdeviceptr> gasBuffer {};
    // Aux data
    CurveGroupAux aux {};

public:

void makeHairGAS(OptixDeviceContext contex);
void makeAuxData(const std::vector<uint>& strands);

void makeCurveGroupGAS(OptixDeviceContext context, 
                        const std::vector<float3>& points, 
                        const std::vector<float>& widths,
                        const std::vector<float3>& normals, 
                        const std::vector<uint>& strands);

void makeCurveGroupGAS(OptixDeviceContext context);

std::vector<float2> strandU(zeno::CurveType curveType, const std::vector<uint>& m_strands);
std::vector<uint2> strandInfo(zeno::CurveType curveType, const std::vector<uint>& m_strands);
std::vector<uint> strandIndices(zeno::CurveType curveType, const std::vector<uint>& m_strands);

};

inline std::map< std::string, std::shared_ptr<Hair> > hair_cache;
inline std::map< std::tuple<std::string, uint>, std::shared_ptr<HairState> > geo_hair_cache;

using hair_state_key = std::tuple<std::string, uint, std::string>;

inline std::map<hair_state_key, std::vector<glm::mat4>> hair_xxx_cache;
inline std::map<hair_state_key, std::vector<glm::mat4>> hair_yyy_cache;

inline void loadHair(const std::string& filePath, const std::string& mtlid, uint mode, glm::mat4 transform=glm::mat4(1.0f)) {

    auto lwt = std::filesystem::last_write_time(filePath);
    bool neo = false;

    auto hair = [&]() -> std::shared_ptr<Hair> {

        if (hair_cache.count(filePath) == 0 || lwt != hair_cache[filePath]->time()) 
        {
            neo = true;
            auto tmp = std::make_shared<Hair>( filePath );
            tmp->prepareWidths();
            hair_cache[filePath] = tmp;
            return tmp;
        }
        return hair_cache[filePath];
    } ();

    auto hairState = [&]() {
        auto key = std::tuple {filePath, mode};

        if (geo_hair_cache.count( key ) == 0 || neo) {

            auto tmp = std::make_shared<HairState>();
            tmp->curveType = (zeno::CurveType)mode;
            tmp->pHair = hair;

            geo_hair_cache[ key ] = tmp;
            return tmp;
        } 
        
        return geo_hair_cache[key];
    } ();

    auto key = std::tuple{ filePath, mode, mtlid};
    if (hair_xxx_cache.count(key)) {
        hair_xxx_cache[key].push_back(transform);
    } else {
        hair_xxx_cache[key] = { transform }; 
    }
}

inline void prepareHairs(OptixDeviceContext context) {

    decltype(hair_cache)     hair_cache_tmp;
    decltype(geo_hair_cache) geo_hair_cache_tmp;

    for (auto& [key, val] : hair_xxx_cache) {
        auto& [filePath, mode, mtlid] = key;

        if (hair_cache.count(filePath)) {
            hair_cache_tmp[filePath] = hair_cache[filePath];
        }

        if (geo_hair_cache.count( {filePath, mode} )) {
            geo_hair_cache_tmp[ {filePath, mode} ] = geo_hair_cache[ {filePath, mode} ];
        }
    }

    hair_cache     = std::move(hair_cache_tmp);
    geo_hair_cache = std::move(geo_hair_cache_tmp);

    hair_yyy_cache = hair_xxx_cache;
    hair_xxx_cache.clear();

    for (auto& [key, state] : geo_hair_cache) {
        state->makeHairGAS(context);
    }
}

inline std::vector<std::shared_ptr<CurveGroup>> curveGroupCache;
inline std::vector<std::shared_ptr<HairState>> curveGroupStateCache;

inline void loadCurveGroup(const std::vector<float3>& points, const std::vector<float>& widths, const std::vector<float3>& normals, const std::vector<uint>& strands, 
                           zeno::CurveType curveType, std::string mtlid) {

    auto cg = std::make_shared<CurveGroup>();
    cg->mtlid = mtlid; 
    cg->curveType = curveType;

    cg->points = std::move(points);
    cg->widths = std::move(widths);
    cg->normals = std::move(normals);
    cg->strands = std::move(strands);

    curveGroupCache.push_back(cg);
}

inline void prepareCurveGroup(OptixDeviceContext context) {

    curveGroupStateCache.clear();

    for (auto& ele : curveGroupCache) {
        auto state = std::make_shared<HairState>();

        state->curveGroup = ele;

        state->curveType = ele->curveType;
        state->mtid = ele->mtlid;

        state->makeCurveGroupGAS(context);

        curveGroupStateCache.push_back(state);
    }

    curveGroupCache.clear();
}


inline void cleanupHairs() {
    hair_cache.clear();
    geo_hair_cache.clear();

    hair_xxx_cache.clear();
    hair_yyy_cache.clear();  

    curveGroupCache.clear();
    curveGroupStateCache.clear();  
}

namespace xinxinoptix {

void updateCurves();

}