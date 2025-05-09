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

#include "../optixCommon.h"

struct CurveGroup {
    bool dirty = true;
    zeno::CurveType curveType;

    std::vector<uint> strands;
    std::vector<float3> points;
    std::vector<float>  widths;
    std::vector<float3> normals;
};

struct CurveGroupWrapper
{
    bool dirty = true;
    std::shared_ptr<Hair> pHair;
    std::shared_ptr<CurveGroup> curveGroup;
    sutil::Aabb aabb;

    zeno::CurveType curveType;
    std::shared_ptr<SceneNode> node {};

    // Aux data
    CurveGroupAux aux {};

    ~CurveGroupWrapper() {
        cudaFreeAsync( reinterpret_cast<void*>( aux.strand_u.data ), 0 );
        cudaFreeAsync( reinterpret_cast<void*>( aux.strand_i.data ), 0 );
        cudaFreeAsync( reinterpret_cast<void*>( aux.strand_info.data ), 0 );
    }

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

