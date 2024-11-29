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

#include <cmath>
#include <cstring>
#include <iomanip>
#include <iterator>

#include <sutil/sutil.h>
#include <sutil/Exception.h>

#include "Util.h"
#include "optixHair.h"

#include <tbb/task.h>
#include <tbb/task_group.h>

void HairState::makeCurveGroupGAS(OptixDeviceContext context, 
                                const std::vector<float3>& points, 
                                const std::vector<float>& widths,
                                const std::vector<float3>& normals, 
                                const std::vector<uint>& strands) 
{

    xinxinoptix::raii<CUdeviceptr> devicePoints;
    {
        size_t byte_count = sizeof(float3) * points.size();
        devicePoints.resize(byte_count);
        cudaMemcpy((void*)devicePoints.handle, points.data(), byte_count, cudaMemcpyHostToDevice);
    }
    xinxinoptix::raii<CUdeviceptr> deviceWidths;
    {
        std::vector<float> dummy = widths;

        for( auto strand = strands.begin(); strand != strands.end() - 1; ++strand ) {
            
            if (zeno::CurveType::LINEAR == curveType) { break; }
            
            const int end = *( strand + 1 ) - 1;

            auto degree = CurveDegree(curveType);
            if (zeno::CurveType::CATROM == curveType) {
                degree -= 1;
            }
            
            for (int i=0; i<degree; ++i) {
                dummy[end-i] = 0;
            }
        }

        size_t byte_count = sizeof(float) * dummy.size();
        deviceWidths.resize(byte_count);
        cudaMemcpy((void*)deviceWidths.handle, dummy.data(), byte_count, cudaMemcpyHostToDevice);
    }

    auto prepare = [](const std::vector<uint>& m_strands, zeno::CurveType curveType) -> std::vector<uint> {
        std::vector<uint> segments;
        // loop to one before end, as last strand value is the "past last valid vertex"
        // index
        auto degree = CurveDegree(curveType);
        for( auto strand = m_strands.begin(); strand != m_strands.end() - 1; ++strand )
        {
            const int start = *( strand );                      // first vertex in first segment
            const int end   = *( strand + 1 ) - degree;         // second vertex of last segment
            for( int i = start; i < end; ++i )
            {
                segments.push_back( i );
            }
        }
        return segments;
    };

    // if (normals.size() > 0) {
    //     this->curveType = zeno::CurveType::Ribbon;
    // }

    xinxinoptix::raii<CUdeviceptr> deviceStrands;
    auto segments = prepare(strands, curveType);

    size_t byte_count = sizeof(uint) * segments.size();
    deviceStrands.resize(byte_count);
    cudaMemcpy((void*)deviceStrands.handle, segments.data(), byte_count, cudaMemcpyHostToDevice);

    xinxinoptix::raii<CUdeviceptr> deviceNormals {};
    if (!normals.empty()) {
        size_t byte_size = sizeof(float3) * normals.size();
        deviceNormals.resize(byte_size);
        cudaMemcpy((void*)deviceNormals.handle, normals.data(), byte_size, cudaMemcpyHostToDevice);
    } 

    // Curve build input.
    OptixBuildInput buildInput = {};
    buildInput.type = OPTIX_BUILD_INPUT_TYPE_CURVES;
    buildInput.curveArray.curveType = CURVE_TYPE_MAP.at(curveType);

    if (zeno::CurveType::LINEAR == curveType) {
        buildInput.curveArray.endcapFlags = OptixCurveEndcapFlags::OPTIX_CURVE_ENDCAP_DEFAULT; 
    } else {
        buildInput.curveArray.endcapFlags = OptixCurveEndcapFlags::OPTIX_CURVE_ENDCAP_ON; 
    }

    buildInput.curveArray.numPrimitives        = segments.size();
    buildInput.curveArray.vertexBuffers        = &devicePoints;
    buildInput.curveArray.numVertices          = points.size();
    buildInput.curveArray.vertexStrideInBytes  = sizeof( float3 );
    buildInput.curveArray.widthBuffers         = &deviceWidths;
    buildInput.curveArray.widthStrideInBytes   = sizeof( float );
    buildInput.curveArray.normalBuffers        = &deviceNormals.handle;
    buildInput.curveArray.normalStrideInBytes  = sizeof(float3);
    buildInput.curveArray.indexBuffer          = deviceStrands;
    buildInput.curveArray.indexStrideInBytes   = sizeof( uint );
    buildInput.curveArray.flag                 = OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL;
    buildInput.curveArray.primitiveIndexOffset = 0;

    OptixAccelBuildOptions accelBuildOptions = {};
    accelBuildOptions.buildFlags             = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS | OPTIX_BUILD_FLAG_ALLOW_RANDOM_INSTANCE_ACCESS;
    accelBuildOptions.operation              = OPTIX_BUILD_OPERATION_BUILD;

    xinxinoptix::buildXAS(context, accelBuildOptions, buildInput, gasBuffer, gasHandle);
    return;
}

void HairState::makeCurveGroupGAS(OptixDeviceContext context) {
    if (nullptr == curveGroup && nullptr == pHair) { return; }

    tbb::task_group tgroup;

    tgroup.run([&]() {
        if (nullptr != pHair) {
            makeCurveGroupGAS(context, pHair->points(), pHair->widths(), {}, pHair->strands());
        } else { 
            makeCurveGroupGAS(context, curveGroup->points, curveGroup->widths, curveGroup->normals, curveGroup->strands);
        } 
    });
    
    tgroup.run([&]() {
        if (nullptr != pHair) {
            makeAuxData(pHair->strands());
        } else {
            makeAuxData(curveGroup->strands);
        }
    });

    tgroup.wait();
}

void HairState::makeHairGAS(OptixDeviceContext context)
{
    auto pState = this;
    const Hair* pHair = pState->pHair.get();
    
    if (pState->gasHandle) return;

    pState->gasHandle = 0;
    pState->gasBuffer.reset();

    makeCurveGroupGAS(context);

    return;
}

std::vector<float2> HairState::strandU(zeno::CurveType curveType, const std::vector<uint>& m_strands)
{
    std::vector<float2> strand_u;
    auto degree = CurveDegree(curveType);

    for( auto strand = m_strands.begin(); strand != m_strands.end() - 1; ++strand )
    {
        const int   start    = *( strand );
        const int   end      = *( strand + 1 ) - degree;
        const int   segments = end - start;  // number of strand's segments
        const float scale    = 1.0f / segments;
        for( int i = 0; i < segments; ++i )
        {
            strand_u.push_back( make_float2( i * scale, scale ) );
        }
    }

    return strand_u;
}

std::vector<uint2> HairState::strandInfo(zeno::CurveType curveType, const std::vector<uint>& m_strands)
{
    std::vector<uint2> strandInfo;
    unsigned int       firstPrimitiveIndex = 0;

    auto degree = CurveDegree(curveType);

    for( auto strand = m_strands.begin(); strand != m_strands.end() - 1; ++strand )
    {
        uint2 info;
        info.x = firstPrimitiveIndex;                        // strand's start index
        info.y = *( strand + 1 ) - *(strand) - degree;       // number of segments in strand
        firstPrimitiveIndex += info.y;                       // increment with number of primitives/segments in strand
        strandInfo.push_back( info );
    }
    return strandInfo;
}

std::vector<uint> HairState::strandIndices(zeno::CurveType curveType, const std::vector<uint>& m_strands)
{
    std::vector<uint> strandIndices;
    int              strandIndex = 0;

    auto degree = CurveDegree(curveType);

    for( auto strand = m_strands.begin(); strand != m_strands.end() - 1; ++strand )
    {
        const int start = *( strand );
        const int end   = *( strand + 1 ) - degree;
        for( auto segment = start; segment != end; ++segment )
        {
            strandIndices.push_back( strandIndex );
        }
        ++strandIndex;
    }

    return strandIndices;
}

void HairState::makeAuxData(const std::vector<uint>& strands)
{
    auto pState = this;
    // clear curves_ data
    cudaFree( reinterpret_cast<void*>( pState->aux.strand_u.data ) );
    pState->aux.strand_u.data = 0;
    cudaFree( reinterpret_cast<void*>( pState->aux.strand_i.data ) );
    pState->aux.strand_i.data = 0;
    cudaFree( reinterpret_cast<void*>( pState->aux.strand_info.data ) );
    pState->aux.strand_info.data = 0;

    CUdeviceptr strandUs = 0;
    {
        auto strand_u = pState->strandU(pState->curveType, strands);
        createOnDevice( strand_u, &strandUs );
        pState->aux.strand_u.data        = strandUs;
        pState->aux.strand_u.count       = strand_u.size();
        pState->aux.strand_u.byte_stride = static_cast<uint16_t>( sizeof( float2 ) );
        //SUTIL_ASSERT( numberOfSegments(pState->curveType) == static_cast<int>( strandU(pState->curveType).size() );
        pState->aux.strand_u.elmt_byte_size = static_cast<uint16_t>( sizeof( float2 ) );
    }

    CUdeviceptr strandIs = 0;
    {
        auto strand_i = strandIndices(pState->curveType, strands);
        createOnDevice( strand_i, &strandIs );
        pState->aux.strand_i.data           = strandIs;
        pState->aux.strand_i.count          = strand_i.size();
        pState->aux.strand_i.byte_stride    = static_cast<uint16_t>( sizeof( uint ) );
        pState->aux.strand_i.elmt_byte_size = static_cast<uint16_t>( sizeof( uint ) );
    }

    CUdeviceptr strandInfos = 0; 
    {
        auto strand_info = pState->strandInfo(pState->curveType, strands);

        createOnDevice( pState->strandInfo(pState->curveType, strands), &strandInfos );
        pState->aux.strand_info.data           = strandInfos;
        pState->aux.strand_info.count          = strand_info.size();
        pState->aux.strand_info.byte_stride    = static_cast<uint16_t>( sizeof( uint2 ) );
        pState->aux.strand_info.elmt_byte_size = static_cast<uint16_t>( sizeof( uint2 ) );
    }
}