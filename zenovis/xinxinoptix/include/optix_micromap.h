/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/**
* @file   optix_micromap.h
* @author NVIDIA Corporation
* @brief  OptiX micromap helper functions
*
* OptiX micromap helper functions. Useable on either host or device.
*/

#ifndef OPTIX_OPTIX_MICROMAP_H
#define OPTIX_OPTIX_MICROMAP_H

#if !defined( OPTIX_DONT_INCLUDE_CUDA )
// If OPTIX_DONT_INCLUDE_CUDA is defined, cuda driver type float2 must be defined through other
// means before including optix headers.
#include <vector_types.h>
#endif
#include "internal/optix_micromap_impl.h"

/// Converts a micromap triangle index to the three base-triangle barycentric coordinates of the micro-triangle vertices in the base triangle.
/// The base triangle is the triangle that the micromap is applied to.
/// Note that for displaced micro-meshes this function can be used to compute a UV mapping from sub triangle to base triangle.
///
/// \param[in]  micromapTriangleIndex  Index of a micro- or sub triangle within a micromap.
/// \param[in]  subdivisionLevel       Number of subdivision levels of the micromap or number of subdivision levels being considered (for sub triangles).
/// \param[out] baseBarycentrics0      Barycentric coordinates in the space of the base triangle of vertex 0 of the micromap triangle.
/// \param[out] baseBarycentrics1      Barycentric coordinates in the space of the base triangle of vertex 1 of the micromap triangle.
/// \param[out] baseBarycentrics2      Barycentric coordinates in the space of the base triangle of vertex 2 of the micromap triangle.
OPTIX_MICROMAP_INLINE_FUNC void optixMicromapIndexToBaseBarycentrics( unsigned int micromapTriangleIndex,
                                                                      unsigned int subdivisionLevel,
                                                                      float2&      baseBarycentrics0,
                                                                      float2&      baseBarycentrics1,
                                                                      float2&      baseBarycentrics2 )
{
    optix_impl::micro2bary( micromapTriangleIndex, subdivisionLevel, baseBarycentrics0, baseBarycentrics1, baseBarycentrics2 );
}

/// Maps barycentrics in the space of the base triangle to barycentrics of a micro triangle.
/// The vertices of the micro triangle are defined by its barycentrics in the space of the base triangle.
/// These can be queried for a DMM hit by using optixGetMicroTriangleBarycentricsData().
OPTIX_MICROMAP_INLINE_FUNC float2 optixBaseBarycentricsToMicroBarycentrics( float2 baseBarycentrics,
                                                                            float2 microVertexBaseBarycentrics[3] )
{
    return optix_impl::base2micro( baseBarycentrics, microVertexBaseBarycentrics );
}

#endif  // OPTIX_OPTIX_MICROMAP_H
