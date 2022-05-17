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

#include <sutil/sutilapi.h>
#include <sutil/vec_math.h>


namespace sutil {

// implementing a perspective camera
class Camera {
public:
    SUTILAPI Camera()
        : m_eye(make_float3(1.0f)), m_lookat(make_float3(0.0f)), m_up(make_float3(0.0f, 1.0f, 0.0f)), m_fovY(35.0f), m_aspectRatio(1.0f)
    {
    }

    SUTILAPI Camera(const float3& eye, const float3& lookat, const float3& up, float fovY, float aspectRatio)
        : m_eye(eye), m_lookat(lookat), m_up(up), m_fovY(fovY), m_aspectRatio(aspectRatio)
    {
    }

    SUTILAPI float3 direction() const { return normalize(m_lookat - m_eye); }
    SUTILAPI void setDirection(const float3& dir) { m_lookat = m_eye + length(m_lookat - m_eye) * dir; }

    SUTILAPI const float3& eye() const { return m_eye; }
    SUTILAPI void setEye(const float3& val) { m_eye = val; }
    SUTILAPI const float3& lookat() const { return m_lookat; }
    SUTILAPI void setLookat(const float3& val) { m_lookat = val; }
    SUTILAPI const float3& up() const { return m_up; }
    SUTILAPI void setUp(const float3& val) { m_up = val; }
    SUTILAPI const float& fovY() const { return m_fovY; }
    SUTILAPI void setFovY(const float& val) { m_fovY = val; }
    SUTILAPI const float& aspectRatio() const { return m_aspectRatio; }
    SUTILAPI void setAspectRatio(const float& val) { m_aspectRatio = val; }

    // UVW forms an orthogonal, but not orthonormal basis!
    SUTILAPI void UVWFrame(float3& U, float3& V, float3& W) const;

    void setZxxViewMatrix(float3 U, float3 V, float3 W) {
        m_isZxx = true;
        m_zxxU = U;
        m_zxxV = V;
        m_zxxW = W;
    }

private:
    float3 m_eye;
    float3 m_lookat;
    float3 m_up;
    float m_fovY;
    float m_aspectRatio;

    bool m_isZxx = false;
    float3 m_zxxU;
    float3 m_zxxV;
    float3 m_zxxW;
};

}
