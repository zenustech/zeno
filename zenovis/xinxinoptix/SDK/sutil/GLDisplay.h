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

#include <glad/glad.h>

#include <cstdint>
#include <string>

#include <sutil/sutil.h>
#include <sutil/sutilapi.h>

namespace sutil
{

class GLDisplay
{
public:
    SUTILAPI GLDisplay(
        BufferImageFormat format = sutil::BufferImageFormat::UNSIGNED_BYTE4);

    SUTILAPI void display(
            const int32_t  screen_res_x,
            const int32_t  screen_res_y,
            const int32_t  framebuf_res_x,
            const int32_t  framebuf_res_y,
            const uint32_t pbo,
            const uint32_t fbo = 0) const;

private:
    GLuint   m_render_tex = 0u;
    GLuint   m_program = 0u;
    GLint    m_render_tex_uniform_loc = -1;
    GLuint   m_quad_vertex_buffer = 0;

    sutil::BufferImageFormat m_image_format;

    static const std::string s_vert_source;
    static const std::string s_frag_source;
};

} // end namespace sutil
