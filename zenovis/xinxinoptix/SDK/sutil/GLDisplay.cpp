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

#include <sutil/Exception.h>
#include <sutil/GLDisplay.h>

#include <iostream>

namespace sutil
{

//-----------------------------------------------------------------------------
//
// Helper functions
//
//-----------------------------------------------------------------------------
namespace
{

GLuint createGLShader( const std::string& source, GLuint shader_type )
{
	GLuint shader = glCreateShader( shader_type );
    {
        const GLchar* source_data= reinterpret_cast<const GLchar*>( source.data() );
		glShaderSource( shader, 1, &source_data, nullptr );
		glCompileShader( shader );

		GLint is_compiled = 0;
		glGetShaderiv( shader, GL_COMPILE_STATUS, &is_compiled );
		if( is_compiled == GL_FALSE )
		{
			GLint max_length = 0;
			glGetShaderiv( shader, GL_INFO_LOG_LENGTH, &max_length );

			std::string info_log( max_length, '\0' );
            GLchar* info_log_data= reinterpret_cast<GLchar*>( &info_log[0]);
			glGetShaderInfoLog( shader, max_length, nullptr, info_log_data );

			glDeleteShader(shader);
            std::cerr << "Compilation of shader failed: " << info_log << std::endl;

			return 0;
		}
	}

    GL_CHECK_ERRORS();

    return shader;
}


GLuint createGLProgram(
		const std::string& vert_source,
		const std::string& frag_source
		)
{
    GLuint vert_shader = createGLShader( vert_source, GL_VERTEX_SHADER );
    if( vert_shader == 0 )
        throw Exception("Compile vert failure");

    GLuint frag_shader = createGLShader( frag_source, GL_FRAGMENT_SHADER );
    if( frag_shader == 0 )
    {
        GL_CHECK(glDeleteShader( vert_shader ));
        throw Exception("Compile frag failure");
    }

    GLuint program = glCreateProgram();
    GL_CHECK(glAttachShader( program, vert_shader ));
    GL_CHECK(glAttachShader( program, frag_shader ));
    GL_CHECK(glLinkProgram( program ));

    GLint is_linked = 0;
    GL_CHECK(glGetProgramiv( program, GL_LINK_STATUS, &is_linked ));
    if (is_linked == GL_FALSE)
    {
        GLint max_length = 0;
        GL_CHECK(glGetProgramiv( program, GL_INFO_LOG_LENGTH, &max_length ));

        std::string info_log( max_length, '\0' );
        GLchar* info_log_data= reinterpret_cast<GLchar*>( &info_log[0]);
        GL_CHECK(glGetProgramInfoLog( program, max_length, nullptr, info_log_data ));
        std::cerr << "Linking of program failed: " << info_log << std::endl;

        GL_CHECK(glDeleteProgram( program ));
        GL_CHECK(glDeleteShader( vert_shader ));
        GL_CHECK(glDeleteShader( frag_shader ));

        throw Exception("Link failure");
    }

    GL_CHECK(glDetachShader( program, vert_shader ));
    GL_CHECK(glDetachShader( program, frag_shader ));

    GL_CHECK_ERRORS();

    return program;
}


GLint getGLUniformLocation( GLuint program, const std::string& name )
{
	GLint loc = glGetUniformLocation( program, name.c_str() );
    SUTIL_ASSERT_MSG( loc != -1, "Failed to get uniform loc for '" + name + "'" );
    return loc;
}

} // anonymous namespace


//-----------------------------------------------------------------------------
//
// GLDisplay implementation
//
//-----------------------------------------------------------------------------

const std::string GLDisplay::s_vert_source = R"(
#version 330 core

layout(location = 0) in vec3 vertexPosition_modelspace;
out vec2 UV;

void main()
{
	gl_Position =  vec4(vertexPosition_modelspace,1);
	UV = (vec2( vertexPosition_modelspace.x, vertexPosition_modelspace.y )+vec2(1,1))/2.0;
}
)";

const std::string GLDisplay::s_frag_source = R"(
#version 330 core

in vec2 UV;
out vec3 color;

uniform sampler2D render_tex;
uniform bool correct_gamma;

void main()
{
    color = texture( render_tex, UV ).xyz;
}
)";



GLDisplay::GLDisplay( BufferImageFormat image_format )
    : m_image_format( image_format )
{
    GLuint m_vertex_array;
    GL_CHECK( glGenVertexArrays(1, &m_vertex_array ) );
    GL_CHECK( glBindVertexArray( m_vertex_array ) );

	m_program = createGLProgram( s_vert_source, s_frag_source );
	m_render_tex_uniform_loc = getGLUniformLocation( m_program, "render_tex");
    if (!m_program) throw Exception("program is null");

    GL_CHECK( glGenTextures( 1, &m_render_tex ) );
    GL_CHECK( glBindTexture( GL_TEXTURE_2D, m_render_tex ) );

    GL_CHECK( glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST ) );
    GL_CHECK( glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST ) );
    GL_CHECK( glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE ) );
    GL_CHECK( glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE ) );

	static const GLfloat g_quad_vertex_buffer_data[] = {
		-1.0f, -1.0f, 0.0f,
		 1.0f, -1.0f, 0.0f,
		-1.0f,  1.0f, 0.0f,

		-1.0f,  1.0f, 0.0f,
		 1.0f, -1.0f, 0.0f,
		 1.0f,  1.0f, 0.0f,
	};

	GL_CHECK( glGenBuffers( 1, &m_quad_vertex_buffer ) );
	GL_CHECK( glBindBuffer( GL_ARRAY_BUFFER, m_quad_vertex_buffer ) );
	GL_CHECK( glBufferData( GL_ARRAY_BUFFER,
            sizeof( g_quad_vertex_buffer_data),
            g_quad_vertex_buffer_data,
            GL_STATIC_DRAW
            )
        );

    GL_CHECK_ERRORS();
}


void GLDisplay::display(
        const int32_t  screen_res_x,
        const int32_t  screen_res_y,
        const int32_t  framebuf_res_x,
        const int32_t  framebuf_res_y,
        const uint32_t pbo,
        const uint32_t fbo
        ) const
{
    GLint oldfbo = 0;
    GL_CHECK( glGetIntegerv( GL_FRAMEBUFFER_BINDING, &oldfbo ) );
    GL_CHECK( glBindFramebuffer( GL_FRAMEBUFFER, fbo ) );
    //GL_CHECK( glViewport( 0, 0, framebuf_res_x, framebuf_res_y ) );

    GL_CHECK( glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT ) );

    if (!m_program) throw Exception("program is null");
    GL_CHECK( glUseProgram( m_program ) );

    // Bind our texture in Texture Unit 0
    GL_CHECK( glActiveTexture( GL_TEXTURE0 ) );
    GL_CHECK( glBindTexture( GL_TEXTURE_2D, m_render_tex ) );
    GL_CHECK( glBindBuffer( GL_PIXEL_UNPACK_BUFFER, pbo ) );

    GL_CHECK( glPixelStorei(GL_UNPACK_ALIGNMENT, 4) ); // TODO!!!!!!

    //size_t elmt_size = pixelFormatSize( m_image_format );
    //if      ( elmt_size % 8 == 0) glPixelStorei(GL_UNPACK_ALIGNMENT, 8);
    //else if ( elmt_size % 4 == 0) glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
    //else if ( elmt_size % 2 == 0) glPixelStorei(GL_UNPACK_ALIGNMENT, 2);
    //else                          glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    bool convertToSrgb = true;

    if( m_image_format == BufferImageFormat::UNSIGNED_BYTE4 )
    {
        // input is assumed to be in srgb since it is only 1 byte per channel in size
        glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA8,   screen_res_x, screen_res_y, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr );
        convertToSrgb = false;
    }
    else if( m_image_format == BufferImageFormat::FLOAT3 )
        glTexImage2D( GL_TEXTURE_2D, 0, GL_RGB32F,  screen_res_x, screen_res_y, 0, GL_RGB,  GL_FLOAT,         nullptr );

    else if( m_image_format == BufferImageFormat::FLOAT4 )
        glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA32F, screen_res_x, screen_res_y, 0, GL_RGBA, GL_FLOAT,         nullptr );

    else
        throw Exception( "Unknown buffer format" );

    GL_CHECK( glBindBuffer( GL_PIXEL_UNPACK_BUFFER, 0 ) );
    if (!m_program) throw Exception("program is null");
    GL_CHECK( glUniform1i( m_render_tex_uniform_loc , 0 ) );

    // 1st attribute buffer : vertices
    GL_CHECK( glEnableVertexAttribArray( 0 ) );
    GL_CHECK( glBindBuffer(GL_ARRAY_BUFFER, m_quad_vertex_buffer ) );
    GL_CHECK( glVertexAttribPointer(
            0,                  // attribute 0. No particular reason for 0, but must match the layout in the shader.
            3,                  // size
            GL_FLOAT,           // type
            GL_FALSE,           // normalized?
            0,                  // stride
            (void*)0            // array buffer offset
            )
        );

    //if( convertToSrgb )
        //GL_CHECK( glEnable( GL_FRAMEBUFFER_SRGB ) );
    //else 
        //GL_CHECK( glDisable( GL_FRAMEBUFFER_SRGB ) );

    // Draw the triangles !
    GL_CHECK( glDrawArrays(GL_TRIANGLES, 0, 6) ); // 2*3 indices starting at 0 -> 2 triangles

    GL_CHECK( glDisableVertexAttribArray(0) );

    //GL_CHECK( glDisable( GL_FRAMEBUFFER_SRGB ) );

    GL_CHECK( glBindFramebuffer( GL_FRAMEBUFFER, oldfbo ) );
    GL_CHECK_ERRORS();
}

} // namespace sutil
