#ifndef __PROBE_HPP__
#define __PROBE_HPP__

#include "MyShader.hpp"
#include "glm/ext/matrix_clip_space.hpp"
#include "glm/ext/matrix_transform.hpp"
#include "glm/geometric.hpp"
#include "stdafx.hpp"
#include "main.hpp"
#include "IGraphic.hpp"
#include <Hg/FPSCounter.hpp>
#include <sstream>
#include <cstdlib>
#include <cmath>
#include <array>
#include <stb_image_write.h>
#include <Hg/OpenGL/stdafx.hpp>
#include <unordered_map>
#include "zenvisapi.hpp"
#include <spdlog/spdlog.h>

#define PI 3.14159265

#ifdef _WIN32
    #ifdef near
        #undef near
    #endif
    #ifdef far
        #undef far
    #endif
#endif

namespace zenvis
{
    struct Probe
    {
        glm::vec3 pos = glm::vec3(0, 0, 0);
        int resolution = 128;

        unsigned int probeFBO = 0;
        unsigned int probeRBO = 0;
        unsigned int textureID = 0;

        Probe()
        {
            initTextureCapture();
        }

        Probe(int size): resolution(size)
        {
            initTextureCapture();
        }

        void initTextureCapture()
        {
            if (textureID == 0) 
            {
                // Create an empty cubemap
                CHECK_GL(glGenTextures(1, &textureID));
                CHECK_GL(glBindTexture(GL_TEXTURE_CUBE_MAP, textureID));

                for (int i = 0; i < 6; i++) {
                    glTexImage2D(
                        GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, 
                        GL_RGBA32F, resolution, resolution, 0, GL_RGBA, GL_FLOAT, nullptr
                        );
                }
                glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
                glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
                glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
                glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
                glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
            }
            if (probeFBO == 0) 
            {
                // generate or select buffer to render based on index？
                CHECK_GL(glGenFramebuffers(1, &probeFBO));
                glBindFramebuffer(GL_FRAMEBUFFER, probeFBO);
                glDrawBuffer(GL_COLOR_ATTACHMENT0);
            }
            if (probeRBO == 0) 
            {
                CHECK_GL(glGenRenderbuffers(1, &probeRBO));
                glBindRenderbuffer(GL_RENDERBUFFER, probeRBO);
                glRenderbufferStorage(
                    GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, 
                    resolution, resolution); // beware size
                glFramebufferRenderbuffer(
                    GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, 
                    GL_RENDERBUFFER, probeRBO);
            }
        }

        GLuint BeginEnvMap(glm::vec3 in_pos, glm::vec3 bgcolor)
        {
            pos = in_pos;
            
            glViewport(0, 0, resolution, resolution);
            glBindFramebuffer(GL_FRAMEBUFFER, probeFBO);
            // Before rendering, clear color buffer and depth buffer
            CHECK_GL(glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA));
            CHECK_GL(glClearColor(bgcolor.r, bgcolor.g, bgcolor.b, 0.0f));
            CHECK_GL(glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT));

            return textureID;
            // return probeFBO;
        }

        void EndEnvMap(int nx, int ny)
        {
            // unbind fbo and recover viewport size
            glBindFramebuffer(GL_FRAMEBUFFER, 0);
            glViewport(0, 0, nx, ny);
            // if only render once, delete rbo & fbo?
            // CHECK_GL(glDeleteRenderbuffers(1, &probeRBO));
            // CHECK_GL(glDeleteFramebuffers(1, &probeFBO));
            // for (int i = 0; i < 6; i++) {
            //     glTexImage2D(
            //         GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, 
            //         GL_RGBA32F, resolution, resolution, 0, GL_RGBA, GL_FLOAT, nullptr
            //         );
            // }
            // todo: cubeMap, to be used as a texture? but textureID (unsigned int)

        }
    };
};

#endif // #include __PROBE_HPP__