#pragma once

#include <zeno/utils/disable_copy.h>
#include <zenovis/Camera.h>
#include <zenovis/IGraphic.h>
#include <zenovis/Light.h>
#include <zenovis/ReflectivePass.h>
#include <zenovis/Scene.h>
#include <zenovis/ShaderManager.h>
#include <zenovis/opengl/shader.h>
#include <zenovis/opengl/vao.h>

namespace zenovis {

struct DepthPass : zeno::disable_copy {
    Scene *scene;

    explicit DepthPass(Scene *scene) : scene(scene) {
    }

    Camera *camera() const {
        return scene->camera.get();
    }

    bool enable_hdr = true;
    /* BEGIN ZHXX HAPPY */

    inline static const char *qvert = R"(
#version 330 core
const vec2 quad_vertices[4] = vec2[4]( vec2( -1.0, -1.0), vec2( 1.0, -1.0), vec2( -1.0, 1.0), vec2( 1.0, 1.0));
void main()
{
    gl_Position = vec4(quad_vertices[gl_VertexID], 0.0, 1.0);
}
)";
    inline static const char *qfrag = R"(#version 330 core
// #extension GL_EXT_gpu_shader4 : enable
// hdr_adaptive.fs

const mat3x3 ACESInputMat = mat3x3
(
    0.59719, 0.35458, 0.04823,
    0.07600, 0.90834, 0.01566,
    0.02840, 0.13383, 0.83777
);

// ODT_SAT => XYZ => D60_2_D65 => sRGB
const mat3x3 ACESOutputMat = mat3x3
(
     1.60475, -0.53108, -0.07367,
    -0.10208,  1.10813, -0.00605,
    -0.00327, -0.07276,  1.07602
);

vec3 RRTAndODTFit(vec3 v)
{
    vec3 a = v * (v + 0.0245786f) - 0.000090537f;
    vec3 b = v * (0.983729f * v + 0.4329510f) + 0.238081f;
    return a / b;
}

vec3 ACESFitted(vec3 color, float gamma)
{
    color = color * ACESInputMat;

    // Apply RRT and ODT
    color = RRTAndODTFit(color);

    color = color * ACESOutputMat;

    // Clamp to [0, 1]
  	color = clamp(color, 0.0, 1.0);
    
    color = pow(color, vec3(1. / gamma));

    return color;
}


uniform sampler2DRect hdr_image;
out vec4 oColor;
uniform float msweight;
void main(void)
{
  vec3 color = texture2DRect(hdr_image, gl_FragCoord.xy).rgb;
	oColor = vec4(color * msweight, 1);
  
}
)";
    opengl::Program *tmProg = nullptr;
    GLuint msfborgb = 0, msfbod = 0, tonemapfbo = 0;
    GLuint ssfborgb = 0, ssfbod = 0, sfbo = 0;
    GLuint texRect = 0, regularFBO = 0;
    GLuint texRects[16];
    GLuint emptyVAO = 0;
    void ScreenFillQuad(GLuint tex, float msweight, int samplei) {
        glDisable(GL_DEPTH_TEST);
        if (emptyVAO == 0)
            glGenVertexArrays(1, &emptyVAO);
        CHECK_GL(glViewport(0, 0, camera()->nx, camera()->ny));
        if (samplei == 0) {
            CHECK_GL(glClearColor(0, 0, 0, 0.0f));
            CHECK_GL(glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT));
        }
        tmProg->use();
        tmProg->set_uniformi("hdr_image", 0);
        tmProg->set_uniform("msweight", msweight);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_RECTANGLE, tex);

        glEnableVertexAttribArray(0);
        glBindVertexArray(emptyVAO);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
        glDisableVertexAttribArray(0);
        glUseProgram(0);
        glEnable(GL_DEPTH_TEST);
    }
    unsigned int getDepthTexture() const {
        return texRect;
    }
    void ZPass() {
        CHECK_GL(glBindFramebuffer(GL_DRAW_FRAMEBUFFER, tonemapfbo));
        CHECK_GL(glFramebufferRenderbuffer(GL_DRAW_FRAMEBUFFER,
                                           GL_COLOR_ATTACHMENT0,
                                           GL_RENDERBUFFER, msfborgb));
        CHECK_GL(glFramebufferRenderbuffer(
            GL_DRAW_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, msfbod));
        CHECK_GL(glDrawBuffer(GL_COLOR_ATTACHMENT0));
        CHECK_GL(glClearColor(0, 0, 0, 0));
        CHECK_GL(glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT));
        scene->my_paint_graphics(1.0, 1.0);
        CHECK_GL(glBindFramebuffer(GL_READ_FRAMEBUFFER, tonemapfbo));
        CHECK_GL(glBindFramebuffer(GL_DRAW_FRAMEBUFFER, regularFBO));
        CHECK_GL(glBindTexture(GL_TEXTURE_RECTANGLE, texRect));
        CHECK_GL(glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                                        GL_TEXTURE_RECTANGLE, texRect, 0));
        glBlitFramebuffer(0, 0, camera()->nx, camera()->ny, 0, 0, camera()->nx,
                          camera()->ny, GL_COLOR_BUFFER_BIT, GL_NEAREST);
    }

    void shadowPass() {
        auto &lights = scene->lights;
        for (auto &light : lights) {
            for (int i = 0; i < Light::cascadeCount + 1; i++) {
                light->BeginShadowMap(camera()->g_near, camera()->g_far,
                                      light->lightDir, camera()->proj,
                                      camera()->view, i);
                scene->vao->bind();
                for (auto const &gra : scene->graphics) {
                    gra->drawShadow(light.get());
                }
                scene->vao->unbind();
                light->EndShadowMap();
            }
        }
    }

    void paint_graphics(GLuint target_fbo = 0) {
        if (enable_hdr && tmProg == nullptr) {
            tmProg = scene->shaderMan->compile_program(qvert, qfrag);
            if (!tmProg) {
                enable_hdr = false;
            }
        }

        if (!enable_hdr || 1) {
            if (target_fbo)
                CHECK_GL(glBindFramebuffer(GL_DRAW_FRAMEBUFFER, target_fbo));
            return scene->my_paint_graphics(1.0, 0.0);
        }

        shadowPass();
        scene->mReflectivePass->reflectivePass();

        GLint zero_fbo = 0;
        CHECK_GL(glGetIntegerv(GL_FRAMEBUFFER_BINDING, &zero_fbo));
        GLint zero_draw_fbo = 0;
        CHECK_GL(glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &zero_draw_fbo));
        if (target_fbo == 0)
            target_fbo = zero_draw_fbo;

        if (msfborgb == 0 || camera()->oldnx != camera()->nx ||
            camera()->oldny != camera()->ny) {
            if (msfborgb != 0) {
                CHECK_GL(glDeleteRenderbuffers(1, &msfborgb));
            }
            if (ssfborgb != 0) {
                CHECK_GL(glDeleteRenderbuffers(1, &ssfborgb));
            }

            CHECK_GL(glGenRenderbuffers(1, &msfborgb));
            CHECK_GL(glBindRenderbuffer(GL_RENDERBUFFER, msfborgb));
            auto num_samples = camera()->getNumSamples();
            CHECK_GL(glRenderbufferStorageMultisample(
                GL_RENDERBUFFER, num_samples, GL_RGBA32F, camera()->nx,
                camera()->ny));

            CHECK_GL(glGenRenderbuffers(1, &ssfborgb));
            CHECK_GL(glBindRenderbuffer(GL_RENDERBUFFER, ssfborgb));
            /* begin cihou mesa */
            /* end cihou mesa */
            CHECK_GL(glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA32F,
                                           camera()->nx, camera()->ny));

            if (msfbod != 0) {
                CHECK_GL(glDeleteRenderbuffers(1, &msfbod));
            }
            CHECK_GL(glGenRenderbuffers(1, &msfbod));
            CHECK_GL(glBindRenderbuffer(GL_RENDERBUFFER, msfbod));
            CHECK_GL(glRenderbufferStorageMultisample(
                GL_RENDERBUFFER, num_samples, GL_DEPTH_COMPONENT32F,
                camera()->nx, camera()->ny));

            if (ssfbod != 0) {
                CHECK_GL(glDeleteRenderbuffers(1, &ssfbod));
            }
            CHECK_GL(glGenRenderbuffers(1, &ssfbod));
            CHECK_GL(glBindRenderbuffer(GL_RENDERBUFFER, ssfbod));
            CHECK_GL(glRenderbufferStorage(GL_RENDERBUFFER,
                                           GL_DEPTH_COMPONENT32F, camera()->nx,
                                           camera()->ny));

            if (tonemapfbo != 0) {
                CHECK_GL(glDeleteFramebuffers(1, &tonemapfbo));
            }
            CHECK_GL(glGenFramebuffers(1, &tonemapfbo));

            if (sfbo != 0) {
                CHECK_GL(glDeleteFramebuffers(1, &sfbo));
            }
            CHECK_GL(glGenFramebuffers(1, &sfbo));

            if (regularFBO != 0) {
                CHECK_GL(glDeleteFramebuffers(1, &regularFBO));
            }
            CHECK_GL(glGenFramebuffers(1, &regularFBO));
            if (texRect != 0) {
                CHECK_GL(glDeleteTextures(1, &texRect));
                for (int i = 0; i < 16; i++) {
                    CHECK_GL(glDeleteTextures(1, &texRects[i]));
                }
            }
            CHECK_GL(glGenTextures(1, &texRect));
            for (int i = 0; i < 16; i++) {
                CHECK_GL(glGenTextures(1, &texRects[i]));
            }
            CHECK_GL(glBindTexture(GL_TEXTURE_RECTANGLE, texRect));
            {
                CHECK_GL(glTexParameteri(GL_TEXTURE_RECTANGLE,
                                         GL_TEXTURE_MIN_FILTER, GL_NEAREST));
                CHECK_GL(glTexParameteri(GL_TEXTURE_RECTANGLE,
                                         GL_TEXTURE_MAG_FILTER, GL_NEAREST));
                CHECK_GL(glTexParameteri(GL_TEXTURE_RECTANGLE,
                                         GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE));
                CHECK_GL(glTexParameteri(GL_TEXTURE_RECTANGLE,
                                         GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE));

                CHECK_GL(glTexImage2D(GL_TEXTURE_RECTANGLE, 0, GL_RGBA32F,
                                      camera()->nx, camera()->ny, 0, GL_RGBA,
                                      GL_FLOAT, nullptr));
            }
            for (int i = 0; i < 16; i++) {
                CHECK_GL(glBindTexture(GL_TEXTURE_RECTANGLE, texRects[i]));
                {
                    CHECK_GL(glTexParameteri(GL_TEXTURE_RECTANGLE,
                                             GL_TEXTURE_MIN_FILTER,
                                             GL_NEAREST));
                    CHECK_GL(glTexParameteri(GL_TEXTURE_RECTANGLE,
                                             GL_TEXTURE_MAG_FILTER,
                                             GL_NEAREST));
                    CHECK_GL(glTexParameteri(GL_TEXTURE_RECTANGLE,
                                             GL_TEXTURE_WRAP_S,
                                             GL_CLAMP_TO_EDGE));
                    CHECK_GL(glTexParameteri(GL_TEXTURE_RECTANGLE,
                                             GL_TEXTURE_WRAP_T,
                                             GL_CLAMP_TO_EDGE));

                    CHECK_GL(glTexImage2D(GL_TEXTURE_RECTANGLE, 0, GL_RGBA32F,
                                          camera()->nx, camera()->ny, 0,
                                          GL_RGBA, GL_FLOAT, nullptr));
                }
            }
            CHECK_GL(glBindFramebuffer(GL_FRAMEBUFFER, regularFBO));
            CHECK_GL(glBindTexture(GL_TEXTURE_RECTANGLE, texRect));
            CHECK_GL(glFramebufferTexture2D(GL_FRAMEBUFFER,
                                            GL_COLOR_ATTACHMENT0,
                                            GL_TEXTURE_RECTANGLE, texRect, 0));
            CHECK_GL(glBindFramebuffer(GL_FRAMEBUFFER, zero_fbo));

            camera()->oldnx = camera()->nx;
            camera()->oldny = camera()->ny;
        }

        if (camera()->g_dof > 0) {

            for (int dofsample = 0; dofsample < 16; dofsample++) {
                glDisable(GL_MULTISAMPLE);
                glm::vec3 object =
                    camera()->g_camPos +
                    camera()->g_dof * glm::normalize(camera()->g_camView);
                glm::vec3 right = glm::normalize(
                    glm::cross(object - camera()->g_camPos, camera()->g_camUp));
                glm::vec3 p_up = glm::normalize(
                    glm::cross(right, object - camera()->g_camPos));
                glm::vec3 bokeh = right * cosf(dofsample * 2.0 * M_PI / 16.0) +
                                  p_up * sinf(dofsample * 2.0 * M_PI / 16.0);
                camera()->view = glm::lookAt(camera()->g_camPos + 0.05f * bokeh,
                                             object, p_up);
                //ZPass();
                CHECK_GL(glBindFramebuffer(GL_DRAW_FRAMEBUFFER, tonemapfbo));
                CHECK_GL(glFramebufferRenderbuffer(GL_DRAW_FRAMEBUFFER,
                                                   GL_COLOR_ATTACHMENT0,
                                                   GL_RENDERBUFFER, msfborgb));
                CHECK_GL(glFramebufferRenderbuffer(GL_DRAW_FRAMEBUFFER,
                                                   GL_DEPTH_ATTACHMENT,
                                                   GL_RENDERBUFFER, msfbod));
                CHECK_GL(glDrawBuffer(GL_COLOR_ATTACHMENT0));
                CHECK_GL(glClearColor(camera()->bgcolor.r, camera()->bgcolor.g,
                                      camera()->bgcolor.b, 0.0f));
                CHECK_GL(glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT));
                scene->my_paint_graphics(1.0, 0.0);
                CHECK_GL(glBindFramebuffer(GL_READ_FRAMEBUFFER, tonemapfbo));
                CHECK_GL(glBindFramebuffer(GL_DRAW_FRAMEBUFFER, regularFBO));
                CHECK_GL(
                    glBindTexture(GL_TEXTURE_RECTANGLE, texRects[dofsample]));
                CHECK_GL(glFramebufferTexture2D(
                    GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_RECTANGLE,
                    texRects[dofsample], 0));
                glBlitFramebuffer(0, 0, camera()->nx, camera()->ny, 0, 0,
                                  camera()->nx, camera()->ny,
                                  GL_COLOR_BUFFER_BIT, GL_NEAREST);
            }
            CHECK_GL(glBindFramebuffer(GL_DRAW_FRAMEBUFFER, tonemapfbo));
            CHECK_GL(glFramebufferRenderbuffer(GL_DRAW_FRAMEBUFFER,
                                               GL_COLOR_ATTACHMENT0,
                                               GL_RENDERBUFFER, msfborgb));
            CHECK_GL(glFramebufferRenderbuffer(GL_DRAW_FRAMEBUFFER,
                                               GL_DEPTH_ATTACHMENT,
                                               GL_RENDERBUFFER, msfbod));
            CHECK_GL(glDrawBuffer(GL_COLOR_ATTACHMENT0));
            glEnable(GL_BLEND);
            glBlendFunc(GL_ONE, GL_ONE);
            glad_glBlendEquation(GL_FUNC_ADD);
            for (int dofsample = 0; dofsample < 16; dofsample++) {
                //CHECK_GL(glBindFramebuffer(GL_READ_FRAMEBUFFER, regularFBO));

                ScreenFillQuad(texRects[dofsample], 1.0 / 16.0, dofsample);
            }
            CHECK_GL(glBindFramebuffer(GL_READ_FRAMEBUFFER, tonemapfbo));
            CHECK_GL(glBindFramebuffer(GL_DRAW_FRAMEBUFFER, regularFBO));
            CHECK_GL(glBindTexture(GL_TEXTURE_RECTANGLE, texRect));
            CHECK_GL(glFramebufferTexture2D(GL_FRAMEBUFFER,
                                            GL_COLOR_ATTACHMENT0,
                                            GL_TEXTURE_RECTANGLE, texRect, 0));
            glBlitFramebuffer(0, 0, camera()->nx, camera()->ny, 0, 0,
                              camera()->nx, camera()->ny, GL_COLOR_BUFFER_BIT,
                              GL_NEAREST);
            CHECK_GL(glBindFramebuffer(GL_DRAW_FRAMEBUFFER, target_fbo));
            ScreenFillQuad(texRect, 1.0, 0);

        } else {
            glDisable(GL_MULTISAMPLE);
            //ZPass();
            glm::vec3 object =
                camera()->g_camPos + 1.0f * glm::normalize(camera()->g_camView);
            glm::vec3 right = glm::normalize(
                glm::cross(object - camera()->g_camPos, camera()->g_camUp));
            glm::vec3 p_up =
                glm::normalize(glm::cross(right, object - camera()->g_camPos));
            camera()->view = glm::lookAt(camera()->g_camPos, object, p_up);
            CHECK_GL(glBindFramebuffer(GL_DRAW_FRAMEBUFFER, tonemapfbo));
            CHECK_GL(glFramebufferRenderbuffer(GL_DRAW_FRAMEBUFFER,
                                               GL_COLOR_ATTACHMENT0,
                                               GL_RENDERBUFFER, msfborgb));
            CHECK_GL(glFramebufferRenderbuffer(GL_DRAW_FRAMEBUFFER,
                                               GL_DEPTH_ATTACHMENT,
                                               GL_RENDERBUFFER, msfbod));
            CHECK_GL(glDrawBuffer(GL_COLOR_ATTACHMENT0));
            CHECK_GL(glClearColor(camera()->bgcolor.r, camera()->bgcolor.g,
                                  camera()->bgcolor.b, 0.0f));
            CHECK_GL(glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT));
            scene->my_paint_graphics(1.0, 0.0);
            CHECK_GL(glBindFramebuffer(GL_READ_FRAMEBUFFER, tonemapfbo));
            CHECK_GL(glBindFramebuffer(GL_DRAW_FRAMEBUFFER, regularFBO));
            CHECK_GL(glBindTexture(GL_TEXTURE_RECTANGLE, texRects[0]));
            CHECK_GL(
                glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                                       GL_TEXTURE_RECTANGLE, texRects[0], 0));
            glBlitFramebuffer(0, 0, camera()->nx, camera()->ny, 0, 0,
                              camera()->nx, camera()->ny, GL_COLOR_BUFFER_BIT,
                              GL_NEAREST);

            CHECK_GL(glBindFramebuffer(GL_READ_FRAMEBUFFER, regularFBO));
            CHECK_GL(glBindFramebuffer(GL_DRAW_FRAMEBUFFER, target_fbo));
            //tmProg->set_uniform("msweight",1.0);//already set in camera::set_program_uniforms
            ScreenFillQuad(texRects[0], 1.0, 0);
        }
        //std::this_thread::sleep_for(std::chrono::milliseconds(30));
        //glBlitFramebuffer(0, 0,camera()->nx,camera()->ny, 0, 0,camera()->nx,camera()->ny, GL_COLOR_BUFFER_BIT, GL_NEAREST);
        //drawScreenQuad here:
        //CHECK_GL(glFlush()); // delete this to cihou zeno2
    }

    /* END ZHXX HAPPY */

    ~DepthPass() {
        /* TODO: delete the C-style frame buffers */
    }
};

} // namespace zenovis
