#pragma once

#include <array>
#include <cmath>
#include <cstdlib>
#include <glm/ext/matrix_clip_space.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <glm/geometric.hpp>
#include <zeno/utils/Error.h>
#include <zeno/utils/fuck_win.h>
#include <zenovis/opengl/shader.h>

namespace zenovis {

struct Light {
    glm::mat4 lightMV;
    glm::mat4 lightSpaceMatrix;
    std::vector<glm::mat4> lightSpaceMatrices;
    std::vector<float> shadowCascadeLevels;
    //std::vector<GLuint> DepthMaps;
    GLuint depthMapsArr{};
    GLuint depthMapTmp{};
    std::vector<float> m_nearPlane;
    std::vector<float> m_farPlane;
    glm::vec3 lightDir = glm::normalize(glm::vec3(1, 1, 0));
    glm::vec3 shadowTint = glm::vec3(0.2f);
    float lightHight = 1000.0;
    float gfov{};
    float gaspect{};
    float shadowSoftness = 1.0;
    unsigned int lightFBO = 0;
    unsigned int lightDepthMaps = 0;
    unsigned int depthMapResolution = 4096;
    unsigned int matricesUBO = 0;
    static constexpr int cascadeCount = 7;
    glm::vec3 lightColor = glm::vec3(1.0);
    float intensity = 10.0;
    float lightScale = 1.0;
    bool m_isEnabled = true;

    glm::vec3 getShadowTint() {
        return m_isEnabled ? shadowTint : glm::vec3(1.0);
    }

    glm::vec3 getIntensity() {
        return m_isEnabled ? intensity * lightColor : glm::vec3(0.0);
    }

    Light() {
        initCascadeShadow();
    }

    void setCascadeLevels(float far) {
        shadowCascadeLevels.resize(cascadeCount);
        shadowCascadeLevels[0] = far / 8192.0;
        shadowCascadeLevels[1] = far / 4096.0;
        shadowCascadeLevels[2] = far / 1024.0;
        shadowCascadeLevels[3] = far / 256.0;
        shadowCascadeLevels[4] = far / 32.0;
        shadowCascadeLevels[5] = far / 8.0;
        shadowCascadeLevels[6] = far / 2.0;
    }

    /* void setShadowMV(opengl::Program *shader) { */
    /*     glm::mat4 model = glm::mat4(1.0f); */
    /*     shader->set_uniform("mView", lightMV); */
    /* } */

    std::vector<glm::vec4>
    getFrustumCornersWorldSpace(const glm::mat4 &projview) {
        const auto inv = glm::inverse(projview);

        std::vector<glm::vec4> frustumCorners;
        for (unsigned int x = 0; x < 2; ++x) {
            for (unsigned int y = 0; y < 2; ++y) {
                for (unsigned int z = 0; z < 2; ++z) {
                    const glm::vec4 pt =
                        inv * glm::vec4(2.0f * x - 1.0f, 2.0f * y - 1.0f,
                                        2.0f * z - 1.0f, 1.0f);
                    frustumCorners.push_back(pt / pt.w);
                }
            }
        }

        return frustumCorners;
    }

    std::vector<glm::vec4> getFrustumCornersWorldSpace(const glm::mat4 &proj,
                                                       const glm::mat4 &view) {
        return getFrustumCornersWorldSpace(proj * view);
    }

    glm::mat4 getLightSpaceMatrix(int layer, const float nearPlane,
                                  const float farPlane, glm::mat4 &proj,
                                  glm::mat4 &view) {
        auto p =
            glm::perspective(glm::radians(gfov), gaspect, nearPlane, farPlane);
        const auto corners = getFrustumCornersWorldSpace(p, view);

        glm::vec3 center = glm::vec3(0, 0, 0);
        for (const auto &v : corners) {
            center += glm::vec3(v);
        }
        center /= corners.size();
        // std::cout<<center.x<<" "<<center.y<<" "<<center.z<<std::endl;
        glm::vec3 up =
            lightDir.y > 0.99 ? glm::vec3(0, 0, 1) : glm::vec3(0, 1, 0);
        const auto lightView =
            glm::lookAt(center + lightHight * normalize(lightDir), center, up);

        float minX = std::numeric_limits<float>::max();
        float maxX = std::numeric_limits<float>::min();
        float minY = std::numeric_limits<float>::max();
        float maxY = std::numeric_limits<float>::min();
        float minZ = std::numeric_limits<float>::max();
        float maxZ = std::numeric_limits<float>::min();
        for (const auto &v : corners) {
            const auto trf = lightView * v;
            minX = std::min(minX, trf.x);
            maxX = std::max(maxX, trf.x);
            minY = std::min(minY, trf.y);
            maxY = std::max(maxY, trf.y);
            minZ = std::min(minZ, trf.z);
            maxZ = std::max(maxZ, trf.z);
        }

        // Tune this parameter according to the scene
        float zMult = 10.0f;
        if (minZ < 0) {
            minZ *= zMult;
        } else {
            minZ /= zMult;
        }
        if (maxZ < 0) {
            maxZ /= zMult;
        } else {
            maxZ *= zMult;
        }

        float size = std::max(maxX - minX, maxY - minY);
        float midX = 0.5 * (maxX + minX);
        float midY = 0.5 * (maxY + minY);
        minX = midX - size;
        maxX = midX + size;
        minY = midY - size;
        maxY = maxY + size;
        const glm::mat4 lightProjection =
            glm::ortho(minX * lightScale, maxX * lightScale, minY * lightScale,
                       maxY * lightScale, maxZ, -minZ);
        m_nearPlane[layer] = maxZ;
        m_farPlane[layer] = -minZ;
        lightMV = lightProjection * lightView;
        return lightProjection * lightView;
    }

    std::vector<glm::mat4> getLightSpaceMatrices(float near, float far,
                                                 glm::mat4 &proj,
                                                 glm::mat4 &view) {
        std::vector<glm::mat4> ret;
        for (size_t i = 0; i < cascadeCount + 1; ++i) {
            if (i == 0) {
                ret.push_back(getLightSpaceMatrix(
                    i, near, shadowCascadeLevels[i], proj, view));
            } else if (i < cascadeCount) {
                ret.push_back(getLightSpaceMatrix(i, shadowCascadeLevels[i - 1],
                                                  shadowCascadeLevels[i], proj,
                                                  view));
            } else {
                ret.push_back(getLightSpaceMatrix(i, shadowCascadeLevels[i - 1],
                                                  far, proj, view));
            }
        }
        lightSpaceMatrices = ret;
        return ret;
    }

    void initCascadeShadow() {
        setCascadeLevels(10000);
        //DepthMaps.resize(cascadeCount + 1);
        m_nearPlane.resize(cascadeCount + 1);
        m_farPlane.resize(cascadeCount + 1);
        if (lightFBO == 0) {
            CHECK_GL(glGenFramebuffers(1, &lightFBO));
            CHECK_GL(glGenTextures(1, &lightDepthMaps));
            CHECK_GL(glBindTexture(GL_TEXTURE_2D, lightDepthMaps));
            CHECK_GL(glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT32F,
                         depthMapResolution, depthMapResolution, 0,
                         GL_DEPTH_COMPONENT, GL_FLOAT, nullptr));
            /* CHECK_GL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)); */
            /* CHECK_GL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)); */
            /* CHECK_GL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER)); */
            /* CHECK_GL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER)); */
            /* CHECK_GL(glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, borderColor)); */
            CHECK_GL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST));
            CHECK_GL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST));
            CHECK_GL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP));
            CHECK_GL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP));
            // attach depth texture as FBO's depth buffer
            CHECK_GL(glBindFramebuffer(GL_FRAMEBUFFER, lightFBO));
            CHECK_GL(glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, lightDepthMaps, 0));
            CHECK_GL(glDrawBuffer(GL_NONE));
            CHECK_GL(glReadBuffer(GL_NONE));

            // glGenTextures(1, &lightDepthMaps);
            // glBindTexture(GL_TEXTURE_2D_ARRAY, lightDepthMaps);
            // glTexImage3D(
            //     GL_TEXTURE_2D_ARRAY, 0, GL_DEPTH_COMPONENT32F, depthMapResolution, depthMapResolution, int(shadowCascadeLevels.size()) + 1,
            //     0, GL_DEPTH_COMPONENT, GL_FLOAT, nullptr);

            // glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
            // glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
            // glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
            // glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);

            // constexpr float bordercolor[] = { 1.0f, 1.0f, 1.0f, 1.0f };
            // glTexParameterfv(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_BORDER_COLOR, bordercolor);

            // glBindFramebuffer(GL_FRAMEBUFFER, lightFBO);
            // glFramebufferTextureLayer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, lightDepthMaps, 0,0);
            // glDrawBuffer(GL_NONE);
            // glReadBuffer(GL_NONE);
            CHECK_GL(glBindFramebuffer(GL_FRAMEBUFFER, 0));

            CHECK_GL(glGenTextures(1, &depthMapTmp));
            CHECK_GL(glBindTexture(GL_TEXTURE_2D, depthMapTmp));
            /* for (int i = 0; i < cascadeCount + 1; i++) { */
            //CHECK_GL(glGenTextures(1, &(DepthMaps[i])));
            //CHECK_GL(glBindTexture(GL_TEXTURE_2D, DepthMaps[i]));
            CHECK_GL(glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT32F,
                    depthMapResolution, depthMapResolution, 0,
                    GL_DEPTH_COMPONENT, GL_FLOAT, nullptr));
            CHECK_GL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,
                    GL_LINEAR));
            CHECK_GL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER,
                    GL_LINEAR));
            CHECK_GL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S,
                    GL_CLAMP_TO_BORDER));
            CHECK_GL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T,
                    GL_CLAMP_TO_BORDER));
            const float borderColor[] = {1.0f, 1.0f, 1.0f, 1.0f};
            CHECK_GL(glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR,
                    borderColor));

            int status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
            if (status != GL_FRAMEBUFFER_COMPLETE) {
                throw zeno::makeError(
                    "ERROR::FRAMEBUFFER:: Framebuffer is not complete!\n");
            }

            CHECK_GL(glBindFramebuffer(GL_FRAMEBUFFER, 0));

            CHECK_GL(glGenTextures(1, &depthMapsArr));
            CHECK_GL(glBindTexture(GL_TEXTURE_2D_ARRAY, depthMapsArr));
            /* for (int i = 0; i < cascadeCount + 1; i++) { */
            //CHECK_GL(glGenTextures(1, &(DepthMaps[i])));
            //CHECK_GL(glBindTexture(GL_TEXTURE_2D, DepthMaps[i]));
            CHECK_GL(glTexImage3D(GL_TEXTURE_2D_ARRAY, 0, GL_DEPTH_COMPONENT32F,
                    depthMapResolution, depthMapResolution, (cascadeCount + 1) * 2, 0,
                    GL_DEPTH_COMPONENT, GL_FLOAT, nullptr));
            CHECK_GL(glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER,
                    GL_LINEAR));
            CHECK_GL(glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER,
                    GL_LINEAR));
            CHECK_GL(glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S,
                    GL_CLAMP_TO_BORDER));
            CHECK_GL(glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T,
                    GL_CLAMP_TO_BORDER));
            //float borderColor[] = {1.0f, 1.0f, 1.0f, 1.0f};
            CHECK_GL(glTexParameterfv(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_BORDER_COLOR,
                    borderColor));
            /* } */
            CHECK_GL(glBindFramebuffer(GL_FRAMEBUFFER, 0));
        /* } */
        }
        if (matricesUBO == 0) {
            CHECK_GL(glGenBuffers(1, &matricesUBO));
            CHECK_GL(glBindBuffer(GL_UNIFORM_BUFFER, matricesUBO));
            CHECK_GL(glBufferData(GL_UNIFORM_BUFFER, sizeof(glm::mat4x4) * 16,
                                  nullptr, GL_STATIC_DRAW));
            CHECK_GL(glBindBufferBase(GL_UNIFORM_BUFFER, 0, matricesUBO));
            CHECK_GL(glBindBuffer(GL_UNIFORM_BUFFER, 0));
        }
    }

    void BeginShadowMap(float near, float far, glm::vec3 lightdir,
                        glm::mat4 &proj, glm::mat4 &view, int i) {
        CHECK_GL(glDisable(GL_BLEND));
        CHECK_GL(glDisable(GL_DEPTH_TEST));
        CHECK_GL(glDisable(GL_PROGRAM_POINT_SIZE)); ///????ZHXX???
        CHECK_GL(glDisable(GL_MULTISAMPLE));
        CHECK_GL(glEnable(GL_DEPTH_TEST));
        CHECK_GL(glDepthFunc(GL_LESS));
        setCascadeLevels(far);
        lightDir = lightdir;

        // 0. UBO setup
        const auto lightMatrices = getLightSpaceMatrices(near, far, proj, view);
        CHECK_GL(glBindBuffer(GL_UNIFORM_BUFFER, matricesUBO));
        for (size_t i = 0; i < lightMatrices.size(); ++i) {
            CHECK_GL(glBufferSubData(GL_UNIFORM_BUFFER, i * sizeof(glm::mat4x4),
                            sizeof(glm::mat4x4), &lightMatrices[i]));
        }
        CHECK_GL(glBindBuffer(GL_UNIFORM_BUFFER, 0));

        // //1 shadow map
        // auto lightProjection = glm::ortho(-10.0f, 10.0f, -10.0f, 10.0f, -1000.0f,1000.0f);

        // auto lightView = glm::lookAt(glm::vec3(0.0f), glm::vec3(0.0f)-lightdir, glm::vec3(0.0, 1.0, 0.0));
        lightSpaceMatrix = lightMatrices[i];
        lightMV = lightSpaceMatrix;

        CHECK_GL(glViewport(0, 0, depthMapResolution, depthMapResolution));
        CHECK_GL(glBindFramebuffer(GL_FRAMEBUFFER, lightFBO));
        CHECK_GL(glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
                               GL_TEXTURE_2D, depthMapTmp, 0));

        CHECK_GL(glClearColor(0.1f, 0.1f, 0.1f, 1.0f));
        CHECK_GL(glClear(GL_DEPTH_BUFFER_BIT));

        // glEnable(GL_CULL_FACE);
        // glCullFace(GL_FRONT);  // peter panning
    }

    void EndShadowMap(int i) {
        CHECK_GL(glReadBuffer(GL_DEPTH_ATTACHMENT));
        CHECK_GL(glCopyTexSubImage3D(GL_TEXTURE_2D_ARRAY, 0,
                    0, 0, i, 0, 0, depthMapResolution, depthMapResolution));
        CHECK_GL(glReadBuffer(GL_NONE));

        // glDisable(GL_CULL_FACE);
        CHECK_GL(glBindFramebuffer(GL_FRAMEBUFFER, 0));
        CHECK_GL(glUseProgram(0));
        CHECK_GL(glEnable(GL_BLEND));
        CHECK_GL(glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA));
        CHECK_GL(glEnable(GL_DEPTH_TEST));
        CHECK_GL(glEnable(GL_PROGRAM_POINT_SIZE));
        // CHECK_GL(glEnable(GL_PROGRAM_POINT_SIZE_ARB));
        // CHECK_GL(glEnable(GL_POINT_SPRITE_ARB));
        // CHECK_GL(glEnable(GL_SAMPLE_COVERAGE));
        // CHECK_GL(glEnable(GL_SAMPLE_ALPHA_TO_COVERAGE));
        // CHECK_GL(glEnable(GL_SAMPLE_ALPHA_TO_ONE));
        CHECK_GL(glEnable(GL_MULTISAMPLE));
    }

}; // struct Light

}; // namespace zenovis
