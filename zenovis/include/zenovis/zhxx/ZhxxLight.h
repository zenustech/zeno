#pragma once

#include <array>
#include <cmath>
#include <cstdlib>
#include <glm/ext/matrix_clip_space.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <glm/geometric.hpp>
#include <zeno/utils/Error.h>
#include <zeno/utils/fuck_win.h>
#include <zeno/types/LightObject.h>
#include <zenovis/zhxx/ZhxxCamera.h>
#include <zenovis/zhxx/ZhxxScene.h>
#include <zenovis/opengl/shader.h>

namespace zenovis::zhxx {

struct ZhxxLight;

struct LightCluster : zeno::disable_copy {
    //std::vector<GLuint> DepthMaps;
    GLuint depthMapsArr{};
    GLuint lightFBO = 0;
    GLuint matricesUBO = 0;
    /* GLuint lightDepthMaps = 0; */
    int depthMapsArrCount = 0;

    static constexpr unsigned int depthMapResolution = 4096;
    static constexpr int layerCount = 8;
    static constexpr int lightCount = 16;
    static constexpr int cascadeCount = layerCount - 1;

    static std::vector<glm::vec4> getFrustumCornersWorldSpace(const glm::mat4 &projview) {
        const auto inv = glm::inverse(projview);
        std::vector<glm::vec4> frustumCorners;
        for (unsigned int x = 0; x < 2; ++x) {
            for (unsigned int y = 0; y < 2; ++y) {
                for (unsigned int z = 0; z < 2; ++z) {
                    const glm::vec4 pt = inv * glm::vec4(2.0f * x - 1.0f, 2.0f * y - 1.0f, 2.0f * z - 1.0f, 1.0f);
                    frustumCorners.push_back(pt / pt.w);
                }
            }
        }
        return frustumCorners;
    }

    ZhxxScene *const scene;

    std::vector<std::unique_ptr<ZhxxLight>> lights;

    explicit LightCluster(ZhxxScene *scene_) : scene(scene_) {
        initCascadeShadow();
    }

    ~LightCluster() {
        if (depthMapsArr != 0) {
            CHECK_GL(glDeleteTextures(1, &depthMapsArr));
            depthMapsArr = 0;
        }
        if (lightFBO != 0) {
            CHECK_GL(glDeleteFramebuffers(1, &lightFBO));
            lightFBO = 0;
        }
        if (matricesUBO != 0) {
            CHECK_GL(glDeleteBuffers(1, &matricesUBO));
            matricesUBO = 0;
        }
    }

    /* void setShadowMV(opengl::Program *shader) { */
    /*     glm::mat4 model = glm::mat4(1.0f); */
    /*     shader->set_uniform("mView", lightMV); */
    /* } */

//#define ZENO_LIGHT_USE_TEXTURE_LAYER
    void initCascadeShadow() {
        //DepthMaps.resize(layerCount);
        if (lightFBO == 0) {
            CHECK_GL(glGenFramebuffers(1, &lightFBO));
//#ifndef ZENO_LIGHT_USE_TEXTURE_LAYER
            //CHECK_GL(glGenTextures(1, &depthMapTmp));
            //CHECK_GL(glBindTexture(GL_TEXTURE_2D, depthMapTmp));
            //CHECK_GL(glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT32F, depthMapResolution, depthMapResolution, 0,
                                  //GL_DEPTH_COMPONENT, GL_FLOAT, nullptr));
            ////[> CHECK_GL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)); <]
            ////[> CHECK_GL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)); <]
            ////[> CHECK_GL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER)); <]
            ////[> CHECK_GL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER)); <]
            ////[> CHECK_GL(glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, borderColor)); <]
            //CHECK_GL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST));
            //CHECK_GL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST));
            //CHECK_GL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE));
            //CHECK_GL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE));
            //// attach depth texture as FBO's depth buffer
            //CHECK_GL(glBindFramebuffer(GL_FRAMEBUFFER, lightFBO));
            //CHECK_GL(glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depthMapTmp, 0));
            //[> CHECK_GL(glDrawBuffer(GL_NONE)); <] //??
            //[> CHECK_GL(glReadBuffer(GL_NONE)); <] //??

            //// glGenTextures(1, &lightDepthMaps);
            //// glBindTexture(GL_TEXTURE_2D_ARRAY, lightDepthMaps);
            //// glTexImage3D(
            ////     GL_TEXTURE_2D_ARRAY, 0, GL_DEPTH_COMPONENT32F, depthMapResolution, depthMapResolution, int(shadowCascadeLevels.size()) + 1,
            ////     0, GL_DEPTH_COMPONENT, GL_FLOAT, nullptr);

            //// glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
            //// glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
            //// glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
            //// glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);

            //// constexpr float bordercolor[] = { 1.0f, 1.0f, 1.0f, 1.0f };
            //// glTexParameterfv(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_BORDER_COLOR, bordercolor);

            //// glBindFramebuffer(GL_FRAMEBUFFER, lightFBO);
            //// glFramebufferTextureLayer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, lightDepthMaps, 0,0);
            //// glDrawBuffer(GL_NONE);
            //// glReadBuffer(GL_NONE);

            //[> CHECK_GL(glGenTextures(1, &depthMapTmp)); <]
            //[> CHECK_GL(glBindTexture(GL_TEXTURE_2D, depthMapTmp)); <]
            //[> // for (int i = 0; i < layerCount; i++) { <]
            //[> //CHECK_GL(glGenTextures(1, &(DepthMaps[i]))); <]
            //[> //CHECK_GL(glBindTexture(GL_TEXTURE_2D, DepthMaps[i])); <]
            //[> CHECK_GL(glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT32F, <]
            //[>         depthMapResolution, depthMapResolution, 0, <]
            //[>         GL_DEPTH_COMPONENT, GL_FLOAT, nullptr)); <]
            //[> CHECK_GL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)); <]
            //[> CHECK_GL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)); <]
            //[> CHECK_GL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP)); <]
            //[> CHECK_GL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP)); <]
            //[> //const float borderColor[] = {1.0f, 1.0f, 1.0f, 1.0f}; <]
            //[> //CHECK_GL(glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, <]
            //[>         //borderColor)); <]

            //int status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
            //if (status != GL_FRAMEBUFFER_COMPLETE) {
                //throw zeno::makeError("ERROR::FRAMEBUFFER:: Framebuffer is not complete!");
            //}
            //CHECK_GL(glBindFramebuffer(GL_FRAMEBUFFER, 0));
//#endif
            /* } */
        }
        if (matricesUBO == 0) {
            CHECK_GL(glGenBuffers(1, &matricesUBO));
            CHECK_GL(glBindBuffer(GL_UNIFORM_BUFFER, matricesUBO));
            CHECK_GL(glBufferData(GL_UNIFORM_BUFFER, sizeof(glm::mat4x4) * 16, nullptr, GL_STATIC_DRAW));
            CHECK_GL(glBindBufferBase(GL_UNIFORM_BUFFER, 0, matricesUBO));
            CHECK_GL(glBindBuffer(GL_UNIFORM_BUFFER, 0));
        }
    }

    void updateDepthMapsArr() {
        int licount = lights.size();
        if (depthMapsArrCount == licount) {
            return;
        }

        if (depthMapsArr != 0) {
            CHECK_GL(glDeleteTextures(1, &depthMapsArr));
            depthMapsArr = 0;
        }

        CHECK_GL(glBindFramebuffer(GL_FRAMEBUFFER, 0));

        CHECK_GL(glGenTextures(1, &depthMapsArr));
        CHECK_GL(glBindTexture(GL_TEXTURE_2D_ARRAY, depthMapsArr));
        /* for (int i = 0; i < layerCount; i++) { */
        //CHECK_GL(glGenTextures(1, &(DepthMaps[i])));
        //CHECK_GL(glBindTexture(GL_TEXTURE_2D, DepthMaps[i]));
        CHECK_GL(glTexImage3D(GL_TEXTURE_2D_ARRAY, 0, GL_DEPTH_COMPONENT32F, depthMapResolution, depthMapResolution,
                              layerCount * licount, 0, GL_DEPTH_COMPONENT, GL_FLOAT, nullptr));
        CHECK_GL(glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_NEAREST));
        CHECK_GL(glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_NEAREST));
        //CHECK_GL(glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER));
        //CHECK_GL(glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER));
        CHECK_GL(glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE));
        CHECK_GL(glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE));
        //float borderColor[] = {1.0f, 1.0f, 1.0f, 1.0f};
        //CHECK_GL(glTexParameterfv(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_BORDER_COLOR, borderColor));
        /* } */
        CHECK_GL(glBindFramebuffer(GL_FRAMEBUFFER, 0));
        depthMapsArrCount = licount;
    }

    template <class LightT = ZhxxLight> int addLight(zeno::LightData const &data) {
        int lightNo(lights.size());
        auto lit = std::make_unique<LightT>(this, lightNo);
        lit->setData(data);
        lights.push_back(std::move(lit));
        return lightNo;
    }

    void clearLights() {
        lights.clear();
    }

}; // struct LightCluster

struct ZhxxLight : zeno::disable_copy, zeno::LightData {
    LightCluster *const cluster;
    int myLightNo = 0;

    explicit ZhxxLight(LightCluster *cluster_, int myLightNo) : cluster(cluster_) {
        m_nearPlane.resize(LightCluster::layerCount);
        m_farPlane.resize(LightCluster::layerCount);
        setCascadeLevels(10000);
    }

    void setData(zeno::LightData const &dat) {
        static_cast<LightData &>(*this) = dat;
    }

    glm::mat4 lightMV;
    std::vector<glm::mat4> lightSpaceMatrices;
    std::vector<float> shadowCascadeLevels;
    std::vector<float> m_nearPlane;
    std::vector<float> m_farPlane;
    //glm::vec3 lightDir = glm::normalize(glm::vec3(1, 1, 0));
    //glm::vec3 shadowTint = glm::vec3(0.2f);
    //float lightHight = 1000.0;
    //float shadowSoftness = 1.0;
    //glm::vec3 lightColor = glm::vec3(1.0);
    //float intensity = 10.0;
    //float lightScale = 1.0;
    //bool isEnabled = true;

    glm::vec3 getShadowTint() const {
        return isEnabled ? zeno::vec_to_other<glm::vec3>(shadowTint) : glm::vec3(1.0);
    }

    glm::vec3 getIntensity() const {
        return isEnabled ? intensity * zeno::vec_to_other<glm::vec3>(lightColor) : glm::vec3(0.0);
    }

    glm::vec3 getLightDir() const {
        return zeno::vec_to_other<glm::vec3>(lightDir);
    }

    void setCascadeLevels(float far) {
        shadowCascadeLevels.resize(LightCluster::cascadeCount);
        shadowCascadeLevels[0] = far / 8192.0;
        shadowCascadeLevels[1] = far / 4096.0;
        shadowCascadeLevels[2] = far / 1024.0;
        shadowCascadeLevels[3] = far / 256.0;
        shadowCascadeLevels[4] = far / 32.0;
        shadowCascadeLevels[5] = far / 8.0;
        shadowCascadeLevels[6] = far / 2.0;
    }

    glm::mat4 getLightSpaceMatrix(int layer, const float nearPlane, const float farPlane, glm::mat4 const &proj,
                                  glm::mat4 const &view) {
        auto p = glm::perspective(glm::radians(cluster->scene->camera->m_fov), cluster->scene->camera->getAspect(),
                                  nearPlane, farPlane);
        const auto corners = LightCluster::getFrustumCornersWorldSpace(p * view);

        glm::vec3 center = glm::vec3(0, 0, 0);
        for (const auto &v : corners) {
            center += glm::vec3(v);
        }
        center /= corners.size();
        // std::cout<<center.x<<" "<<center.y<<" "<<center.z<<std::endl;
        auto lidir = getLightDir();
        glm::vec3 up = lidir.y > 0.99 ? glm::vec3(0, 0, 1) : glm::vec3(0, 1, 0);
        const auto lightView = glm::lookAt(center + lightHight * normalize(lidir), center, up);

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

        // Tune this parameter according to the scene //???
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
            glm::ortho(minX * lightScale, maxX * lightScale, minY * lightScale, maxY * lightScale, maxZ, -minZ);
        m_nearPlane[layer] = maxZ;
        m_farPlane[layer] = -minZ;
        lightMV = lightProjection * lightView;
        return lightProjection * lightView;
    }

    void calcLightSpaceMatrices(float near, float far, glm::mat4 const &proj, glm::mat4 const &view) {
        std::vector<glm::mat4> ret;
        for (size_t i = 0; i < LightCluster::layerCount; ++i) {
            if (i == 0) {
                ret.push_back(getLightSpaceMatrix(i, near, shadowCascadeLevels[i], proj, view));
            } else if (i < LightCluster::cascadeCount) {
                ret.push_back(getLightSpaceMatrix(i, shadowCascadeLevels[i - 1], shadowCascadeLevels[i], proj, view));
            } else {
                ret.push_back(getLightSpaceMatrix(i, shadowCascadeLevels[i - 1], far, proj, view));
            }
        }
        lightSpaceMatrices = std::move(ret);
    }

    void BeginShadowMap(float near, float far, glm::mat4 const &proj, glm::mat4 const &view, int i) {
        CHECK_GL(glDisable(GL_BLEND));
        CHECK_GL(glDisable(GL_DEPTH_TEST));
        CHECK_GL(glDisable(GL_PROGRAM_POINT_SIZE)); ///????ZHXX???
        CHECK_GL(glDisable(GL_MULTISAMPLE));
        CHECK_GL(glEnable(GL_DEPTH_TEST));
        CHECK_GL(glDepthFunc(GL_LESS));
        setCascadeLevels(far);

        // 0. UBO setup
        calcLightSpaceMatrices(near, far, proj, view);
        CHECK_GL(glBindBuffer(GL_UNIFORM_BUFFER, cluster->matricesUBO));
        for (size_t i = 0; i < lightSpaceMatrices.size(); ++i) {
            CHECK_GL(glBufferSubData(GL_UNIFORM_BUFFER, i * sizeof(glm::mat4x4), sizeof(glm::mat4x4),
                                     &lightSpaceMatrices[i]));
        }
        CHECK_GL(glBindBuffer(GL_UNIFORM_BUFFER, 0));

        // //1 shadow map
        // auto lightProjection = glm::ortho(-10.0f, 10.0f, -10.0f, 10.0f, -1000.0f,1000.0f);

        // auto lightView = glm::lookAt(glm::vec3(0.0f), glm::vec3(0.0f)-lightdir, glm::vec3(0.0, 1.0, 0.0));
        lightMV = lightSpaceMatrices[i];

        CHECK_GL(glViewport(0, 0, LightCluster::depthMapResolution, LightCluster::depthMapResolution));
        CHECK_GL(glBindFramebuffer(GL_FRAMEBUFFER, cluster->lightFBO));
        /* CHECK_GL(glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, */
        /*                        GL_TEXTURE_2D, depthMapTmp, 0)); */

//#ifdef ZENO_LIGHT_USE_TEXTURE_LAYER
        const int index = myLightNo * LightCluster::layerCount + i;
        CHECK_GL(glFramebufferTextureLayer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, cluster->depthMapsArr, 0, index));
        int status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
        if (status != GL_FRAMEBUFFER_COMPLETE) {
            throw zeno::makeError("ERROR::FRAMEBUFFER:: Framebuffer is not complete!");
        }
//#endif

        //CHECK_GL(glClearColor(0.0f, 0.0f, 0.0f, 0.0f));
        CHECK_GL(glClear(GL_DEPTH_BUFFER_BIT));

        // glEnable(GL_CULL_FACE);
        // glCullFace(GL_FRONT);  // peter panning
    }

    void EndShadowMap(int i) {
//#ifndef ZENO_LIGHT_USE_TEXTURE_LAYER
        //[> CHECK_GL(glReadBuffer(GL_COLOR_ATTACHMENT0)); <]
        //CHECK_GL(glBindTexture(GL_TEXTURE_2D_ARRAY, cluster->depthMapsArr));
        //const int index = myLightNo * LightCluster::layerCount + i;
        //CHECK_GL(glCopyTexSubImage3D(GL_TEXTURE_2D_ARRAY, 0, 0, 0, index, 0, 0, LightCluster::depthMapResolution,
                                     //LightCluster::depthMapResolution));
        //[> CHECK_GL(glReadBuffer(GL_NONE)); <]
//#endif

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
};

}; // namespace zenovis
