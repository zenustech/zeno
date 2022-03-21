#include "MyShader.hpp"
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
#include <zeno/types/MaterialObject.h>
#include <zeno/types/TextureObject.h>
namespace zenvis{
extern void setCascadeLevels(float far);
extern void initCascadeShadow();
extern std::vector<glm::mat4> getLightSpaceMatrices(float near, float far, glm::mat4 &proj, glm::mat4 &view);
extern void BeginShadowMap(float near, float far, glm::vec3 lightdir, glm::mat4 &proj, glm::mat4 &view);
extern void setShadowMV(Program* shader);
extern void EndShadowMap();
extern unsigned int getShadowMap();


static std::vector<float> shadowCascadeLevels(0);
void setCascadeLevels(float far)
{
    shadowCascadeLevels.resize(0);
    shadowCascadeLevels.push_back(far/8192.0);
    shadowCascadeLevels.push_back(far/2048.0);
    shadowCascadeLevels.push_back(far/512.0);
    shadowCascadeLevels.push_back(far/128.0);
    shadowCascadeLevels.push_back(far/32.0);
    shadowCascadeLevels.push_back(far/8.0);
    shadowCascadeLevels.push_back(far/2.0);
}

static glm::vec3 lightDir = glm::normalize(glm::vec3(1, 1, 0));
static unsigned int lightFBO=0;
static unsigned int lightDepthMaps=0;
unsigned int depthMapResolution = 4096;
static unsigned int matricesUBO = 0;
static void setShadowLightDir(glm::vec3 _dir)
{
    lightDir = _dir;
}
std::vector<glm::mat4> lightMatricesCache;
void initCascadeShadow()
{
    setCascadeLevels(10000);
    if(lightFBO==0)
    {
        CHECK_GL(glGenFramebuffers(1, &lightFBO));
        CHECK_GL(glGenTextures(1, &lightDepthMaps));
        CHECK_GL(glBindTexture(GL_TEXTURE_2D_ARRAY, lightDepthMaps));
        CHECK_GL(glTexImage3D(
            GL_TEXTURE_2D_ARRAY, 0, GL_DEPTH_COMPONENT32F, depthMapResolution, depthMapResolution, int(shadowCascadeLevels.size()) ,
            0, GL_DEPTH_COMPONENT, GL_FLOAT, nullptr));
        CHECK_GL(glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_NEAREST));
        CHECK_GL(glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_NEAREST));
        CHECK_GL(glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER));
        CHECK_GL(glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER));
        float bordercolor[4] = { 1.0f, 1.0f, 1.0f, 1.0f };
        CHECK_GL(glTexParameterfv(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_BORDER_COLOR, bordercolor));
        CHECK_GL(glBindFramebuffer(GL_FRAMEBUFFER, lightFBO));
        std::cout<<"6"<<std::endl;
        CHECK_GL(glFramebufferTextureLayer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, lightDepthMaps, 0,0));
        std::cout<<"7"<<std::endl;

        CHECK_GL(glDrawBuffer(GL_NONE));
        CHECK_GL(glReadBuffer(GL_NONE));
        std::cout<<"8"<<std::endl;

        
        int status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
        if (status != GL_FRAMEBUFFER_COMPLETE)
        {
            std::cout << "ERROR::FRAMEBUFFER:: Framebuffer is not complete!";
            throw 0;
        }

        CHECK_GL(glBindFramebuffer(GL_FRAMEBUFFER, 0));
    }
    if(matricesUBO==0)
    {
        CHECK_GL(glGenBuffers(1, &matricesUBO));
        CHECK_GL(glBindBuffer(GL_UNIFORM_BUFFER, matricesUBO));
        CHECK_GL(glBufferData(GL_UNIFORM_BUFFER, sizeof(glm::mat4x4) * 16, nullptr, GL_STATIC_DRAW));
        CHECK_GL(glBindBufferBase(GL_UNIFORM_BUFFER, 0, matricesUBO));
        CHECK_GL(glBindBuffer(GL_UNIFORM_BUFFER, 0));
    }
}
std::vector<glm::vec4> getFrustumCornersWorldSpace(const glm::mat4& projview)
{
    const auto inv = glm::inverse(projview);

    std::vector<glm::vec4> frustumCorners;
    for (unsigned int x = 0; x < 2; ++x)
    {
        for (unsigned int y = 0; y < 2; ++y)
        {
            for (unsigned int z = 0; z < 2; ++z)
            {
                const glm::vec4 pt = inv * glm::vec4(2.0f * x - 1.0f, 2.0f * y - 1.0f, 2.0f * z - 1.0f, 1.0f);
                frustumCorners.push_back(pt / pt.w);
            }
        }
    }

    return frustumCorners;
}


std::vector<glm::vec4> getFrustumCornersWorldSpace(const glm::mat4& proj, const glm::mat4& view)
{
    return getFrustumCornersWorldSpace(proj * view);
}
static glm::mat4 lightMV;
glm::mat4 getLightSpaceMatrix(const float nearPlane, const float farPlane, glm::mat4& proj, glm::mat4& view)
{
    
    const auto corners = getFrustumCornersWorldSpace(proj, view);

    glm::vec3 center = glm::vec3(0, 0, 0);
    for (const auto& v : corners)
    {
        center += glm::vec3(v);
    }
    center /= corners.size();
    glm::vec3 up = lightDir.y>0.99?glm::vec3(0,0,1):glm::vec3(0,1,0);
    const auto lightView = glm::lookAt(center + lightDir, center, up);
    lightMV = lightView;

    float minX = std::numeric_limits<float>::max();
    float maxX = std::numeric_limits<float>::min();
    float minY = std::numeric_limits<float>::max();
    float maxY = std::numeric_limits<float>::min();
    float minZ = std::numeric_limits<float>::max();
    float maxZ = std::numeric_limits<float>::min();
    for (const auto& v : corners)
    {
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
    if (minZ < 0)
    {
        minZ *= zMult;
    }
    else
    {
        minZ /= zMult;
    }
    if (maxZ < 0)
    {
        maxZ /= zMult;
    }
    else
    {
        maxZ *= zMult;
    }

    const glm::mat4 lightProjection = glm::ortho(minX, maxX, minY, maxY, minZ, maxZ);

    return lightProjection * lightView;
}

std::vector<glm::mat4> getLightSpaceMatrices(float near, float far, glm::mat4 &proj, glm::mat4 &view)
{
    std::vector<glm::mat4> ret;
    for (size_t i = 0; i < shadowCascadeLevels.size() + 1; ++i)
    {
        if (i == 0)
        {
            ret.push_back(getLightSpaceMatrix(near, shadowCascadeLevels[i], proj, view));
        }
        else if (i < shadowCascadeLevels.size())
        {
            ret.push_back(getLightSpaceMatrix(shadowCascadeLevels[i - 1], shadowCascadeLevels[i], proj, view));
        }
        else
        {
            ret.push_back(getLightSpaceMatrix(shadowCascadeLevels[i - 1], far, proj, view));
        }
    }
    return ret;
}
void BeginShadowMap(float near, float far, glm::vec3 lightdir, glm::mat4 &proj, glm::mat4 &view)
{
    setCascadeLevels(far);
    setShadowLightDir(lightdir);

    // 0. UBO setup
    const auto lightMatrices = getLightSpaceMatrices(near, far, proj, view);
    glBindBuffer(GL_UNIFORM_BUFFER, matricesUBO);
    for (size_t i = 0; i < lightMatrices.size(); ++i)
    {
        glBufferSubData(GL_UNIFORM_BUFFER, i * sizeof(glm::mat4x4), sizeof(glm::mat4x4), &lightMatrices[i]);
    }
    glBindBuffer(GL_UNIFORM_BUFFER, 0);

    glBindFramebuffer(GL_FRAMEBUFFER, lightFBO);
    glViewport(0, 0, depthMapResolution, depthMapResolution);
    glClear(GL_DEPTH_BUFFER_BIT);
    glEnable(GL_CULL_FACE);
    glCullFace(GL_FRONT);  // peter panning
}
void setShadowMV(Program* shader)
{
    glm::mat4 model = glm::mat4(1.0f);
    shader->set_uniform("mView", model);
}
void EndShadowMap()
{
    
    glDisable(GL_CULL_FACE);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glUseProgram(0);
}
unsigned int getShadowMap()
{
    return lightDepthMaps;
}

}