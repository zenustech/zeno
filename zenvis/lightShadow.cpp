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
#include <zeno/types/MaterialObject.h>
#include <zeno/types/TextureObject.h>

#ifdef _WIN32
    #ifdef near
        #undef near
    #endif
    #ifdef far
        #undef far
    #endif
#endif

namespace zenvis{
static glm::mat4 lightMV;
extern void setCascadeLevels(float far);
extern void initCascadeShadow();
extern std::vector<glm::mat4> getLightSpaceMatrices(float near, float far, glm::mat4 &proj, glm::mat4 &view);
extern void BeginShadowMap(float near, float far, glm::vec3 lightdir, glm::mat4 &proj, glm::mat4 &view, int i);
extern void setShadowMV(Program* shader);
extern void EndShadowMap();
extern unsigned int getShadowMap(int i);

extern glm::mat4 getLightMV()
{
    return lightMV;
}
static std::vector<float> shadowCascadeLevels(0);
extern std::vector<float> getCascadeDistances()
{
    return shadowCascadeLevels;
}
void setCascadeLevels(float far)
{
    shadowCascadeLevels.resize(0);
    shadowCascadeLevels.push_back(far/8192.0);
    shadowCascadeLevels.push_back(far/4096.0);
    shadowCascadeLevels.push_back(far/1024.0);
    shadowCascadeLevels.push_back(far/256.0);
    shadowCascadeLevels.push_back(far/32.0);
    shadowCascadeLevels.push_back(far/8.0);
    shadowCascadeLevels.push_back(far/2.0);
}
extern int getCascadeCount()
{
    return shadowCascadeLevels.size();
}
unsigned int reflectFBO = 0;
unsigned int reflectRBO = 0;
unsigned int reflectResolution = 2048;
std::vector<glm::mat4> reflectViews;
std::vector<glm::mat4> reflectMVPs;
std::vector<unsigned int> reflectiveMaps;
extern std::vector<unsigned int> getReflectMaps()
{
    return reflectiveMaps;
}
extern void setReflectMVP(int i, glm::mat4 mvp)
{
    reflectMVPs[i] = mvp;
}
extern glm::mat4 reflectView(glm::vec3 camPos, glm::vec3 viewDir, glm::vec3 up, glm::vec3 planeCenter, glm::vec3 planeNormal)
{
    glm::vec3 v = glm::normalize(viewDir);
    glm::vec3 R = v - 2.0f*glm::dot(v, planeNormal) * planeNormal;
    glm::vec3 RC = camPos + planeNormal * 2.0f * glm::dot((planeCenter - camPos), planeNormal);
    glm::vec3 Ru = up - 2.0f * glm::dot(up, planeNormal) * planeNormal;
    return glm::lookAt(RC, RC+R, Ru);

}

struct ReflectivePlane{
    glm::vec3 n;
    glm::vec3 c;
};
extern glm::mat4 getReflectViewMat(int i)
{
    return reflectViews[i];
}
extern glm::mat4 getReflectMVP(int i)
{
    return reflectMVPs[i];
}
std::vector<int> reflectMask;
std::vector<ReflectivePlane> ReflectivePlanes;
extern std::vector<ReflectivePlane> getReflectivePlanes()
{
    return ReflectivePlanes;
}
extern void setReflectivePlane(int i, glm::vec3 n, glm::vec3 c)
{
    
    if(i<0||i>=16)
        return;
    reflectMask[i] = 1;
    ReflectivePlane p;
    p.n = n; p.c = c;
    ReflectivePlanes[i] = p;

}
extern void clearReflectMask()
{
    reflectMask.assign(16,0);
}
extern bool renderReflect(int i)
{
    return reflectMask[i] == 1 ;
}
extern int getReflectivePlaneCount()
{ 
  return ReflectivePlanes.size();  
}
extern void setReflectivePlane(int i, glm::vec3 n, glm::vec3 c, glm::vec3 camPos, glm::vec3 camView, glm::vec3 camUp)
{
    reflectMask[i]=1;
    ReflectivePlane p;
    p.n = n; p.c = c;
    ReflectivePlanes[i] = p;
    reflectViews[i] = reflectView(camPos, camView, camUp, p.c, p.n);
}
extern void setReflectivePlane(int i, glm::vec3 camPos, glm::vec3 camView, glm::vec3 camUp)
{
    reflectMask[i] = 1;
    setReflectivePlane(i, ReflectivePlanes[i].n, ReflectivePlanes[i].c, camPos, camView, camUp);
}
static int reflectionID = -1;
extern void setReflectionViewID(int id)
{
    reflectionID = id;
}
extern int getReflectionViewID()
{
    return reflectionID;
}
static int mnx, mny;
static int moldnx, moldny;
extern void initReflectiveMaps(int nx, int ny)
{
    mnx = nx, mny = ny;
    moldnx = nx, moldny = ny;
    reflectiveMaps.resize(16);
    ReflectivePlanes.resize(16);
    reflectViews.resize(16);
    reflectMVPs.resize(16);
    reflectMask.assign(16,0);
    if(reflectFBO==0 && reflectRBO==0){
        CHECK_GL(glGenFramebuffers(1, &reflectFBO));
        CHECK_GL(glGenRenderbuffers(1, &reflectRBO));

        CHECK_GL(glBindFramebuffer(GL_FRAMEBUFFER, reflectFBO));
        CHECK_GL(glBindRenderbuffer(GL_RENDERBUFFER, reflectRBO));
        CHECK_GL(glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, reflectResolution, reflectResolution));
        CHECK_GL(glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, reflectRBO));
        
        for(int i=0;i<reflectiveMaps.size();i++)
        {
            CHECK_GL(glGenTextures(1, &(reflectiveMaps[i])));
            CHECK_GL(glBindTexture(GL_TEXTURE_RECTANGLE, reflectiveMaps[i]));
            CHECK_GL(glTexImage2D(GL_TEXTURE_RECTANGLE, 0, GL_RGB16F, nx, ny, 
            0, GL_RGB, GL_FLOAT, 0));
            CHECK_GL(glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE));
            CHECK_GL(glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE));
            CHECK_GL(glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_MIN_FILTER, GL_LINEAR));
            CHECK_GL(glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_MAG_FILTER, GL_LINEAR));
            

        }
    }
}
extern void updateReflectTexture(int nx, int ny)
{
    if(moldnx!=nx || moldny!=ny){
        CHECK_GL(glBindFramebuffer(GL_FRAMEBUFFER, 0));
        
        for(int i=0;i<reflectiveMaps.size();i++)
        {
            CHECK_GL(glBindTexture(GL_TEXTURE_RECTANGLE, reflectiveMaps[i]));
            CHECK_GL(glTexImage2D(GL_TEXTURE_RECTANGLE, 0, GL_RGB16F, nx, ny, 
            0, GL_RGB, GL_FLOAT, 0));
            CHECK_GL(glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE));
            CHECK_GL(glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE));
            CHECK_GL(glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_MIN_FILTER, GL_LINEAR));
            CHECK_GL(glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_MAG_FILTER, GL_LINEAR));
            

        }
        moldnx = nx;
        moldny = ny;
    }
}
extern void BeginReflective(int i, int nx, int ny)
{
    CHECK_GL(glDisable(GL_BLEND));
    CHECK_GL(glDisable(GL_DEPTH_TEST));
    CHECK_GL(glDisable(GL_PROGRAM_POINT_SIZE));///????ZHXX???
    CHECK_GL(glDisable(GL_MULTISAMPLE));
    CHECK_GL(glEnable(GL_DEPTH_TEST));
    CHECK_GL(glDepthFunc(GL_LESS));
    

    CHECK_GL(glBindTexture(GL_TEXTURE_RECTANGLE, reflectiveMaps[i]));
    CHECK_GL(glViewport(0, 0, nx, ny));
    CHECK_GL(glBindFramebuffer(GL_FRAMEBUFFER, reflectFBO));
    CHECK_GL(glBindRenderbuffer(GL_RENDERBUFFER, reflectRBO));
    CHECK_GL(glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, nx, ny));
    CHECK_GL(glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_RECTANGLE, reflectiveMaps[i], 0));
    

    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}
extern void BeginSecondReflective(int i, int nx, int ny)
{
    CHECK_GL(glDisable(GL_BLEND));
    CHECK_GL(glDisable(GL_DEPTH_TEST));
    CHECK_GL(glDisable(GL_PROGRAM_POINT_SIZE));///????ZHXX???
    CHECK_GL(glDisable(GL_MULTISAMPLE));
    CHECK_GL(glEnable(GL_DEPTH_TEST));
    CHECK_GL(glDepthFunc(GL_LESS));
    

    CHECK_GL(glBindTexture(GL_TEXTURE_RECTANGLE, reflectiveMaps[i+8]));
    CHECK_GL(glViewport(0, 0, nx, ny));
    CHECK_GL(glBindFramebuffer(GL_FRAMEBUFFER, reflectFBO));
    CHECK_GL(glBindRenderbuffer(GL_RENDERBUFFER, reflectRBO));
    CHECK_GL(glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, nx, ny));
    CHECK_GL(glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_RECTANGLE, reflectiveMaps[i+8], 0));
    

    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}
extern void EndReflective()
{
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glUseProgram(0);
    CHECK_GL(glEnable(GL_BLEND));
    CHECK_GL(glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA));
    CHECK_GL(glEnable(GL_DEPTH_TEST));
    CHECK_GL(glEnable(GL_PROGRAM_POINT_SIZE));
    //CHECK_GL(glEnable(GL_PROGRAM_POINT_SIZE_ARB));
    //CHECK_GL(glEnable(GL_POINT_SPRITE_ARB));
    //CHECK_GL(glEnable(GL_SAMPLE_COVERAGE));
    //CHECK_GL(glEnable(GL_SAMPLE_ALPHA_TO_COVERAGE));
    //CHECK_GL(glEnable(GL_SAMPLE_ALPHA_TO_ONE));
    CHECK_GL(glEnable(GL_MULTISAMPLE));

}
extern void EndSecondReflective()
{
    for(int i=0;i<8;i++)
    {

        auto temp = reflectiveMaps[i];
        reflectiveMaps[i]=reflectiveMaps[i+8];
        reflectiveMaps[i+8]=temp;
    }
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glUseProgram(0);
    CHECK_GL(glEnable(GL_BLEND));
    CHECK_GL(glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA));
    CHECK_GL(glEnable(GL_DEPTH_TEST));
    CHECK_GL(glEnable(GL_PROGRAM_POINT_SIZE));
    //CHECK_GL(glEnable(GL_PROGRAM_POINT_SIZE_ARB));
    //CHECK_GL(glEnable(GL_POINT_SPRITE_ARB));
    //CHECK_GL(glEnable(GL_SAMPLE_COVERAGE));
    //CHECK_GL(glEnable(GL_SAMPLE_ALPHA_TO_COVERAGE));
    //CHECK_GL(glEnable(GL_SAMPLE_ALPHA_TO_ONE));
    CHECK_GL(glEnable(GL_MULTISAMPLE));

}
static glm::vec3 lightDir = glm::normalize(glm::vec3(1, 1, 0));
static unsigned int lightFBO=0;
static unsigned int lightDepthMaps=0;
std::vector<unsigned int> DepthMaps;
unsigned int depthMapResolution = 8192;
static unsigned int matricesUBO = 0;
static void setShadowLightDir(glm::vec3 _dir)
{
    lightDir = _dir;
}
std::vector<glm::mat4> lightMatricesCache;
void initCascadeShadow()
{
    setCascadeLevels(10000);
    DepthMaps.resize(shadowCascadeLevels.size()+1);
    if(lightFBO==0)
    {
        CHECK_GL(glGenFramebuffers(1, &lightFBO));
        CHECK_GL(glGenTextures(1, &lightDepthMaps));
        CHECK_GL(glBindTexture(GL_TEXTURE_2D, lightDepthMaps));
        glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT32F, depthMapResolution, depthMapResolution, 
        0, GL_DEPTH_COMPONENT, GL_FLOAT, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
        float borderColor[] = { 1.0, 1.0, 1.0, 1.0 };
        glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, borderColor);
        // attach depth texture as FBO's depth buffer
        glBindFramebuffer(GL_FRAMEBUFFER, lightFBO);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, lightDepthMaps, 0);
        glDrawBuffer(GL_NONE);
        glReadBuffer(GL_NONE);

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
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        for(int i=0;i<DepthMaps.size();i++)
        {
            CHECK_GL(glGenTextures(1, &(DepthMaps[i])));
            CHECK_GL(glBindTexture(GL_TEXTURE_2D, DepthMaps[i]));
            glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT32F, depthMapResolution, depthMapResolution, 
            0, GL_DEPTH_COMPONENT, GL_FLOAT, nullptr);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
            float borderColor[] = { 1.0, 1.0, 1.0, 1.0 };
            glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, borderColor);

        }


        
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
float gfov;
float gaspect;
extern void setfov(float f)
{
    gfov = f;
}
extern void setaspect(float f)
{
    gaspect = f;
}
static float LightHight=1000.0;
extern void setLightHight(float h)
{
    LightHight = h;
}
extern float getLightHight()
{
    return LightHight;
}

glm::mat4 getLightSpaceMatrix(const float nearPlane, const float farPlane, glm::mat4& proj, glm::mat4& view)
{
    auto p = glm::perspective(glm::radians(gfov), gaspect, nearPlane, farPlane);
    const auto corners = getFrustumCornersWorldSpace(p, view);

    glm::vec3 center = glm::vec3(0, 0, 0);
    for (const auto& v : corners)
    {
        center += glm::vec3(v);
    }
    center /= corners.size();
    //std::cout<<center.x<<" "<<center.y<<" "<<center.z<<std::endl;
    glm::vec3 up = lightDir.y>0.99?glm::vec3(0,0,1):glm::vec3(0,1,0);
    const auto lightView = glm::lookAt(center + LightHight*normalize(lightDir), center, up);

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

    float size = std::max(maxX-minX, maxY-minY);
    float midX = 0.5*(maxX + minX);
    float midY = 0.5*(maxY + minY);
    minX = midX - size; maxX = midX + size;
    minY = midY - size; maxY = maxY + size;
    const glm::mat4 lightProjection = glm::ortho(minX*20, maxX*20, minY*20, maxY*20, maxZ, -minZ);
    //std::cout<<minX<<" "<<maxX<<" "<<minY<<" "<<maxY<<" "<<minZ<<" "<<maxZ<<std::endl;
    lightMV = lightProjection * lightView;
    return lightProjection * lightView;
}
static std::vector<glm::mat4> lightSpaceMatrices;
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
            ret.push_back(getLightSpaceMatrix(shadowCascadeLevels[i-1], shadowCascadeLevels[i], proj, view));
        }
        else
        {
            ret.push_back(getLightSpaceMatrix(shadowCascadeLevels[i-1], far, proj, view));
        }
    }
    lightSpaceMatrices = ret;
    return ret;
}
extern std::vector<glm::mat4> getLightSpaceMatrices()
{
    return lightSpaceMatrices;
}
void printMat4(glm::mat4 &m)
{
    for(int j=0;j<4;j++)
    {
        for(int i=0;i<4;i++)
        {
            std::cout<<m[j][i]<<" ";
        }
        std::cout<<std::endl;
    }
}
static glm::mat4 lightSpaceMatrix;
void BeginShadowMap(float near, float far, glm::vec3 lightdir, glm::mat4 &proj, glm::mat4 &view, int i)
{
  CHECK_GL(glDisable(GL_BLEND));
  CHECK_GL(glDisable(GL_DEPTH_TEST));
  CHECK_GL(glDisable(GL_PROGRAM_POINT_SIZE));///????ZHXX???
  CHECK_GL(glDisable(GL_MULTISAMPLE));
  CHECK_GL(glEnable(GL_DEPTH_TEST));
  CHECK_GL(glDepthFunc(GL_LESS));
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


    // //1 shadow map
    // auto lightProjection = glm::ortho(-10.0f, 10.0f, -10.0f, 10.0f, -1000.0f,1000.0f);
    
    // auto lightView = glm::lookAt(glm::vec3(0.0f), glm::vec3(0.0f)-lightdir, glm::vec3(0.0, 1.0, 0.0));
    lightSpaceMatrix = lightMatrices[i];
    lightMV = lightSpaceMatrix;

    
    glViewport(0, 0, depthMapResolution, depthMapResolution);
    glBindFramebuffer(GL_FRAMEBUFFER, lightFBO);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, DepthMaps[i], 0);
    

    glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
    glClear(GL_DEPTH_BUFFER_BIT);

    //glEnable(GL_CULL_FACE);
    //glCullFace(GL_FRONT);  // peter panning
}
void setShadowMV(Program* shader)
{
    glm::mat4 model = glm::mat4(1.0f);
    shader->set_uniform("mView", lightMV);
}
void EndShadowMap()
{
    
    //glDisable(GL_CULL_FACE);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glUseProgram(0);
    CHECK_GL(glEnable(GL_BLEND));
  CHECK_GL(glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA));
  CHECK_GL(glEnable(GL_DEPTH_TEST));
  CHECK_GL(glEnable(GL_PROGRAM_POINT_SIZE));
  //CHECK_GL(glEnable(GL_PROGRAM_POINT_SIZE_ARB));
  //CHECK_GL(glEnable(GL_POINT_SPRITE_ARB));
  //CHECK_GL(glEnable(GL_SAMPLE_COVERAGE));
  //CHECK_GL(glEnable(GL_SAMPLE_ALPHA_TO_COVERAGE));
  //CHECK_GL(glEnable(GL_SAMPLE_ALPHA_TO_ONE));
  CHECK_GL(glEnable(GL_MULTISAMPLE));
}
unsigned int getShadowMap(int i)
{
    return DepthMaps[i];
}

}
