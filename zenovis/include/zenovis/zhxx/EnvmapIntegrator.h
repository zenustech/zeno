#pragma once

#include <array>
#include <cmath>
#include <cstdlib>
#include <sstream>
#include <stb_image.h>
#include <unordered_map>
#include <zeno/utils/log.h>
#include <zenovis/opengl/shader.h>
#include <zenovis/ShaderManager.h>
#include <zenovis/zhxx/Scene.h>

namespace zenovis::zhxx {
/* begin zhxx happy */

struct EnvmapIntegrator {

Scene *scene;

explicit EnvmapIntegrator(Scene *scene) : scene(scene) {}

constexpr static const char *VS = R"(
#version 330 core
layout (location = 0) in vec3 aPos;

out vec3 WorldPos;

uniform mat4 projection;
uniform mat4 view;

void main()
{
    WorldPos = aPos;  
    gl_Position =  projection * view * vec4(WorldPos, 1.0);
}
)";
constexpr static const char *SPFS = R"(
#version 330 core
out vec4 FragColor;
in vec3 WorldPos;

uniform samplerCube environmentMap;
uniform float roughness;

const float PI = 3.14159265359;
// ----------------------------------------------------------------------------
float DistributionGGX(vec3 N, vec3 H, float roughness)
{
    float a = roughness*roughness;
    float a2 = a*a;
    float NdotH = max(dot(N, H), 0.0);
    float NdotH2 = NdotH*NdotH;

    float nom   = a2;
    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = PI * denom * denom;

    return nom / denom;
}
// ----------------------------------------------------------------------------
// http://holger.dammertz.org/stuff/notes_HammersleyOnHemisphere.html
// efficient VanDerCorpus calculation.
float RadicalInverse_VdC(uint bits) 
{
     bits = (bits << 16u) | (bits >> 16u);
     bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
     bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
     bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
     bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
     return float(bits) * 2.3283064365386963e-10; // / 0x100000000
}
// ----------------------------------------------------------------------------
vec2 Hammersley(uint i, uint N)
{
	return vec2(float(i)/float(N), RadicalInverse_VdC(i));
}
// ----------------------------------------------------------------------------
vec3 ImportanceSampleGGX(vec2 Xi, vec3 N, float roughness)
{
	float a = roughness*roughness;
	
	float phi = 2.0 * PI * Xi.x;
	float cosTheta = sqrt((1.0 - Xi.y) / (1.0 + (a*a - 1.0) * Xi.y));
	float sinTheta = sqrt(1.0 - cosTheta*cosTheta);
	
	// from spherical coordinates to cartesian coordinates - halfway vector
	vec3 H;
	H.x = cos(phi) * sinTheta;
	H.y = sin(phi) * sinTheta;
	H.z = cosTheta;
	
	// from tangent-space H vector to world-space sample vector
	vec3 up          = abs(N.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
	vec3 tangent   = normalize(cross(up, N));
	vec3 bitangent = cross(N, tangent);
	
	vec3 sampleVec = tangent * H.x + bitangent * H.y + N * H.z;
	return normalize(sampleVec);
}
// ----------------------------------------------------------------------------
void main()
{		
    vec3 N = normalize(WorldPos);
    
    // make the simplyfying assumption that V equals R equals the normal 
    vec3 R = N;
    vec3 V = R;

    const uint SAMPLE_COUNT = 2048u;
    vec3 prefilteredColor = vec3(0.0);
    float totalWeight = 0.0;
    
    for(uint i = 0u; i < SAMPLE_COUNT; ++i)
    {
        // generates a sample vector that's biased towards the preferred alignment direction (importance sampling).
        vec2 Xi = Hammersley(i, SAMPLE_COUNT);
        vec3 H = ImportanceSampleGGX(Xi, N, roughness);
        vec3 L  = normalize(2.0 * dot(V, H) * H - V);

        float NdotL = max(dot(N, L), 0.0);
        if(NdotL > 0.0)
        {
            // sample from the environment's mip level based on roughness/pdf
            float D   = DistributionGGX(N, H, roughness);
            float NdotH = max(dot(N, H), 0.0);
            float HdotV = max(dot(H, V), 0.0);
            float pdf = D * NdotH / (4.0 * HdotV) + 0.0001; 

            float resolution = 1024.0; // resolution of source cubemap (per face)
            float saTexel  = 4.0 * PI / (6.0 * resolution * resolution);
            float saSample = 1.0 / (float(SAMPLE_COUNT) * pdf + 0.0001);

            float mipLevel = roughness == 0.0 ? 0.0 : 0.5 * log2(saSample / saTexel); 
            
            prefilteredColor += textureLod(environmentMap, L, mipLevel).rgb * NdotL;
            totalWeight      += NdotL;
        }
    }

    prefilteredColor = prefilteredColor / totalWeight;

    FragColor = vec4(prefilteredColor, 1.0);
}
)";
constexpr static const char *IRFS = R"(
#version 330 core
out vec4 FragColor;
in vec3 WorldPos;

uniform samplerCube environmentMap;

const float PI = 3.14159265359;

void main()
{		
    vec3 N = normalize(WorldPos);

    vec3 irradiance = vec3(0.0);   
    
    // tangent space calculation from origin point
    vec3 up    = vec3(0.0, 1.0, 0.0);
    vec3 right = normalize(cross(up, N));
    up         = normalize(cross(N, right));
       
    float sampleDelta = 0.025;
    float nrSamples = 0.0f;
    for(float phi = 0.0; phi < 2.0 * PI; phi += sampleDelta)
    {
        for(float theta = 0.0; theta < 0.5 * PI; theta += sampleDelta)
        {
            // spherical to cartesian (in tangent space)
            vec3 tangentSample = vec3(sin(theta) * cos(phi),  sin(theta) * sin(phi), cos(theta));
            // tangent space to world
            vec3 sampleVec = tangentSample.x * right + tangentSample.y * up + tangentSample.z * N; 

            irradiance += texture(environmentMap, sampleVec).rgb * cos(theta) * sin(theta);
            nrSamples++;
        }
    }
    irradiance = PI * irradiance * (1.0 / float(nrSamples));
    
    FragColor = vec4(irradiance, 1.0);
}
)";
constexpr static const char *BRDFVS = R"(
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec2 aTexCoords;

out vec2 TexCoords;

void main()
{
    TexCoords = aTexCoords;
	gl_Position = vec4(aPos, 1.0);
}
)";
constexpr static const char *BRDFFS = R"(
#version 330 core
out vec2 FragColor;
in vec2 TexCoords;

const float PI = 3.14159265359;
// ----------------------------------------------------------------------------
// http://holger.dammertz.org/stuff/notes_HammersleyOnHemisphere.html
// efficient VanDerCorpus calculation.
float RadicalInverse_VdC(uint bits) 
{
     bits = (bits << 16u) | (bits >> 16u);
     bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
     bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
     bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
     bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
     return float(bits) * 2.3283064365386963e-10; // / 0x100000000
}
// ----------------------------------------------------------------------------
vec2 Hammersley(uint i, uint N)
{
	return vec2(float(i)/float(N), RadicalInverse_VdC(i));
}
// ----------------------------------------------------------------------------
vec3 ImportanceSampleGGX(vec2 Xi, vec3 N, float roughness)
{
	float a = roughness*roughness;
	
	float phi = 2.0 * PI * Xi.x;
	float cosTheta = sqrt((1.0 - Xi.y) / (1.0 + (a*a - 1.0) * Xi.y));
	float sinTheta = sqrt(1.0 - cosTheta*cosTheta);
	
	// from spherical coordinates to cartesian coordinates - halfway vector
	vec3 H;
	H.x = cos(phi) * sinTheta;
	H.y = sin(phi) * sinTheta;
	H.z = cosTheta;
	
	// from tangent-space H vector to world-space sample vector
	vec3 up          = abs(N.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
	vec3 tangent   = normalize(cross(up, N));
	vec3 bitangent = cross(N, tangent);
	
	vec3 sampleVec = tangent * H.x + bitangent * H.y + N * H.z;
	return normalize(sampleVec);
}
// ----------------------------------------------------------------------------
float GeometrySchlickGGX(float NdotV, float roughness)
{
    // note that we use a different k for IBL
    float a = roughness;
    float k = (a * a) / 2.0;

    float nom   = NdotV;
    float denom = NdotV * (1.0 - k) + k;

    return nom / denom;
}
// ----------------------------------------------------------------------------
float GeometrySmith(vec3 N, vec3 V, vec3 L, float roughness)
{
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    float ggx2 = GeometrySchlickGGX(NdotV, roughness);
    float ggx1 = GeometrySchlickGGX(NdotL, roughness);

    return ggx1 * ggx2;
}
// ----------------------------------------------------------------------------
vec2 IntegrateBRDF(float NdotV, float roughness)
{
    vec3 V;
    V.x = sqrt(1.0 - NdotV*NdotV);
    V.y = 0.0;
    V.z = NdotV;

    float A = 0.0;
    float B = 0.0; 

    vec3 N = vec3(0.0, 0.0, 1.0);
    
    const uint SAMPLE_COUNT = 2048u;
    for(uint i = 0u; i < SAMPLE_COUNT; ++i)
    {
        // generates a sample vector that's biased towards the
        // preferred alignment direction (importance sampling).
        vec2 Xi = Hammersley(i, SAMPLE_COUNT);
        vec3 H = ImportanceSampleGGX(Xi, N, roughness);
        vec3 L = normalize(2.0 * dot(V, H) * H - V);

        float NdotL = max(L.z, 0.0);
        float NdotH = max(H.z, 0.0);
        float VdotH = max(dot(V, H), 0.0);

        if(NdotL > 0.0)
        {
            float G = GeometrySmith(N, V, L, roughness);
            float G_Vis = (G * VdotH) / (NdotH * NdotV);
            float Fc = pow(1.0 - VdotH, 5.0);

            A += (1.0 - Fc) * G_Vis;
            B += Fc * G_Vis;
        }
    }
    A /= float(SAMPLE_COUNT);
    B /= float(SAMPLE_COUNT);
    return vec2(A, B);
}
// ----------------------------------------------------------------------------
void main() 
{
    vec2 integratedBRDF = IntegrateBRDF(TexCoords.x, TexCoords.y);
    FragColor = integratedBRDF;
}
)";
opengl::Program* preIntSpecularProg=nullptr;
opengl::Program* preintIrradianceProg=nullptr;
opengl::Program* preintBRDFProg=nullptr;
unsigned int captureFBO=0;
unsigned int captureRBO=0;
unsigned int envCubemap=0;
unsigned int irradianceMap=0;
unsigned int prefilterMap=0;    
unsigned int brdfLUTTexture=0;

unsigned int getIrradianceMap() {
    return irradianceMap;
}
unsigned int getPrefilterMap() {
    return prefilterMap;
}
unsigned int getBRDFLut() {
    return brdfLUTTexture;
}
unsigned int cubeVAO = 0;
unsigned int cubeVBO = 0;
void renderCube()
{
    // initialize (if necessary)
    if (cubeVAO == 0)
    {
        float vertices[] = {
            // back face
            -1.0f, -1.0f, -1.0f,  0.0f,  0.0f, -1.0f, 0.0f, 0.0f, // bottom-left
             1.0f,  1.0f, -1.0f,  0.0f,  0.0f, -1.0f, 1.0f, 1.0f, // top-right
             1.0f, -1.0f, -1.0f,  0.0f,  0.0f, -1.0f, 1.0f, 0.0f, // bottom-right         
             1.0f,  1.0f, -1.0f,  0.0f,  0.0f, -1.0f, 1.0f, 1.0f, // top-right
            -1.0f, -1.0f, -1.0f,  0.0f,  0.0f, -1.0f, 0.0f, 0.0f, // bottom-left
            -1.0f,  1.0f, -1.0f,  0.0f,  0.0f, -1.0f, 0.0f, 1.0f, // top-left
            // front face
            -1.0f, -1.0f,  1.0f,  0.0f,  0.0f,  1.0f, 0.0f, 0.0f, // bottom-left
             1.0f, -1.0f,  1.0f,  0.0f,  0.0f,  1.0f, 1.0f, 0.0f, // bottom-right
             1.0f,  1.0f,  1.0f,  0.0f,  0.0f,  1.0f, 1.0f, 1.0f, // top-right
             1.0f,  1.0f,  1.0f,  0.0f,  0.0f,  1.0f, 1.0f, 1.0f, // top-right
            -1.0f,  1.0f,  1.0f,  0.0f,  0.0f,  1.0f, 0.0f, 1.0f, // top-left
            -1.0f, -1.0f,  1.0f,  0.0f,  0.0f,  1.0f, 0.0f, 0.0f, // bottom-left
            // left face
            -1.0f,  1.0f,  1.0f, -1.0f,  0.0f,  0.0f, 1.0f, 0.0f, // top-right
            -1.0f,  1.0f, -1.0f, -1.0f,  0.0f,  0.0f, 1.0f, 1.0f, // top-left
            -1.0f, -1.0f, -1.0f, -1.0f,  0.0f,  0.0f, 0.0f, 1.0f, // bottom-left
            -1.0f, -1.0f, -1.0f, -1.0f,  0.0f,  0.0f, 0.0f, 1.0f, // bottom-left
            -1.0f, -1.0f,  1.0f, -1.0f,  0.0f,  0.0f, 0.0f, 0.0f, // bottom-right
            -1.0f,  1.0f,  1.0f, -1.0f,  0.0f,  0.0f, 1.0f, 0.0f, // top-right
            // right face
             1.0f,  1.0f,  1.0f,  1.0f,  0.0f,  0.0f, 1.0f, 0.0f, // top-left
             1.0f, -1.0f, -1.0f,  1.0f,  0.0f,  0.0f, 0.0f, 1.0f, // bottom-right
             1.0f,  1.0f, -1.0f,  1.0f,  0.0f,  0.0f, 1.0f, 1.0f, // top-right         
             1.0f, -1.0f, -1.0f,  1.0f,  0.0f,  0.0f, 0.0f, 1.0f, // bottom-right
             1.0f,  1.0f,  1.0f,  1.0f,  0.0f,  0.0f, 1.0f, 0.0f, // top-left
             1.0f, -1.0f,  1.0f,  1.0f,  0.0f,  0.0f, 0.0f, 0.0f, // bottom-left     
            // bottom face
            -1.0f, -1.0f, -1.0f,  0.0f, -1.0f,  0.0f, 0.0f, 1.0f, // top-right
             1.0f, -1.0f, -1.0f,  0.0f, -1.0f,  0.0f, 1.0f, 1.0f, // top-left
             1.0f, -1.0f,  1.0f,  0.0f, -1.0f,  0.0f, 1.0f, 0.0f, // bottom-left
             1.0f, -1.0f,  1.0f,  0.0f, -1.0f,  0.0f, 1.0f, 0.0f, // bottom-left
            -1.0f, -1.0f,  1.0f,  0.0f, -1.0f,  0.0f, 0.0f, 0.0f, // bottom-right
            -1.0f, -1.0f, -1.0f,  0.0f, -1.0f,  0.0f, 0.0f, 1.0f, // top-right
            // top face
            -1.0f,  1.0f, -1.0f,  0.0f,  1.0f,  0.0f, 0.0f, 1.0f, // top-left
             1.0f,  1.0f , 1.0f,  0.0f,  1.0f,  0.0f, 1.0f, 0.0f, // bottom-right
             1.0f,  1.0f, -1.0f,  0.0f,  1.0f,  0.0f, 1.0f, 1.0f, // top-right     
             1.0f,  1.0f,  1.0f,  0.0f,  1.0f,  0.0f, 1.0f, 0.0f, // bottom-right
            -1.0f,  1.0f, -1.0f,  0.0f,  1.0f,  0.0f, 0.0f, 1.0f, // top-left
            -1.0f,  1.0f,  1.0f,  0.0f,  1.0f,  0.0f, 0.0f, 0.0f  // bottom-left        
        };
        CHECK_GL(glGenVertexArrays(1, &cubeVAO));
        CHECK_GL(glGenBuffers(1, &cubeVBO));
        // fill buffer
        CHECK_GL(glBindBuffer(GL_ARRAY_BUFFER, cubeVBO));
        CHECK_GL(glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW));
        // link vertex attributes
        CHECK_GL(glBindVertexArray(cubeVAO));
        CHECK_GL(glEnableVertexAttribArray(0));
        CHECK_GL(glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)0));
        CHECK_GL(glEnableVertexAttribArray(1));
        CHECK_GL(glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(3 * sizeof(float))));
        CHECK_GL(glEnableVertexAttribArray(2));
        CHECK_GL(glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(6 * sizeof(float))));
        CHECK_GL(glBindBuffer(GL_ARRAY_BUFFER, 0));
        CHECK_GL(glBindVertexArray(0));
    }
    // render Cube
    CHECK_GL(glBindVertexArray(cubeVAO));
    CHECK_GL(glDrawArrays(GL_TRIANGLES, 0, 36));
    CHECK_GL(glBindVertexArray(0));
}
unsigned int quadVAO = 0;
unsigned int quadVBO;
void renderQuad()
{
    if (quadVAO == 0)
    {
        float quadVertices[] = {
            // positions        // texture Coords
            -1.0f,  1.0f, 0.0f, 0.0f, 1.0f,
            -1.0f, -1.0f, 0.0f, 0.0f, 0.0f,
             1.0f,  1.0f, 0.0f, 1.0f, 1.0f,
             1.0f, -1.0f, 0.0f, 1.0f, 0.0f,
        };
        // setup plane VAO
        CHECK_GL(glGenVertexArrays(1, &quadVAO));
        CHECK_GL(glGenBuffers(1, &quadVBO));
        CHECK_GL(glBindVertexArray(quadVAO));
        CHECK_GL(glBindBuffer(GL_ARRAY_BUFFER, quadVBO));
        CHECK_GL(glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), &quadVertices, GL_STATIC_DRAW));
        CHECK_GL(glEnableVertexAttribArray(0));
        CHECK_GL(glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0));
        CHECK_GL(glEnableVertexAttribArray(1));
        CHECK_GL(glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float))));
    }
    CHECK_GL(glBindVertexArray(quadVAO));
    CHECK_GL(glDrawArrays(GL_TRIANGLE_STRIP, 0, 4));
    CHECK_GL(glBindVertexArray(0));
}
void preIntegrate(GLuint inEnvMap)
{
    if (inEnvMap == (GLuint)-1) {
        irradianceMap=(GLuint)-1;
 prefilterMap=(GLuint)-1;    
 brdfLUTTexture=(GLuint)-1;
        return;
    }
        GLint oldFBO = 0;
        CHECK_GL(glGetIntegerv(GL_FRAMEBUFFER_BINDING, &oldFBO));
        GLint oldVAO = 0;
        CHECK_GL(glGetIntegerv(GL_VERTEX_ARRAY_BINDING, &oldVAO));
  CHECK_GL(glDisable(GL_BLEND));
  CHECK_GL(glDisable(GL_DEPTH_TEST));
  CHECK_GL(glDisable(GL_PROGRAM_POINT_SIZE));///????ZHXX???
  CHECK_GL(glDisable(GL_MULTISAMPLE));
  CHECK_GL(glEnable(GL_DEPTH_TEST));
  // set depth function to less than AND equal for skybox depth trick.
  CHECK_GL(glDepthFunc(GL_LEQUAL));
  // enable seamless cubemap sampling for lower mip levels in the pre-filter map.
  CHECK_GL(glEnable(GL_TEXTURE_CUBE_MAP_SEAMLESS));

  if(preintIrradianceProg==nullptr)
  {
    preintIrradianceProg = scene->shaderMan->compile_program(VS, IRFS);
  }
  if(preIntSpecularProg==nullptr)
  {
    preIntSpecularProg = scene->shaderMan->compile_program(VS, SPFS);
  }
  if(preintBRDFProg==nullptr)
  {
    preintBRDFProg = scene->shaderMan->compile_program(BRDFVS, BRDFFS);
  }
  if(captureFBO==0 && captureRBO==0){
    CHECK_GL(glGenFramebuffers(1, &captureFBO));
    CHECK_GL(glGenRenderbuffers(1, &captureRBO));

    CHECK_GL(glBindFramebuffer(GL_FRAMEBUFFER, captureFBO));
    CHECK_GL(glBindRenderbuffer(GL_RENDERBUFFER, captureRBO));
    CHECK_GL(glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, 1024, 1024));
    CHECK_GL(glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, captureRBO));
  }

  // pbr: set up projection and view matrices for capturing data onto the 6 cubemap face directions
  // ----------------------------------------------------------------------------------------------
  glm::mat4 captureProjection = glm::perspective(glm::radians(90.0f), 1.0f, 0.1f, 10.0f);
  glm::mat4 captureViews[] =
  {
      glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(1.0f,  0.0f,  0.0f), glm::vec3(0.0f, -1.0f,  0.0f)),
      glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(-1.0f,  0.0f,  0.0f), glm::vec3(0.0f, -1.0f,  0.0f)),
      glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f,  1.0f,  0.0f), glm::vec3(0.0f,  0.0f,  1.0f)),
      glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, -1.0f,  0.0f), glm::vec3(0.0f,  0.0f, -1.0f)),
      glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f,  0.0f,  1.0f), glm::vec3(0.0f, -1.0f,  0.0f)),
      glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f,  0.0f, -1.0f), glm::vec3(0.0f, -1.0f,  0.0f))
  };
  envCubemap = inEnvMap;
  CHECK_GL(glBindTexture(GL_TEXTURE_CUBE_MAP, envCubemap));
  CHECK_GL(glGenerateMipmap(GL_TEXTURE_CUBE_MAP));
/////////////////////////////////////////////////////////////////////////////////////////////////////
  if(irradianceMap==0){
    CHECK_GL(glGenTextures(1, &irradianceMap));
    CHECK_GL(glBindTexture(GL_TEXTURE_CUBE_MAP, irradianceMap));
    for (unsigned int i = 0; i < 6; ++i)
    {
        CHECK_GL(glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, GL_RGB16F, 128, 128, 0, GL_RGB, GL_FLOAT, nullptr));
    }
    CHECK_GL(glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE));
    CHECK_GL(glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE));
    CHECK_GL(glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE));
    CHECK_GL(glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR));
    CHECK_GL(glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR));
  }
  CHECK_GL(glBindFramebuffer(GL_FRAMEBUFFER, captureFBO));
  CHECK_GL(glBindRenderbuffer(GL_RENDERBUFFER, captureRBO));
  CHECK_GL(glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, 128, 128));

  // pbr: solve diffuse integral by convolution to create an irradiance (cube)map.
  // -----------------------------------------------------------------------------
  preintIrradianceProg->use();
  preintIrradianceProg->set_uniformi("environmentMap", 0);
  preintIrradianceProg->set_uniform("projection", captureProjection);
  CHECK_GL(glActiveTexture(GL_TEXTURE0));
  CHECK_GL(glBindTexture(GL_TEXTURE_CUBE_MAP, envCubemap));

  CHECK_GL(glViewport(0, 0, 128, 128)); // don't forget to configure the viewport to the capture dimensions.
  CHECK_GL(glBindFramebuffer(GL_FRAMEBUFFER, captureFBO));
  CHECK_GL(glClearColor(0.0f, 0.0f, 0.0f, 0.0f));
  CHECK_GL(glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT));

  for (unsigned int i = 0; i < 6; ++i)
  {
      preintIrradianceProg->set_uniform("view", captureViews[i]);
      CHECK_GL(glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, irradianceMap, 0));
      CHECK_GL(glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT));

      renderCube();
  }
  CHECK_GL(glBindFramebuffer(GL_FRAMEBUFFER, 0));
////////////////////////////////////////////////////////////////////////////////////////////
  if(prefilterMap==0){
    CHECK_GL(glGenTextures(1, &prefilterMap));
    CHECK_GL(glBindTexture(GL_TEXTURE_CUBE_MAP, prefilterMap));
    for (unsigned int i = 0; i < 6; ++i)
    {
        CHECK_GL(glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, GL_RGB16F, 1024, 1024, 0, GL_RGB, GL_FLOAT, nullptr));
    }
    CHECK_GL(glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE));
    CHECK_GL(glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE));
    CHECK_GL(glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE));
    CHECK_GL(glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)); // be sure to set minification filter to mip_linear 
    CHECK_GL(glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR));
    // generate mipmaps for the cubemap so OpenGL automatically allocates the required memory.
    CHECK_GL(glGenerateMipmap(GL_TEXTURE_CUBE_MAP));
  }

  // pbr: run a quasi monte-carlo simulation on the environment lighting to create a prefilter (cube)map.
  // ----------------------------------------------------------------------------------------------------
  preIntSpecularProg->use();
  preIntSpecularProg->set_uniformi("environmentMap", 0);
  preIntSpecularProg->set_uniform("projection", captureProjection);
  CHECK_GL(glActiveTexture(GL_TEXTURE0));
  CHECK_GL(glBindTexture(GL_TEXTURE_CUBE_MAP, envCubemap));

  CHECK_GL(glBindFramebuffer(GL_FRAMEBUFFER, captureFBO));
  CHECK_GL(glClearColor(0.0f, 0.0f, 0.0f, 0.0f));
  CHECK_GL(glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT));
  unsigned int maxMipLevels = 8;
  for (unsigned int mip = 0; mip < maxMipLevels; ++mip)
  {
      // reisze framebuffer according to mip-level size.
      unsigned int mipWidth  = static_cast<unsigned int>(1024 * std::pow(0.5, mip));
      unsigned int mipHeight = static_cast<unsigned int>(1024 * std::pow(0.5, mip));
      CHECK_GL(glBindRenderbuffer(GL_RENDERBUFFER, captureRBO));
      CHECK_GL(glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, mipWidth, mipHeight));
      CHECK_GL(glViewport(0, 0, mipWidth, mipHeight));

      float roughness = (float)mip / (float)(maxMipLevels - 1);
      preIntSpecularProg->set_uniform("roughness", roughness);
      for (unsigned int i = 0; i < 6; ++i)
      {
          preIntSpecularProg->set_uniform("view", captureViews[i]);
          CHECK_GL(glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, prefilterMap, mip));

          CHECK_GL(glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT));
          renderCube();
      }
  }
  CHECK_GL(glBindFramebuffer(GL_FRAMEBUFFER, 0));
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  if(brdfLUTTexture==0)
  {
    CHECK_GL(glGenTextures(1, &brdfLUTTexture));

    // pre-allocate enough memory for the LUT texture.
    CHECK_GL(glBindTexture(GL_TEXTURE_2D, brdfLUTTexture));
    CHECK_GL(glTexImage2D(GL_TEXTURE_2D, 0, GL_RG16F, 1024, 1024, 0, GL_RG, GL_FLOAT, 0));
    // be sure to set wrapping mode to GL_CLAMP_TO_EDGE
    CHECK_GL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE));
    CHECK_GL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE));
    CHECK_GL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR));
    CHECK_GL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR));
  }
  // then re-configure capture framebuffer object and render screen-space quad with BRDF shader.
  CHECK_GL(glBindTexture(GL_TEXTURE_2D, brdfLUTTexture));
  CHECK_GL(glBindFramebuffer(GL_FRAMEBUFFER, captureFBO));
  CHECK_GL(glBindRenderbuffer(GL_RENDERBUFFER, captureRBO));
  CHECK_GL(glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, 1024, 1024));
  CHECK_GL(glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, brdfLUTTexture, 0));

  CHECK_GL(glViewport(0, 0, 1024, 1024));
  CHECK_GL(glClearColor(0.0f, 0.0f, 0.0f, 0.0f));
  CHECK_GL(glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT));
  preintBRDFProg->use();
  CHECK_GL(glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT));
  renderQuad();

  CHECK_GL(glBindFramebuffer(GL_FRAMEBUFFER, oldFBO));
  CHECK_GL(glBindVertexArray(oldVAO));
  CHECK_GL(glUseProgram(0));
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
/* end zhxx happy */
~EnvmapIntegrator() { /* TODO: delete all the buffers */ }
};
}
