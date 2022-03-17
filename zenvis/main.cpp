#include "MyShader.hpp"
#include "glad/glad.h"
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
#include "zenvisapi.hpp"

namespace zenvis {


int curr_frameid = -1;

static int num_samples = 16;
static bool show_grid = true;
static bool smooth_shading = false;
static bool normal_check = false;
bool render_wireframe = false;

static int nx = 960, ny = 800;

static glm::vec3 bgcolor(0.23f, 0.23f, 0.23f);

static double last_xpos, last_ypos;
static glm::vec3 center;

static glm::mat4x4 view(1), proj(1);
static glm::mat4x4 gizmo_view(1), gizmo_proj(1);
static float point_scale = 1.f;
static float camera_radius = 1.f;
static float grid_scale = 1.f;
static float grid_blend = 0.f;

void set_perspective(
    std::array<double, 16> viewArr,
    std::array<double, 16> projArr)
{
  std::memcpy(glm::value_ptr(view), viewArr.data(), viewArr.size());
  std::memcpy(glm::value_ptr(proj), projArr.data(), projArr.size());
}

void look_perspective(
    double cx, double cy, double cz,
    double theta, double phi, double radius,
    double fov, bool ortho_mode) {
  center = glm::vec3(cx, cy, cz);

  point_scale = ny / (50.f * tanf(fov*0.5f*3.1415926f/180.0f));

  double cos_t = glm::cos(theta), sin_t = glm::sin(theta);
  double cos_p = glm::cos(phi), sin_p = glm::sin(phi);
  glm::vec3 back(cos_t * sin_p, sin_t, -cos_t * cos_p);
  glm::vec3 up(-sin_t * sin_p, cos_t, sin_t * cos_p);

  if (ortho_mode) {
    view = glm::lookAt(center - back, center, up);
    proj = glm::ortho(-radius * nx / ny, radius * nx / ny, -radius, radius,
                      -100.0, 100.0);
  } else {
    view = glm::lookAt(center - back * (float)radius, center, up);
    proj = glm::perspective(glm::radians(fov), nx * 1.0 / ny, 0.05, 20000.0 * std::max(1.0f, (float)radius / 10000.f));
  }
  camera_radius = radius;
  float level = std::fmax(std::log(radius) / std::log(5) - 1.0, -1);
  grid_scale = std::pow(5, std::floor(level));
  auto ratio_clamp = [](float value, float lower_bound, float upper_bound) {
      float ratio = (value - lower_bound) / (upper_bound - lower_bound);
      return fmin(fmax(ratio, 0.0), 1.0);
  };
  grid_blend = ratio_clamp(level - std::floor(level), 0.8, 1.0);
  center = glm::vec3(0, 0, 0);
  radius = 5.0;
  gizmo_view = glm::lookAt(center - back, center, up);
  gizmo_proj = glm::ortho(-radius * nx / ny, radius * nx / ny, -radius, radius,
                      -100.0, 100.0);
}

void set_program_uniforms(Program *pro) {
  pro->use();

  auto pers = proj * view;
  pro->set_uniform("mVP", pers);
  pro->set_uniform("mInvVP", glm::inverse(pers));
  pro->set_uniform("mView", view);
  pro->set_uniform("mProj", proj);
  pro->set_uniform("mInvView", glm::inverse(view));
  pro->set_uniform("mInvProj", glm::inverse(proj));
  pro->set_uniform("mPointScale", point_scale);
  pro->set_uniform("mSmoothShading", smooth_shading);
  pro->set_uniform("mNormalCheck", normal_check);
  pro->set_uniform("mCameraRadius", camera_radius);
  pro->set_uniform("mCameraCenter", center);
  pro->set_uniform("mGridScale", grid_scale);
  pro->set_uniform("mGridBlend", grid_blend);
}

static std::unique_ptr<VAO> vao;
static std::unique_ptr<IGraphic> grid;
static std::unique_ptr<IGraphic> axis;

std::unique_ptr<IGraphic> makeGraphicGrid();
std::unique_ptr<IGraphic> makeGraphicAxis();
extern void preIntegrate(GLuint inEnvMap);
void initialize() {
  gladLoadGL();
  
  
  auto version = (const char *)glGetString(GL_VERSION);
  printf("OpenGL version: %s\n", version ? version : "null");

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
  vao = std::make_unique<VAO>();
  grid = makeGraphicGrid();
  axis = makeGraphicAxis();
  setup_env_map("Default");
}

static void draw_small_axis() {
  glm::mat4x4 backup_view = view;
  glm::mat4x4 backup_proj = proj;
  view = gizmo_view;
  proj = gizmo_proj;
  CHECK_GL(glViewport(0, 0, nx * 0.1, ny * 0.1));
  CHECK_GL(glDisable(GL_DEPTH_TEST));
  axis->draw();
  CHECK_GL(glEnable(GL_DEPTH_TEST));
  CHECK_GL(glViewport(0, 0, nx, ny));
  view = backup_view;
  proj = backup_proj;
}




static void my_paint_graphics() {
  CHECK_GL(glViewport(0, 0, nx, ny));
  CHECK_GL(glClearColor(bgcolor.r, bgcolor.g, bgcolor.b, 0.0f));
  CHECK_GL(glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT));
  vao->bind();
  for (auto const &[key, gra]: current_frame_data()->graphics) {
    gra->draw();
  }
  if (show_grid) {
      axis->draw();
      grid->draw();
      draw_small_axis();
  }
  vao->unbind();
}


static bool enable_hdr = true;
/* BEGIN ZHXX HAPPY */
namespace {
auto qvert = R"(
#version 330 core
const vec2 quad_vertices[4] = vec2[4]( vec2( -1.0, -1.0), vec2( 1.0, -1.0), vec2( -1.0, 1.0), vec2( 1.0, 1.0));
void main()
{
    gl_Position = vec4(quad_vertices[gl_VertexID], 0.0, 1.0);
}
)";
auto qfrag = R"(#version 330 core
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
void main(void)
{
	int i;
	float lum[25];
	vec2 tex_scale = vec2(1.0);
	for (i = 0; i < 25; i++)
	{
		vec2 tc = (2.0 * gl_FragCoord.xy + 3.5 * vec2(i % 5 - 2, i / 5 - 2));
		vec3 col = texture2DRect(hdr_image, tc).rgb;
    lum[i] = dot(col, vec3(0.3, 0.59, 0.11));
	} 
	// Calculate weighted color of region
	float kernelLuminance = (
		(1.0 * (lum[0] + lum[4] + lum[20] + lum[24])) +
		(4.0 * (lum[1] + lum[3] + lum[5] + lum[9] +
		lum[15] + lum[19] + lum[21] + lum[23])) +
		(7.0 * (lum[2] + lum[10] + lum[14] + lum[22])) +
		(16.0 * (lum[6] + lum[8] + lum[16] + lum[18])) +
		(26.0 * (lum[7] + lum[11] + lum[13] + lum[17])) +
		(41.0 * lum[12])
	) / 273.0;
	// Compute the corresponding exposure
	float exposure = sqrt(8.0 / (kernelLuminance + 0.25));
	// Apply the exposure to this texel
  //oColor.rgb = 1.0 - exp2(-texture2DRect(hdr_image, gl_FragCoord.xy).rgb * exposure);
	//oColor.a = 1.0f;
	oColor = vec4(texture2DRect(hdr_image, gl_FragCoord.xy).rgb, 1.0);
  
}
)";
hg::OpenGL::Program* tmProg=nullptr;
GLuint msfborgb=0, msfbod=0, tonemapfbo=0;
int oldnx, oldny;
GLuint texRect=0, regularFBO = 0;
GLuint emptyVAO=0;
void ScreenFillQuad(GLuint tex)
{
  if(emptyVAO==0)
    glGenVertexArrays(1, &emptyVAO); 
  CHECK_GL(glViewport(0, 0, nx, ny));
  CHECK_GL(glClearColor(bgcolor.r, bgcolor.g, bgcolor.b, 0.0f));
  CHECK_GL(glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT));
  tmProg->use();
  tmProg->set_uniformi("hdr_image",0);
  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_RECTANGLE, tex);

  glEnableVertexAttribArray(0);
  glBindVertexArray(emptyVAO); 
  glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
  glDisableVertexAttribArray(0);
  glUseProgram(0);
}


auto VS = R"(
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
auto SPFS = R"(
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
auto IRFS = R"(
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
auto BRDFVS = R"(
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
auto BRDFFS = R"(
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
hg::OpenGL::Program* preIntSpecularProg=nullptr;
hg::OpenGL::Program* preintIrradianceProg=nullptr;
hg::OpenGL::Program* preintBRDFProg=nullptr;
unsigned int captureFBO=0;
unsigned int captureRBO=0;
unsigned int envCubemap=0;
unsigned int irradianceMap=0;
unsigned int prefilterMap=0;    
unsigned int brdfLUTTexture=0;
}
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
static void renderCube()
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
        glGenVertexArrays(1, &cubeVAO);
        glGenBuffers(1, &cubeVBO);
        // fill buffer
        glBindBuffer(GL_ARRAY_BUFFER, cubeVBO);
        glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
        // link vertex attributes
        glBindVertexArray(cubeVAO);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(3 * sizeof(float)));
        glEnableVertexAttribArray(2);
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(6 * sizeof(float)));
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glBindVertexArray(0);
    }
    // render Cube
    glBindVertexArray(cubeVAO);
    glDrawArrays(GL_TRIANGLES, 0, 36);
    glBindVertexArray(0);
}
unsigned int quadVAO = 0;
unsigned int quadVBO;
static void renderQuad()
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
        glGenVertexArrays(1, &quadVAO);
        glGenBuffers(1, &quadVBO);
        glBindVertexArray(quadVAO);
        glBindBuffer(GL_ARRAY_BUFFER, quadVBO);
        glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), &quadVertices, GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
    }
    glBindVertexArray(quadVAO);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    glBindVertexArray(0);
}
void preIntegrate(GLuint inEnvMap)
{
  CHECK_GL(glDisable(GL_BLEND));
  CHECK_GL(glDisable(GL_DEPTH_TEST));
  //CHECK_GL((GL_PROGRAM_POINT_SIZE));///????ZHXX???
  CHECK_GL(glDisable(GL_MULTISAMPLE));
  glEnable(GL_DEPTH_TEST);
  // set depth function to less than AND equal for skybox depth trick.
  glDepthFunc(GL_LEQUAL);
  // enable seamless cubemap sampling for lower mip levels in the pre-filter map.
  glEnable(GL_TEXTURE_CUBE_MAP_SEAMLESS);

  if(preintIrradianceProg==nullptr)
  {
    preintIrradianceProg = compile_program(VS, IRFS);
  }
  if(preIntSpecularProg==nullptr)
  {
    preIntSpecularProg = compile_program(VS, SPFS);
  }
  if(preintBRDFProg==nullptr)
  {
    preintBRDFProg = compile_program(BRDFVS, BRDFFS);
  }
  if(captureFBO==0 && captureRBO==0){
    glGenFramebuffers(1, &captureFBO);
    glGenRenderbuffers(1, &captureRBO);

    glBindFramebuffer(GL_FRAMEBUFFER, captureFBO);
    glBindRenderbuffer(GL_RENDERBUFFER, captureRBO);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, 1024, 1024);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, captureRBO);
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
  glBindTexture(GL_TEXTURE_CUBE_MAP, envCubemap);
  glGenerateMipmap(GL_TEXTURE_CUBE_MAP);
/////////////////////////////////////////////////////////////////////////////////////////////////////
  if(irradianceMap==0){
    glGenTextures(1, &irradianceMap);
    glBindTexture(GL_TEXTURE_CUBE_MAP, irradianceMap);
    for (unsigned int i = 0; i < 6; ++i)
    {
        glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, GL_RGB16F, 128, 128, 0, GL_RGB, GL_FLOAT, nullptr);
    }
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  }
  glBindFramebuffer(GL_FRAMEBUFFER, captureFBO);
  glBindRenderbuffer(GL_RENDERBUFFER, captureRBO);
  glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, 128, 128);

  // pbr: solve diffuse integral by convolution to create an irradiance (cube)map.
  // -----------------------------------------------------------------------------
  preintIrradianceProg->use();
  preintIrradianceProg->set_uniformi("environmentMap", 0);
  preintIrradianceProg->set_uniform("projection", captureProjection);
  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_CUBE_MAP, envCubemap);

  glViewport(0, 0, 128, 128); // don't forget to configure the viewport to the capture dimensions.
  glBindFramebuffer(GL_FRAMEBUFFER, captureFBO);
  CHECK_GL(glClearColor(bgcolor.r, bgcolor.g, bgcolor.b, 0.0f));
  CHECK_GL(glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT));

  for (unsigned int i = 0; i < 6; ++i)
  {
      preintIrradianceProg->set_uniform("view", captureViews[i]);
      glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, irradianceMap, 0);
      glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

      renderCube();
  }
  glBindFramebuffer(GL_FRAMEBUFFER, 0);
////////////////////////////////////////////////////////////////////////////////////////////
  if(prefilterMap==0){
    glGenTextures(1, &prefilterMap);
    glBindTexture(GL_TEXTURE_CUBE_MAP, prefilterMap);
    for (unsigned int i = 0; i < 6; ++i)
    {
        glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, GL_RGB16F, 1024, 1024, 0, GL_RGB, GL_FLOAT, nullptr);
    }
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR); // be sure to set minification filter to mip_linear 
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    // generate mipmaps for the cubemap so OpenGL automatically allocates the required memory.
    glGenerateMipmap(GL_TEXTURE_CUBE_MAP);
  }

  // pbr: run a quasi monte-carlo simulation on the environment lighting to create a prefilter (cube)map.
  // ----------------------------------------------------------------------------------------------------
  preIntSpecularProg->use();
  preIntSpecularProg->set_uniformi("environmentMap", 0);
  preIntSpecularProg->set_uniform("projection", captureProjection);
  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_CUBE_MAP, envCubemap);

  glBindFramebuffer(GL_FRAMEBUFFER, captureFBO);
  CHECK_GL(glClearColor(bgcolor.r, bgcolor.g, bgcolor.b, 0.0f));
  CHECK_GL(glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT));
  unsigned int maxMipLevels = 8;
  for (unsigned int mip = 0; mip < maxMipLevels; ++mip)
  {
      // reisze framebuffer according to mip-level size.
      unsigned int mipWidth  = static_cast<unsigned int>(1024 * std::pow(0.5, mip));
      unsigned int mipHeight = static_cast<unsigned int>(1024 * std::pow(0.5, mip));
      glBindRenderbuffer(GL_RENDERBUFFER, captureRBO);
      glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, mipWidth, mipHeight);
      glViewport(0, 0, mipWidth, mipHeight);

      float roughness = (float)mip / (float)(maxMipLevels - 1);
      preIntSpecularProg->set_uniform("roughness", roughness);
      for (unsigned int i = 0; i < 6; ++i)
      {
          preIntSpecularProg->set_uniform("view", captureViews[i]);
          glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, prefilterMap, mip);

          glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
          renderCube();
      }
  }
  glBindFramebuffer(GL_FRAMEBUFFER, 0);
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  if(brdfLUTTexture==0)
  {
    glGenTextures(1, &brdfLUTTexture);

    // pre-allocate enough memory for the LUT texture.
    glBindTexture(GL_TEXTURE_2D, brdfLUTTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RG16F, 1024, 1024, 0, GL_RG, GL_FLOAT, 0);
    // be sure to set wrapping mode to GL_CLAMP_TO_EDGE
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  }
  // then re-configure capture framebuffer object and render screen-space quad with BRDF shader.
  glBindTexture(GL_TEXTURE_2D, brdfLUTTexture);
  glBindFramebuffer(GL_FRAMEBUFFER, captureFBO);
  glBindRenderbuffer(GL_RENDERBUFFER, captureRBO);
  glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, 1024, 1024);
  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, brdfLUTTexture, 0);

  glViewport(0, 0, 1024, 1024);
  CHECK_GL(glClearColor(bgcolor.r, bgcolor.g, bgcolor.b, 0.0f));
  CHECK_GL(glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT));
  preintBRDFProg->use();
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  renderQuad();

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
static void paint_graphics(GLuint target_fbo = 0) {
  if(enable_hdr && tmProg==nullptr)
  {
    std::cout<<"compiling zhxx hdr program"<<std::endl;
    tmProg = compile_program(qvert, qfrag);
    if (!tmProg) {
    std::cout<<"failed to compile zhxx hdr program, giving up"<<std::endl;
        enable_hdr = false;
    }
  }
    if (!enable_hdr) {
        CHECK_GL(glBindFramebuffer(GL_DRAW_FRAMEBUFFER, target_fbo));
        return my_paint_graphics();
    }
  
  if(msfborgb==0||oldnx!=nx||oldny!=ny)
  {
    if(msfborgb!=0)
    {
      CHECK_GL(glDeleteRenderbuffers(1, &msfborgb));
    }
    CHECK_GL(glGenRenderbuffers(1, &msfborgb));
    CHECK_GL(glBindRenderbuffer(GL_RENDERBUFFER, msfborgb));
    CHECK_GL(glRenderbufferStorageMultisample(GL_RENDERBUFFER, num_samples, GL_RGBA32F, nx, ny));
    
  
    if(msfbod!=0)
    {
      CHECK_GL(glDeleteRenderbuffers(1, &msfbod));
    }
    CHECK_GL(glGenRenderbuffers(1, &msfbod));
    CHECK_GL(glBindRenderbuffer(GL_RENDERBUFFER, msfbod));
    CHECK_GL(glRenderbufferStorageMultisample(GL_RENDERBUFFER, num_samples, GL_DEPTH_COMPONENT24, nx, ny));

    
    if(tonemapfbo!=0)
    {
      CHECK_GL(glDeleteFramebuffers(1, &tonemapfbo));
    }
    CHECK_GL(glGenFramebuffers(1, &tonemapfbo));


    if(regularFBO!=0)
    {
      CHECK_GL(glDeleteFramebuffers(1, &regularFBO));
    }
    CHECK_GL(glGenFramebuffers(1, &regularFBO));
    if(texRect!=0)
    {
      CHECK_GL(glDeleteTextures(1, &texRect));
    }
    CHECK_GL(glGenTextures(1, &texRect));
    CHECK_GL(glBindTexture(GL_TEXTURE_RECTANGLE, texRect));
    {
        CHECK_GL(glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_MIN_FILTER, GL_NEAREST));
        CHECK_GL(glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_MAG_FILTER, GL_NEAREST));
        CHECK_GL(glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE));
        CHECK_GL(glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE));

        CHECK_GL(glTexImage2D(GL_TEXTURE_RECTANGLE, 0, GL_RGBA32F, nx, ny, 0, GL_RGBA, GL_FLOAT, nullptr));
    }
    CHECK_GL(glBindFramebuffer(GL_FRAMEBUFFER, regularFBO));
    CHECK_GL(glBindTexture(GL_TEXTURE_RECTANGLE, texRect));
    CHECK_GL(glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                    GL_TEXTURE_RECTANGLE, texRect, 0));
    CHECK_GL(glBindFramebuffer(GL_FRAMEBUFFER, 0));

    oldnx = nx;
    oldny = ny;
    
  }

    
  
  CHECK_GL(glBindFramebuffer(GL_DRAW_FRAMEBUFFER, tonemapfbo));
  CHECK_GL(glFramebufferRenderbuffer(GL_DRAW_FRAMEBUFFER,
                GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, msfborgb));
  CHECK_GL(glFramebufferRenderbuffer(GL_DRAW_FRAMEBUFFER,
                GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, msfbod));
  CHECK_GL(glDrawBuffer(GL_COLOR_ATTACHMENT0));


  my_paint_graphics();
  CHECK_GL(glBindFramebuffer(GL_READ_FRAMEBUFFER, tonemapfbo));
  CHECK_GL(glBindFramebuffer(GL_DRAW_FRAMEBUFFER, regularFBO));
  glBlitFramebuffer(0, 0, nx, ny, 0, 0, nx, ny, GL_COLOR_BUFFER_BIT, GL_NEAREST);

  //CHECK_GL(glBindFramebuffer(GL_READ_FRAMEBUFFER, regularFBO));
  CHECK_GL(glBindFramebuffer(GL_DRAW_FRAMEBUFFER, target_fbo));
  ScreenFillQuad(texRect);
  //glBlitFramebuffer(0, 0, nx, ny, 0, 0, nx, ny, GL_COLOR_BUFFER_BIT, GL_NEAREST);
  //drawScreenQuad here:
  CHECK_GL(glFlush());
}
/* END ZHXX HAPPY */

static double get_time() {
  static auto start = std::chrono::system_clock::now();
  auto now = std::chrono::system_clock::now();
  std::chrono::duration<double> diff = now - start;
  return diff.count();
}

static hg::FPSCounter solverFPS(get_time, 1);
static hg::FPSCounter renderFPS(get_time, 10);

void finalize() {
  vao = nullptr;
}

void new_frame() {
  paint_graphics();
  renderFPS.tick();
}

void set_window_size(int nx_, int ny_) {
  nx = nx_;
  ny = ny_;
}

void set_curr_frameid(int frameid) {
  curr_frameid = std::max(frameid, 0);
}

int get_curr_frameid() {
  return curr_frameid;
}

double get_render_fps() {
  return renderFPS.fps();
}

double get_solver_interval() {
  return solverFPS.interval();
}

void set_show_grid(bool flag) {
    show_grid = flag;
}

std::vector<char> record_frame_offline() {
    std::vector<char> pixels(nx * ny * 3);

    GLuint fbo, rbo1, rbo2;
    CHECK_GL(glGenRenderbuffers(1, &rbo1));
    CHECK_GL(glGenRenderbuffers(1, &rbo2));
    CHECK_GL(glBindRenderbuffer(GL_RENDERBUFFER, rbo1));
    CHECK_GL(glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA, nx, ny));
    CHECK_GL(glBindRenderbuffer(GL_RENDERBUFFER, rbo2));
    CHECK_GL(glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, nx, ny));

    CHECK_GL(glGenFramebuffers(1, &fbo));
    CHECK_GL(glBindFramebuffer(GL_DRAW_FRAMEBUFFER, fbo));
    CHECK_GL(glFramebufferRenderbuffer(GL_DRAW_FRAMEBUFFER,
                GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, rbo1));
    CHECK_GL(glFramebufferRenderbuffer(GL_DRAW_FRAMEBUFFER,
                GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, rbo2));
    CHECK_GL(glDrawBuffer(GL_COLOR_ATTACHMENT0));

    paint_graphics(fbo);

    if (glCheckFramebufferStatus(GL_DRAW_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE) {
        CHECK_GL(glBindFramebuffer(GL_READ_FRAMEBUFFER, fbo));
        CHECK_GL(glBlitFramebuffer(0, 0, nx, ny, 0, 0, nx, ny, GL_COLOR_BUFFER_BIT, GL_NEAREST));
        CHECK_GL(glBindFramebuffer(GL_READ_FRAMEBUFFER, fbo));
        CHECK_GL(glBindBuffer(GL_PIXEL_PACK_BUFFER, 0));
        CHECK_GL(glReadBuffer(GL_COLOR_ATTACHMENT0));

        CHECK_GL(glReadPixels(0, 0, nx, ny, GL_RGB, GL_UNSIGNED_BYTE, &pixels[0]));
    }

    CHECK_GL(glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0));
    CHECK_GL(glBindRenderbuffer(GL_RENDERBUFFER, 0));

    CHECK_GL(glDeleteRenderbuffers(1, &rbo1));
    CHECK_GL(glDeleteRenderbuffers(1, &rbo2));
    CHECK_GL(glDeleteFramebuffers(1, &fbo));

    return pixels;
}

void new_frame_offline(std::string path) {
    char buf[1024];
    sprintf(buf, "%s/%06d.png", path.c_str(), curr_frameid);
    printf("saving screen %dx%d to %s\n", nx, ny, buf);

    std::vector<char> pixels = record_frame_offline();
    stbi_flip_vertically_on_write(true);
    stbi_write_png(buf, nx, ny, 3, &pixels[0], 0);
}

void do_screenshot(std::string path) {
    std::vector<char> pixels = record_frame_offline();
    stbi_flip_vertically_on_write(true);
    stbi_write_png(path.c_str(), nx, ny, 3, &pixels[0], 0);
}

void set_background_color(float r, float g, float b) {
    bgcolor = glm::vec3(r, g, b);
}

std::tuple<float, float, float> get_background_color() {
    return {bgcolor.r, bgcolor.g, bgcolor.b};
}

void set_smooth_shading(bool smooth) {
    smooth_shading = smooth;
}
void set_normal_check(bool check) {
    normal_check = check;
}

void set_render_wireframe(bool render_wireframe_) {
    render_wireframe = render_wireframe_;
}

void set_num_samples(int num_samples_) {
    num_samples = num_samples_;
}
}
