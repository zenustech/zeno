#include "glad/glad.h"
#include "stdafx.hpp"
#include "IGraphic.hpp"
#include "MyShader.hpp"
#include "main.hpp"
#include <memory>
#include <string>
#include <vector>
#include <zeno/utils/vec.h>
#include <zeno/utils/ticktock.h>
#include <zeno/utils/orthonormal.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/PrimitiveTools.h>
#include <zeno/types/MaterialObject.h>
#include <zeno/types/TextureObject.h>
#include <zeno/types/InstancingObject.h>
#include <Hg/IOUtils.h>
#include <Hg/IterUtils.h>
#include <Scene.hpp>

namespace zenvis {
    Program * get_voxelize_program(std::shared_ptr<zeno::MaterialObject> mtl, std::shared_ptr<zeno::InstancingObject> inst)
  {
std::string VXVS;
    if (inst != nullptr)
    {
VXVS = R"(
#version 430 core

uniform mat4 mVP;
uniform mat4 mInvVP;
uniform mat4 mView;
uniform mat4 mProj;
uniform mat4 mInvView;
uniform mat4 mInvProj;
uniform vec3 u_scene_voxel_scale;

in vec3 vPosition;
in vec3 vColor;
in vec3 vNormal;
in vec3 vTexCoord;
in vec3 vTangent;
in mat4 mInstModel;

out vec3 g_world_pos;
out vec3 g_normal;
out vec3 g_color;
out vec3 g_tex_coords;
out vec3 g_tangent;

void main()
{
  g_world_pos = vec3(mInstModel * vec4(vPosition, 1.0));
  g_color = vColor;
  g_normal = transpose(inverse(mat3(mInstModel))) * vNormal;
  g_tex_coords = vTexCoord;
  g_tangent = mat3(mInstModel) * vTangent;
  gl_Position = vec4(vec3(mView * vec4(g_world_pos, 1.0)) * u_scene_voxel_scale, 1.0);
}
)";
    }
    else
  {
VXVS = R"(
#version 430 core

uniform mat4 mVP;
uniform mat4 mInvVP;
uniform mat4 mView;
uniform mat4 mProj;
uniform mat4 mInvView;
uniform mat4 mInvProj;
uniform vec3 u_scene_voxel_scale;

in vec3 vPosition;
in vec3 vColor;
in vec3 vNormal;
in vec3 vTexCoord;
in vec3 vTangent;
in mat4 mInstModel;

out vec3 g_world_pos;
out vec3 g_normal;
out vec3 g_color;
out vec3 g_tex_coords;
out vec3 g_tangent;

void main()
{
  g_world_pos = vPosition;
  g_color = vColor;
  g_normal = vNormal;
  g_tex_coords = vTexCoord;
  g_tangent = vTangent;
  gl_Position = vec4(vec3(mView * vec4(g_world_pos, 1.0)) * u_scene_voxel_scale, 1.0);
}
)";
    }

auto VXFS = "#version 430 core\n#extension GL_ARB_shader_image_load_store : require\n/* common_funcs_begin */\n" + mtl->common + "\n/* common_funcs_end */\n"+R"(

const float PI = 3.14159265358979323846;

uniform mat4 mVP;
uniform mat4 mInvVP;
uniform mat4 mView;
uniform mat4 mProj;
uniform mat4 mInvView;
uniform mat4 mInvProj;
uniform bool mSmoothShading;
uniform bool mNormalCheck;
uniform bool mRenderWireframe;


in vec3 position;
in vec3 iColor;
in vec3 iNormal;
in vec3 iTexCoord;
in vec3 iTangent;
out vec4 fColor;
const float minDot = 1e-5;

// Clamped dot product
float dot_c(vec3 a, vec3 b){
	return max(dot(a, b), minDot);
}

// Get orthonormal basis from surface normal
// https://graphics.pixar.com/library/OrthonormalB/paper.pdf
void pixarONB(vec3 n, out vec3 b1, out vec3 b2){
	vec3 up        = abs(n.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
    b1   = normalize(cross(up, n));
    b2 = cross(n, b1);
}

vec3 CalculateDiffuse(
    in vec3 albedo){                              
    return (albedo / 3.1415926);
}
layout(binding = 0, r32ui) uniform volatile coherent uimage3D u_tex_voxelgrid;
vec4 convRGBA8ToVec4(uint val){
    return vec4(float((val & 0x000000FF)), float ((val & 0x0000FF00) >> 8U), float (( val & 0x00FF0000) >> 16U), float ((val & 0xFF000000) >> 24U));
}

uint convVec4ToRGBA8(vec4 val){
    return (uint(val.w) & 0x000000FF) << 24U | (uint(val.z) &0x000000FF) << 16U | (uint(val.y) & 0x000000FF) << 8U | (uint(val.x) & 0x000000FF);
}

void imageAtomicRGBA8Avg(layout(r32ui) coherent volatile uimage3D img, ivec3 coords, vec4 val){
    val.rgb *= 255.0f;
    uint newVal = convVec4ToRGBA8(val);
    uint prevStoredVal = 0;
    uint curStoredVal;
    uint numIterations = 0;
    while ((curStoredVal = imageAtomicCompSwap(img, coords, prevStoredVal, newVal)) != prevStoredVal 
            && numIterations < 255) {
        prevStoredVal = curStoredVal;
        vec4 rval = convRGBA8ToVec4(curStoredVal);
        rval.xyz = (rval.xyz * rval.w);
        vec4 curValF = rval + val;
        curValF.xyz /= (curValF.w);
        newVal = convVec4ToRGBA8(curValF);
        ++numIterations;
    }
}
vec3 calcRayDir(vec3 pos)
{
  vec4 vpos = mView * vec4(pos, 1);
  return normalize(vpos.xyz);
}
vec3 fresnelSchlickRoughness(float cosTheta, vec3 F0, float roughness){
    return F0 + (max(vec3(1.0-roughness), F0) - F0) * pow(1.0 - cosTheta, 5.0);
}
uniform samplerCube irradianceMap;
uniform samplerCube prefilterMap;
uniform sampler2D brdfLUT;
vec3 CalculateLightingIBL(
    in vec3  N,
    in vec3  V,
    in vec3  albedo,
    in float roughness,
    in float metallic)
{
    mat3 m = inverse(mat3(mView[0].xyz, mView[1].xyz, mView[2].xyz));
    vec3 F0 = mix(vec3(0.04), albedo, metallic);
    vec3 F = fresnelSchlickRoughness(dot_c(N, V), F0, roughness);
    vec3 kS = F;
    vec3 kD = 1.0 - kS;
    kD *= 1.0 - metallic;
    const float MAX_REFLECTION_LOD = 7.0;
    vec3 irradiance = textureLod(prefilterMap, m*N,  MAX_REFLECTION_LOD).rgb;
    vec3 diffuse      = irradiance * CalculateDiffuse(albedo);
    vec3 R = reflect(-V, N); 
    vec3 prefilteredColor = textureLod(prefilterMap, m*R,  roughness * MAX_REFLECTION_LOD).rgb;    
    vec2 brdf  = texture(brdfLUT, vec2(max(dot(N, V), 0.0), roughness)).rg;
    vec3 specular = prefilteredColor * (F * brdf.x + brdf.y);

    return (kD * diffuse + specular);

}
float sqr(float x) { return x*x; }

float SchlickFresnel(float u)
{
    float m = clamp(1-u, 0, 1);
    float m2 = m*m;
    return m2*m2*m; // pow(m,5)
}

float GTR1(float NdotH, float a)
{
    if (a >= 1) return 1/PI;
    float a2 = a*a;
    float t = 1 + (a2-1)*NdotH*NdotH;
    return (a2-1) / (PI*log(a2)*t);
}

float GTR2(float NdotH, float a)
{
    float a2 = a*a;
    float t = 1 + (a2-1)*NdotH*NdotH;
    return a2 / (PI * t*t);
}

float GTR2_aniso(float NdotH, float HdotX, float HdotY, float ax, float ay)
{
    return 1 / (PI * ax*ay * sqr( sqr(HdotX/ax) + sqr(HdotY/ay) + NdotH*NdotH ));
}

float smithG_GGX(float NdotV, float alphaG)
{
    float a = alphaG*alphaG;
    float b = NdotV*NdotV;
    return 1 / (NdotV + sqrt(a + b - a*b));
}

float smithG_GGX_aniso(float NdotV, float VdotX, float VdotY, float ax, float ay)
{
    return 1 / (NdotV + sqrt( sqr(VdotX*ax) + sqr(VdotY*ay) + sqr(NdotV) ));
}

vec3 mon2lin(vec3 x)
{
    return vec3(pow(x[0], 2.2), pow(x[1], 2.2), pow(x[2], 2.2));
}
vec3 BRDF(vec3 baseColor, float metallic, float subsurface, 
float specular, 
float roughness,
float specularTint,
float anisotropic,
float sheen,
float sheenTint,
float clearcoat,
float clearcoatGloss,
vec3 L, vec3 V, vec3 N, vec3 X, vec3 Y)
{
    float NdotL = dot(N,L);
    float NdotV = dot(N,V);
    //if (NdotL < 0 || NdotV < 0) return vec3(0);

    vec3 H = normalize(L+V);
    float NdotH = dot(N,H);
    float LdotH = dot(L,H);

    vec3 Cdlin = mon2lin(baseColor);
    float Cdlum = .3*Cdlin[0] + .6*Cdlin[1]  + .1*Cdlin[2]; // luminance approx.

    vec3 Ctint = Cdlum > 0 ? Cdlin/Cdlum : vec3(1); // normalize lum. to isolate hue+sat
    vec3 Cspec0 = mix(specular*.08*mix(vec3(1), Ctint, specularTint), Cdlin, metallic);
    vec3 Csheen = mix(vec3(1), Ctint, sheenTint);

    // Diffuse fresnel - go from 1 at normal incidence to .5 at grazing
    // and mix in diffuse retro-reflection based on roughness
    float FL = clamp(SchlickFresnel(NdotL),0,1);
    float FV = clamp(SchlickFresnel(NdotV),0,1);
    float Fd90 = 0.5 + 2 * LdotH*LdotH * roughness;
    float Fd = mix(1.0, Fd90, FL) * mix(1.0, Fd90, FV);

    // Based on Hanrahan-Krueger brdf approximation of isotropic bssrdf
    // 1.25 scale is used to (roughly) preserve albedo
    // Fss90 used to "flatten" retroreflection based on roughness
    float Fss90 = LdotH*LdotH*roughness;
    float Fss = mix(1.0, Fss90, FL) * mix(1.0, Fss90, FV);
    float NDLV = (NdotL + NdotV)>0?clamp((NdotL + NdotV),0.0001, 2.0):clamp((NdotL + NdotV), -2.0, -0.0001);
    float ss = 1.25 * (Fss * (1 /NDLV  - .5) + .5);

    // specular
    float aspect = sqrt(1-anisotropic*.9);
    float ax = max(.001, sqr(roughness)/aspect);
    float ay = max(.001, sqr(roughness)*aspect);
    float Ds = GTR2_aniso(NdotH, dot(H, X), dot(H, Y), ax, ay);
    float FH = SchlickFresnel(LdotH);
    
    vec3 Fs = mix(Cspec0, vec3(1), FH);
    float Gs;
    Gs  = smithG_GGX_aniso(NdotL, dot(L, X), dot(L, Y), ax, ay);
    Gs *= smithG_GGX_aniso(NdotV, dot(V, X), dot(V, Y), ax, ay);

    // sheen
    vec3 Fsheen = FH * sheen * Csheen;

    // clearcoat (ior = 1.5 -> F0 = 0.04)
    float Dr = GTR1(NdotH, mix(.1,.001,clearcoatGloss));
    float Fr = mix(.04, 1.0, FH);
    float Gr = smithG_GGX(NdotL, .25) * smithG_GGX(NdotV, .25);
    float angle = clamp(dot(N, L), 0.0, 1.0);
    float c1 = (1/PI) * mix(Fd, ss, subsurface);
    
    return ((c1 * Cdlin + Fsheen)
        * (1-metallic)
        + Gs*Fs*Ds + .25*clearcoat*Gr*Fr*Dr)*angle;
}
uniform int lightNum; 
uniform vec3 light[16];
uniform sampler2D shadowMap[128];
uniform vec3 lightIntensity[16];
uniform vec3 shadowTint[16];
uniform float shadowSoftness[16];
uniform vec3 lightDir[16];
uniform float farPlane;
uniform mat4 lview[16];
uniform float near[128];
uniform float far[128];
uniform mat4 lightSpaceMatrices[128];
uniform float cascadePlaneDistances[112];
uniform int cascadeCount;   // number of frusta - 1
vec3 random3(vec3 c) {
	float j = 4096.0*sin(dot(c,vec3(17.0, 59.4, 15.0)));
	vec3 r;
	r.z = fract(512.0*j);
	j *= .125;
	r.x = fract(512.0*j);
	j *= .125;
	r.y = fract(512.0*j);
	return r-0.5;
}
float sampleShadowArray(int lightNo, vec2 coord, int layer)
{
    vec4 res;
    
    res = texture(shadowMap[lightNo * (cascadeCount + 1) + layer], clamp(coord,vec2(0.00), vec2(1)));

    return res.r;    
}
float PCFLayer(int lightNo, float currentDepth, float bias, vec3 pos, int layer, int k, float softness, vec2 coord)
{
    float shadow = 0.0;
    float near1, far1;
    near1 = near[lightNo * 8 + layer];
    far1 = far[lightNo   * 8 + layer];
    vec2 texelSize = 1.0 / vec2(textureSize(shadowMap[lightNo * (cascadeCount + 1) + 0], 0));
    for(int x = -k; x <= k; ++x)
    {
        for(int y = -k; y <= k; ++y)
        {
            vec3 noise = random3(pos+vec3(x, y,0)*0.01*softness);
            float pcfDepth = sampleShadowArray(lightNo, coord + (vec2(x, y) * softness + noise.xy) * texelSize, layer)  * (far1-near1) + near1 ; 
            shadow += (currentDepth  * (far1-near1) + near1  - bias) > pcfDepth  ? 1.0 : 0.0;        
        }    
    }
    float size = 2.0*float(k)+1.0;
    return shadow /= (size*size);
}
float PCFLayer2(int lightNo, float currentDepth1, float currentDepth2, float bias, vec3 pos, int layer, int k, float softness, vec2 coord1, vec2 coord2)
{
    float shadow = 0.0;
    float near1, far1, near2, far2;
    near1 = near[lightNo * 8 + layer];
    far1 = far[lightNo   * 8 + layer];
    near2 = near[lightNo * 8 + layer + 1];
    far2 = far[lightNo   * 8 + layer + 1];
    vec2 texelSize = 1.0 / vec2(textureSize(shadowMap[lightNo * (cascadeCount + 1) + 0], 0));
    for(int x = -k; x <= k; ++x)
    {
        for(int y = -k; y <= k; ++y)
        {
            vec3 noise = random3(pos+vec3(x, y,0)*0.01*softness);
            float pcfDepth1 = sampleShadowArray(lightNo, coord1 + (vec2(x, y) * softness + noise.xy) * texelSize, layer) * (far1-near1) + near1;
            
            float pcfDepth2 = sampleShadowArray(lightNo, coord2 + (vec2(x, y) * softness + noise.xy) * texelSize, layer+1) * (far2-near2) + near2; 
            float s1 = ((currentDepth1 * (far1-near1) + near1 - bias) > pcfDepth1)?1.0 : 0.0;
            float s2 = ((currentDepth2 * (far2-near2) + near2 - bias) > pcfDepth2)?1.0 : 0.0;
            shadow += mix(s1, s2, 0.5);        
        }    
    }
    float size = 2.0*float(k)+1.0;
    return shadow /= (size*size);
}
float ShadowCalculation(int lightNo, vec3 fragPosWorldSpace, float softness)
{
    // select cascade layer
    vec4 fragPosViewSpace = mView * vec4(fragPosWorldSpace, 1.0);
    float depthValue = abs(fragPosViewSpace.z);

    int layer = -1;
    for (int i = 0; i < cascadeCount; ++i)
    {
        vec4 fragPosLightSpace = lightSpaceMatrices[lightNo * (cascadeCount + 1) + i] * vec4(fragPosWorldSpace, 1.0);
        // perform perspective divide
        vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;
        // transform to [0,1] range
        projCoords = projCoords * 0.5 + 0.5;
        if (projCoords.x>=0&&projCoords.x<=1&&projCoords.y>=0&&projCoords.y<=1)
        {
            layer = i;
            break;
        }
    }
    if (layer == -1)
    {
        layer = cascadeCount;
    }

    vec4 fragPosLightSpace = lightSpaceMatrices[lightNo * (cascadeCount + 1) + layer] * vec4(fragPosWorldSpace, 1.0);
    // perform perspective divide
    vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;
    // transform to [0,1] range
    projCoords = projCoords * 0.5 + 0.5;

    // get depth of current fragment from light's perspective
    float currentDepth = projCoords.z;

    // keep the shadow at 0.0 when outside the far_plane region of the light's frustum.
    if (currentDepth > 1.0)
    {
        return 0.0;
    }
    // calculate bias (based on depth map resolution and slope)
    vec3 normal = normalize(iNormal);
    float slop = abs(dot( normalize(normal), normalize(light[lightNo])));
    float bias = (1-pow(slop,0.1)) * 0.2 + pow(slop,0.1) * 0.002;
    //bias *= 1 / (far[lightNo * (cascadeCount + 1) + layer] * 0.5f);
    

    

    // PCF
    //float shadow1 = PCFLayer(lightNo, currentDepth, bias, fragPosWorldSpace, layer, 3, softness, projCoords.xy);
    //float shadow2 = shadow1;
    //float coef = 0.0;
    if(layer<cascadeCount){
        //bm = 1 / (cascadePlaneDistances[lightNo * cascadeCount + (layer-1)] * biasModifier);
        fragPosLightSpace = lightSpaceMatrices[lightNo * (cascadeCount + 1) + (layer+1)] * vec4(fragPosWorldSpace, 1.0);
        vec3 projCoords2 = fragPosLightSpace.xyz / fragPosLightSpace.w;
        projCoords2 = projCoords2 * 0.5 + 0.5;
        return PCFLayer2(lightNo, currentDepth, projCoords2.z, bias, fragPosWorldSpace, layer, 3, softness, projCoords.xy, projCoords2.xy);
        // vec2 d = abs(projCoords.xy-vec2(0.5));
        // float sdf = max(d.x, d.y)-0.5;
        // float coef = smoothstep(0,0.05,sdf + 0.05);
    }
    else
    {
        return PCFLayer(lightNo, currentDepth, bias, fragPosWorldSpace, layer, 3, softness, projCoords.xy);
    }
    
        
}
vec2 getLightCoord(int lightNo, vec3 fragPosWorldSpace)
{
    
    vec4 fragPosLightSpace = lightSpaceMatrices[lightNo * (cascadeCount + 1) + 0] * vec4(fragPosWorldSpace, 1.0);
    // perform perspective divide
    vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;
    // transform to [0,1] range
    return vec2(projCoords * 0.5 + 0.5);
}
float ShadowHit(int lightNo, vec3 fragPosWorldSpace)
{
    vec4 fragPosViewSpace = mView * vec4(fragPosWorldSpace, 1.0);
    float depthValue = abs(fragPosViewSpace.z);

    int layer = -1;
    for (int i = 0; i < cascadeCount; ++i)
    {
        vec4 fragPosLightSpace = lightSpaceMatrices[lightNo * (cascadeCount + 1) + i] * vec4(fragPosWorldSpace, 1.0);
        // perform perspective divide
        vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;
        // transform to [0,1] range
        projCoords = projCoords * 0.5 + 0.5;
        if (projCoords.x>=0&&projCoords.x<=1&&projCoords.y>=0&&projCoords.y<=1)
        {
            layer = i;
            break;
        }
    }
    if (layer == -1)
    {
        layer = cascadeCount;
    }

    vec4 fragPosLightSpace = lightSpaceMatrices[lightNo * (cascadeCount + 1) + layer] * vec4(fragPosWorldSpace, 1.0);
    // perform perspective divide
    vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;
    // transform to [0,1] range
    projCoords = projCoords * 0.5 + 0.5;
    float near1, far1, near2, far2;
    near1 = near[lightNo * 8 + layer];
    far1 = far[lightNo   * 8 + layer];
    // get depth of current fragment from light's perspective
    float currentDepth = projCoords.z;
    vec3 normal = normalize(iNormal);
    float slop = abs(dot( normalize(normal), normalize(light[lightNo])));
    float bias = (1-pow(slop,0.1)) * 0.1 + pow(slop,0.1) * 0.001;
    return (currentDepth  * (far1-near1) + near1 - bias) > (sampleShadowArray(lightNo, projCoords.xy, layer)  * (far1-near1) + near1)?1.0:0.0;
}
float ShadowCalculation(int lightNo, vec3 fragPosWorldSpace, float softness, vec3 tang, vec3 bitang, int k)
{
    float shadow = 0.0;
    vec2 coord1 = getLightCoord(lightNo, fragPosWorldSpace - 0.001 * tang - 0.001 * bitang);
    vec2 coord2 = getLightCoord(lightNo, fragPosWorldSpace + 0.001 * tang + 0.001 * bitang);
    
    vec2 texelSize = 1.0 / vec2(textureSize(shadowMap[lightNo * (cascadeCount + 1) + 0], 0));


    softness = softness / (length(coord2-coord1) / texelSize.x) * 2;
    for(int x = -k; x <= k; ++x)
    {
        for(int y = -k; y <= k; ++y)
        {
            vec3 noise = random3(fragPosWorldSpace+vec3(x, y,0))*0.001*softness;
            vec2 pvec = noise.xy + vec2(x,y) * 0.001 * softness;
            vec3 ppos = fragPosWorldSpace + pvec.x * tang + pvec.y * bitang;
            shadow += ShadowHit(lightNo, ppos);
        }    
    }
    float size = 2.0*float(k)+1.0;
    return shadow /= (size*size); 
}

vec4 studioShading(vec3 albedo, vec3 view_dir, vec3 normal, vec3 old_tangent) {
    vec4 projPos = mView * vec4(position.xyz, 1.0);
    //normal = normalize(normal);
    vec3 L1 = light[0];
    vec3 att_pos = position;
    vec3 att_clr = iColor;
    vec3 att_nrm = normal;
    vec3 att_uv = iTexCoord;
    vec3 att_tang = old_tangent;
    float att_NoL = dot(normal, L1);
    /* custom_shader_begin */
)" + mtl->frag + R"(
    if(mat_opacity>=0.99)
        discard; 
    vec3 colorEmission = mat_emission;
    mat_metallic = clamp(mat_metallic, 0, 1);
    vec3 new_normal = normal; /* TODO: use mat_normal to transform this */
    vec3 color = vec3(0,0,0);
    vec3 light_dir;
    vec3 albedo2 = mat_basecolor;
    float roughness = mat_roughness;
    vec3 tan = normalize(old_tangent - dot(normal, old_tangent)*normal);
    mat3 TBN = mat3(tan, cross(normal, tan), normal);

    new_normal = TBN * normalize(mat_normal);
    
    color = vec3(0,0,0);
    vec3 realColor = vec3(0,0,0);
    float lightsNo = 0;
    for(int lightId=0; lightId<lightNum; lightId++){
        vec3 photoReal = BRDF(mat_basecolor, mat_metallic,mat_subsurface,mat_specular,mat_roughness,mat_specularTint,mat_anisotropic,mat_sheen,mat_sheenTint,mat_clearcoat,mat_clearcoatGloss,normalize(light_dir), normalize(light_dir), normalize(new_normal),normalize(tan), normalize(cross(normal, tan))
        ) * lightIntensity[lightId];
        vec3 lcolor = photoReal;
        float shadow = ShadowCalculation(lightId, position, shadowSoftness[lightId], tan, TBN[1],3);
        vec3 sclr = clamp(vec3(1.0-shadow) + shadowTint[lightId], vec3(0), vec3(1));
        color += lcolor * sclr;
    }
    vec3 iblPhotoReal =  CalculateLightingIBL(new_normal,new_normal,albedo2,roughness,mat_metallic);
    return vec4(color + colorEmission + 0.1 * iblPhotoReal, 1.0 - mat_opacity);


}

void main()
{   
  vec3 normal;
  normal = normalize(iNormal);
  
  vec3 viewdir = -calcRayDir(position);
  vec3 albedo = iColor;
  vec3 normalInView = transpose(inverse(mat3(mView[0].xyz, mView[1].xyz, mView[2].xyz)))*normal;
  if(dot(-viewdir, normalInView)>0)
    normal = - normal;

  //normal = faceforward(normal, -viewdir, normal);
  vec3 tangent = iTangent;
  if (tangent == vec3(0)) {
   vec3 unusedbitan;
   pixarONB(normal, tangent, unusedbitan);
  }

  vec4 color = studioShading(albedo, viewdir, normal, tangent);
  fColor = color;
}
)";

auto VXGS = R"(
#version 430 core
layout (triangles) in;
layout (triangle_strip, max_vertices=3) out;
uniform mat4 mVP;
uniform mat4 mInvVP;
uniform mat4 mView;
uniform mat4 mProj;
uniform mat4 mInvView;
uniform mat4 mInvProj;
uniform bool mSmoothShading;
uniform bool mNormalCheck;
uniform bool mRenderWireframe;
in vec3 g_world_pos[];
in vec3 g_normal[];
in vec3 g_color[];
in vec3 g_tex_coords[];
in vec3 g_tangent[];

out vec3 position;
out vec3 iColor;
out vec3 iNormal;
out vec3 iTexCoord;
out vec3 iTangent;
out vec3 f_voxel_pos;

void main()
{          
	mat3 swizzle_mat;

	const vec3 edge1 = gl_in[1].gl_Position.xyz - gl_in[0].gl_Position.xyz;
	const vec3 edge2 = gl_in[2].gl_Position.xyz - gl_in[0].gl_Position.xyz;
	const vec3 face_normal = abs(cross(edge1, edge2)); 

	if (face_normal.x >= face_normal.y && face_normal.x >= face_normal.z) { // see: Introduction to Geometric Computing, page 33 (Ghali, 2008)
		swizzle_mat = mat3(
			vec3(0.0, 0.0, 1.0),
			vec3(0.0, 1.0, 0.0),
			vec3(1.0, 0.0, 0.0));
	} else if (face_normal.y >= face_normal.z) {
		swizzle_mat = mat3(
			vec3(1.0, 0.0, 0.0),
			vec3(0.0, 0.0, 1.0),
			vec3(0.0, 1.0, 0.0));
	} else {
		swizzle_mat = mat3(
			vec3(1.0, 0.0, 0.0),
			vec3(0.0, 1.0, 0.0),
			vec3(0.0, 0.0, 1.0));
	}
    
    for (int i=0; i < 3; i++)
	{
		gl_Position = vec4(gl_in[i].gl_Position.xyz * swizzle_mat, 1.0f);
        f_voxel_pos = gl_in[i].gl_Position.xyz;
		position = g_world_pos[i];	
		iNormal = g_normal[i];
		iTexCoord = g_tex_coords[i];
        iColor = g_color[i];
        iTangent = g_tangent[i];

		EmitVertex();
	}

	EndPrimitive();

}  
)";
    return compile_program(VXVS, VXFS, VXGS);
  }

  
  

}
