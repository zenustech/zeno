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
    Program * get_shadow_program(std::shared_ptr<zeno::MaterialObject> mtl, std::shared_ptr<zeno::InstancingObject> inst)
  {
std::string SMVS;
    if (inst != nullptr)
    {
        SMVS = R"(
#version 330 core

uniform mat4 mVP;
uniform mat4 mInvVP;
uniform mat4 mView;
uniform mat4 mProj;
uniform mat4 mInvView;
uniform mat4 mInvProj;
uniform float fInstDeltaTime;
uniform int iInstFrameAmount;
uniform int iInstVertexAmount;
uniform sampler2D sInstVertexFrameSampler;

in vec3 vPosition;
in vec3 vColor;
in vec3 vNormal;
in vec3 vTexCoord;
in vec3 vTangent;
in mat4 mInstModel;
in float fInstTime;

out vec3 position;
out vec3 iColor;
out vec3 iNormal;
out vec3 iTexCoord;
out vec3 iTangent;

vec3 computeFramePosition()
{
  if (fInstDeltaTime == 0.0 || iInstFrameAmount == 0 || iInstVertexAmount == 0)
  {
    return vPosition;
  }

  int prevFrameID = int(fInstTime / fInstDeltaTime); 
  int nextFrameID = prevFrameID + 1;
  float dt = fInstTime - fInstDeltaTime * prevFrameID;

  prevFrameID = clamp(prevFrameID, 0, iInstFrameAmount - 1);  
  nextFrameID = clamp(nextFrameID, 0, iInstFrameAmount - 1);  

  vec3 prevPosition = texelFetch(sInstVertexFrameSampler, ivec2(gl_VertexID, prevFrameID), 0).rgb;
  vec3 nextPosition = texelFetch(sInstVertexFrameSampler, ivec2(gl_VertexID, nextFrameID), 0).rgb;
  return mix(prevPosition, nextPosition, dt);
}

void main()
{
  vec3 framePosition = computeFramePosition();
  position = vec3(mInstModel * vec4(framePosition, 1.0));
  iColor = vColor;
  iNormal = transpose(inverse(mat3(mInstModel))) * vNormal;
  iTexCoord = vTexCoord;
  iTangent = mat3(mInstModel) * vTangent;
  gl_Position = mVP * vec4(position, 1.0);
}
)";
    }
    else
  {
SMVS = R"(
#version 330 core

uniform mat4 mVP;
uniform mat4 mInvVP;
uniform mat4 mView;
uniform mat4 mProj;
uniform mat4 mInvView;
uniform mat4 mInvProj;

in vec3 vPosition;
in vec3 vColor;
in vec3 vNormal;
in vec3 vTexCoord;
in vec3 vTangent;

out vec3 position;
out vec3 iColor;
out vec3 iNormal;
out vec3 iTexCoord;
out vec3 iTangent;

void main()
{
  position = vPosition;
  iColor = vColor;
  iNormal = vNormal;
  iTexCoord = vTexCoord;
  iTangent = vTangent;
  gl_Position = mVP * vec4(position, 1.0);
}
)";
    }

auto SMFS = "#version 330 core\n/* common_funcs_begin */\n" + mtl->common + "\n/* common_funcs_end */\n"+R"(
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

void main()
{   
    vec3 att_pos = position;
    vec3 att_clr = iColor;
    vec3 att_nrm = iNormal;
    vec3 att_uv = iTexCoord;
    vec3 att_tang = iTangent;
    float att_NoL = 0;
)" + mtl->frag + R"(
    if(mat_opacity>=0.99)
         discard;
    //fColor = vec4(gl_FragCoord.zzz,1);
}
)";

auto SMGS = R"(
#version 330 core

layout(triangles, invocations = 8) in;
layout(triangle_strip, max_vertices = 3) out;

layout (std140, binding = 0) uniform LightSpaceMatrices
{
    mat4 lightSpaceMatrices[128];
};

void main()
{          
	for (int i = 0; i < 3; ++i)
	{
		gl_Position = lightSpaceMatrices[gl_InvocationID] * gl_in[i].gl_Position;
		gl_Layer = gl_InvocationID;
		EmitVertex();
	}
	EndPrimitive();
}  
)";
    return compile_program(SMVS, SMFS);
  }

  Program *get_points_program(std::string const &path) {
    auto vert = hg::file_get_content(path + ".points.vert");
    auto frag = hg::file_get_content(path + ".points.frag");

    if (vert.size() == 0) {
      vert = R"(
#version 330

uniform mat4 mVP;
uniform mat4 mInvVP;
uniform mat4 mView;
uniform mat4 mProj;
uniform float mPointScale;

in vec3 vPosition;
in vec3 vColor;
in vec3 vNormal;

out vec3 position;
out vec3 color;
out float radius;
out float opacity;
void main()
{
  position = vPosition;
  color = vColor;
  radius = vNormal.x;
  opacity = vNormal.y;

  vec3 posEye = vec3(mView * vec4(position, 1.0));
  float dist = length(posEye);
  if (radius != 0)
    gl_PointSize = max(1, radius * mPointScale / dist);
  else
    gl_PointSize = 1.5;
  gl_Position = mVP * vec4(position, 1.0);
}
)";
    }
    if (frag.size() == 0) {
      frag = R"(
#version 330

uniform mat4 mVP;
uniform mat4 mInvVP;
uniform mat4 mView;
uniform mat4 mProj;

in vec3 position;
in vec3 color;
in float radius;
in float opacity;
out vec4 fColor;
void main()
{
  const vec3 lightDir = vec3(0.577, 0.577, 0.577);
  vec2 coor = gl_PointCoord * 2 - 1;
  float len2 = dot(coor, coor);
  if (len2 > 1 && radius != 0)
    discard;
  vec3 oColor;
  if (radius != 0)
  {
    vec3 N;
    N.xy = gl_PointCoord*vec2(2.0, -2.0) + vec2(-1.0, 1.0);
    float mag = dot(N.xy, N.xy);
    N.z = sqrt(1.0-mag);

    // calculate lighting
    float diffuse = max(0.0, dot(lightDir, N) * 0.6 + 0.4);
    oColor = color * diffuse;
  }
  else
    oColor = color;
  fColor = vec4(oColor, 1.0 - opacity);
}
)";
    }

    return compile_program(vert, frag);
  }

  Program *get_lines_program(std::string const &path) {
    auto vert = hg::file_get_content(path + ".lines.vert");
    auto frag = hg::file_get_content(path + ".lines.frag");

    if (vert.size() == 0) {
      vert = R"(
#version 330

uniform mat4 mVP;
uniform mat4 mInvVP;
uniform mat4 mView;
uniform mat4 mProj;
uniform mat4 mInvView;
uniform mat4 mInvProj;

in vec3 vPosition;
in vec3 vColor;

out vec3 position;
out vec3 color;

void main()
{
  position = vPosition;
  color = vColor;

  gl_Position = mVP * vec4(position, 1.0);
}
)";
    }
    if (frag.size() == 0) {
      frag = R"(
#version 130

uniform mat4 mVP;
uniform mat4 mInvVP;
uniform mat4 mView;
uniform mat4 mProj;
uniform mat4 mInvView;
uniform mat4 mInvProj;

in vec3 position;
in vec3 color;
out vec4 fColor;
void main()
{
  fColor = vec4(color, 1.0);
}
)";
    }

    return compile_program(vert, frag);
  }

  Program *get_tris_program(std::string const &path, std::shared_ptr<zeno::MaterialObject> mtl, std::shared_ptr<zeno::InstancingObject> inst) {
    auto vert = hg::file_get_content(path + ".tris.vert");
    auto frag = hg::file_get_content(path + ".tris.frag");

    if (vert.size() == 0) {
        if (inst != nullptr)
        {
            vert = R"(
#version 330

uniform mat4 mVP;
uniform mat4 mInvVP;
uniform mat4 mView;
uniform mat4 mProj;
uniform mat4 mInvView;
uniform mat4 mInvProj;
uniform float fInstDeltaTime;
uniform int iInstFrameAmount;
uniform int iInstVertexAmount;
uniform sampler2D sInstVertexFrameSampler;

in vec3 vPosition;
in vec3 vColor;
in vec3 vNormal;
in vec3 vTexCoord;
in vec3 vTangent;
in mat4 mInstModel;
in float fInstTime;

out vec3 position;
out vec3 iColor;
out vec3 iNormal;
out vec3 iTexCoord;
out vec3 iTangent;

vec3 computeFramePosition()
{
  if (fInstDeltaTime == 0.0 || iInstFrameAmount == 0 || iInstVertexAmount == 0)
  {
    return vPosition;
  }

  int prevFrameID = int(fInstTime / fInstDeltaTime); 
  int nextFrameID = prevFrameID + 1;
  float dt = fInstTime - fInstDeltaTime * prevFrameID;

  prevFrameID = clamp(prevFrameID, 0, iInstFrameAmount - 1);  
  nextFrameID = clamp(nextFrameID, 0, iInstFrameAmount - 1);  

  vec3 prevPosition = texelFetch(sInstVertexFrameSampler, ivec2(gl_VertexID, prevFrameID), 0).rgb;
  vec3 nextPosition = texelFetch(sInstVertexFrameSampler, ivec2(gl_VertexID, nextFrameID), 0).rgb;
  return mix(prevPosition, nextPosition, dt);
}

void main()
{
  vec3 framePosition = computeFramePosition();
  position = vec3(mInstModel * vec4(framePosition, 1.0));
  iColor = vColor;
  iNormal = transpose(inverse(mat3(mInstModel))) * vNormal;
  iTexCoord = vTexCoord;
  iTangent = mat3(mInstModel) * vTangent;
  gl_Position = mVP * vec4(position, 1.0);
}
)";
        }
        else
        {
      vert = R"(
#version 330

uniform mat4 mVP;
uniform mat4 mInvVP;
uniform mat4 mView;
uniform mat4 mProj;
uniform mat4 mInvView;
uniform mat4 mInvProj;

in vec3 vPosition;
in vec3 vColor;
in vec3 vNormal;
in vec3 vTexCoord;
in vec3 vTangent;

out vec3 position;
out vec3 iColor;
out vec3 iNormal;
out vec3 iTexCoord;
out vec3 iTangent;
void main()
{
  position = vPosition;
  iColor = vColor;
  iNormal = vNormal;
  iTexCoord = vTexCoord;
  iTangent = vTangent;
  gl_Position = mVP * vec4(position, 1.0);
}
)";
        }
    }
    if (frag.size() == 0) {
        frag = R"(
#version 330
)" + (mtl ? mtl->extensions : "") +
               R"(
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
uniform samplerCube skybox;

uniform samplerCube irradianceMap;
uniform samplerCube prefilterMap;
uniform sampler2D brdfLUT;

vec3 pbr(vec3 albedo, float roughness, float metallic, float specular,
    vec3 nrm, vec3 idir, vec3 odir) {

  vec3 hdir = normalize(idir + odir);
  float NoH = max(0., dot_c(hdir, nrm));
  float NoL = max(0., dot_c(idir, nrm));
  float NoV = max(0., dot_c(odir, nrm));
  float VoH = clamp(dot_c(odir, hdir), 0., 1.);
  float LoH = clamp(dot_c(idir, hdir), 0., 1.);

  vec3 f0 = metallic * albedo + (1. - metallic) * 0.16 * specular;
  vec3 fdf = f0 + (1. - f0) * pow(1. - VoH, 5.);

  roughness *= roughness;
  float k = (roughness + 1.) * (roughness + 1.) / 8.;
  float vdf = 0.25 / ((NoV * k + 1. - k) * (NoL * k + 1. - k));

  float alpha2 = max(0., roughness * roughness);
  float denom = 1. - NoH * NoH * (1. - alpha2);
  float ndf = alpha2 / (denom * denom);

  vec3 brdf = fdf * vdf * ndf * f0 + (1. - f0) * albedo;
  return brdf * NoL;
}

)" + (!mtl ?
           R"(
vec3 studioShading(vec3 albedo, vec3 view_dir, vec3 normal, vec3 tangent) {
    vec3 color = vec3(0.0);
    vec3 light_dir;

    light_dir = normalize((mInvView * vec4(1., 2., 5., 0.)).xyz);
    color += vec3(0.45, 0.47, 0.5) * pbr(albedo, 0.44, 0.0, 1.0, normal, light_dir, view_dir);

    light_dir = normalize((mInvView * vec4(-4., -2., 1., 0.)).xyz);
    color += vec3(0.3, 0.23, 0.18) * pbr(albedo, 0.37, 0.0, 1.0, normal, light_dir, view_dir);

    light_dir = normalize((mInvView * vec4(3., -5., 2., 0.)).xyz);
    color += vec3(0.15, 0.2, 0.22) * pbr(albedo, 0.48, 0.0, 1.0, normal, light_dir, view_dir);

    color *= 1.2;
    //color = pow(clamp(color, 0., 1.), vec3(1./2.2));
    return color;
}
)"
           : "\n/* common_funcs_begin */\n" + mtl->common + "\n/* common_funcs_end */\n"
                                                            R"(
  
vec3 CalculateDiffuse(
    in vec3 albedo){                              
    return (albedo / 3.1415926);
}


vec3 CalculateHalfVector(
    in vec3 toLight, in vec3 toView){
    return normalize(toLight + toView);
}

// Specular D -  Normal distribution function (NDF)
float CalculateNDF( // GGX/Trowbridge-Reitz NDF
    in vec3  surfNorm,
    in vec3  halfVector,
    in float roughness){
    float a2 = (roughness * roughness * roughness * roughness);
    float halfAngle = dot(surfNorm, halfVector);
    float d = (halfAngle * a2 - halfAngle) * halfAngle + 1;
    return (a2 / (3.1415926 *  d * d));
}

// Specular G - Microfacet geometric attenuation
float CalculateAttenuation( // GGX/Schlick-Beckmann
    in vec3  surfNorm,
    in vec3  vector,
    in float k)
{
    float d = max(dot_c(surfNorm, vector), 0.0);
 	return (d / ((d * (1.0 - k)) + k));
}
float CalculateAttenuationAnalytical(// Smith for analytical light
    in vec3  surfNorm,
    in vec3  toLight,
    in vec3  toView,
    in float roughness)
{
    float k = pow((roughness*roughness + 1.0), 2.0) * 0.125;

    // G(l) and G(v)
    float lightAtten = CalculateAttenuation(surfNorm, toLight, k);
    float viewAtten  = CalculateAttenuation(surfNorm, toView, k);

    // Smith
    return (lightAtten * viewAtten);
}

vec3 fresnelSchlickRoughness(float cosTheta, vec3 F0, float roughness){
    return F0 + (max(vec3(1.0-roughness), F0) - F0) * pow(1.0 - cosTheta, 5.0);
}
// Specular F - Fresnel reflectivity
vec3 CalculateFresnel(
    in vec3 surfNorm,
    in vec3 toView,
    in vec3 fresnel0)
{
	float d = max(dot_c(surfNorm, toView), 0.0); 
    float p = ((-5.55473 * d) - 6.98316) * d;
        
    return fresnel0 + ((1.0 - fresnel0) * pow(1.0 - d, 5.0));
}

// Specular Term - put together
vec3 CalculateSpecularAnalytical(
    in    vec3  surfNorm,            // Surface normal
    in    vec3  toLight,             // Normalized vector pointing to light source
    in    vec3  toView,              // Normalized vector point to the view/camera
    in    vec3  fresnel0,            // Fresnel incidence value
    inout vec3  sfresnel,            // Final fresnel value used a kS
    in    float roughness)           // Roughness parameter (microfacet contribution)
{
    vec3 halfVector = CalculateHalfVector(toLight, toView);

    float ndf      = CalculateNDF(surfNorm, halfVector, roughness);
    float geoAtten = CalculateAttenuationAnalytical(surfNorm, toLight, toView, roughness);

    sfresnel = CalculateFresnel(surfNorm, toView, fresnel0);

    vec3  numerator   = (sfresnel * ndf * geoAtten); // FDG
    float denominator = 4.0 * dot_c(surfNorm, toLight) * dot_c(surfNorm, toView);

    return (numerator / denominator);
}
float D_GGX( float a2, float NoH )
{
	float d = ( NoH * a2 - NoH ) * NoH + 1;	// 2 mad
	return a2 / ( 3.1415926*d*d );					// 4 mul, 1 rcp
}
float Vis_SmithJointApprox( float a2, float NoV, float NoL )
{
	float a = sqrt(a2);
	float Vis_SmithV = NoL * ( NoV * ( 1 - a ) + a );
	float Vis_SmithL = NoV * ( NoL * ( 1 - a ) + a );
	return 0.5 / ( Vis_SmithV + Vis_SmithL );
}
vec3 F_Schlick( vec3 SpecularColor, float VoH )
{
	float Fc = pow( 1 - VoH , 5.0 );					// 1 sub, 3 mul
	//return Fc + (1 - Fc) * SpecularColor;		// 1 add, 3 mad
	
	
	return clamp( 50.0 * SpecularColor.g, 0, 1 ) * Fc + (1 - Fc) * SpecularColor;
	
}
vec3 SpecularGGX( float Roughness, vec3 SpecularColor, float NoL, float NoH, float NoV, float VoH)
{
	float a2 = pow( Roughness, 4);
	
	// Generalized microfacet specular
    float D = D_GGX( a2,  NoH);
	float Vis = Vis_SmithJointApprox( a2, NoV, NoL );
	vec3 F = F_Schlick( SpecularColor, VoH );

	return (D * Vis) * F;
}
vec3 UELighting(
    in vec3  surfNorm,
    in vec3  toLight,
    in vec3  toView,
    in vec3  albedo,
    in float roughness,
    in float metallic)
{
    vec3 ks       = vec3(0.0);
    vec3 diffuse  = CalculateDiffuse(albedo);
    vec3 halfVec = normalize(toLight + toView);
    float NoL = dot(surfNorm, toLight);
    float NoH = dot(surfNorm, halfVec);
    float NoV = dot(surfNorm, toView);
    float VoH = dot(toView, halfVec);
    float angle = clamp(dot_c(surfNorm, toLight), 0.0, 1.0);
    return (diffuse * (1-metallic) + SpecularGGX(roughness, vec3(0,0,0), NoL, NoH, NoV, NoH))*angle;

}
// Solve Rendering Integral - Final
vec3 CalculateLightingAnalytical(
    in vec3  surfNorm,
    in vec3  toLight,
    in vec3  toView,
    in vec3  albedo,
    in float roughness,
    in float metallic)
{
    vec3 fresnel0 = mix(vec3(0.04), albedo, metallic);
    vec3 ks       = vec3(0.0);
    vec3 diffuse  = CalculateDiffuse(albedo);
    vec3 specular = CalculateSpecularAnalytical(surfNorm, toLight, toView, fresnel0, ks, roughness);
    vec3 kd       = (1.0 - ks);

    float angle = clamp(dot_c(surfNorm, toLight), 0.0, 1.0);

    return ((kd * diffuse) + specular) * angle;
}
float VanDerCorpus(int n, int base) {
    float invBase = 1.0 / float(base);
    float denom   = 1.0;
    float result  = 0.0;

    for(int i = 0; i < 32; ++i)
    {
        if(n > 0)
        {
            denom   = mod(float(n), 2.0);
            result += denom * invBase;
            invBase = invBase / 2.0;
            n       = int(float(n) / 2.0);
        }
    }

    return result;
}

vec2 Hammersley(int i, int N) {
    return vec2(float(i)/float(N), VanDerCorpus(i, 2));
}  
float CalculateAttenuationIBL(
    in float roughness,
    in float normDotLight,          // Clamped to [0.0, 1.0]
    in float normDotView)           // Clamped to [0.0, 1.0]
{
    float k = pow(roughness*roughness, 2.0) * 0.5;
    
    float lightAtten = (normDotLight / ((normDotLight * (1.0 - k)) + k));
    float viewAtten  = (normDotView / ((normDotView * (1.0 - k)) + k));
    
    return (lightAtten * viewAtten);
}

vec3 ImportanceSample(vec2 Xi, vec3 N, float roughness) {
    float a = roughness*roughness;
	
    float phi = 2.0 * 3.1415926 * Xi.x;
    float cosTheta = sqrt((1.0 - Xi.y) / (1.0 + (a*a - 1.0) * Xi.y));
    float sinTheta = sqrt(1.0 - cosTheta*cosTheta);
	
    vec3 H;
    H.x = cos(phi) * sinTheta;
    H.y = sin(phi) * sinTheta;
    H.z = cosTheta;

    //vec3 up        = abs(N.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
    vec3 tangent;//   = normalize(cross(up, N));
    vec3 bitangent;// = cross(N, tangent);
	pixarONB(N, tangent, bitangent);
    vec3 sampleVec = tangent * H.x + bitangent * H.y + N * H.z;
    return normalize(sampleVec);
}

#define time 0
float hash2(in vec2 n){ return fract(sin(dot(n, vec2(12.9898, 4.1414))) * 43758.5453); }

mat2 mm2(in float a){float c = cos(a), s = sin(a);return mat2(c,-s,s,c);}

vec2 field(in vec2 x)
{
    vec2 n = floor(x);
	vec2 f = fract(x);
	vec2 m = vec2(5.,1.);
	for(int j=0; j<=1; j++)
	for(int i=0; i<=1; i++)
    {
		vec2 g = vec2( float(i),float(j) );
		vec2 r = g - f;
        float d = length(r)*(sin(time*0.12)*0.5+1.5); //any metric can be used
        d = sin(d*5.+abs(fract(time*0.1)-0.5)*1.8+0.2);
		m.x *= d;
		m.y += d*1.2;
    }
	return abs(m);
}

vec3 tex(in vec2 p, in float ofst)
{    
    vec2 rz = field(p*ofst*0.5);
	vec3 col = sin(vec3(2.,1.,.1)*rz.y*.2+3.+ofst*2.)+.9*(rz.x+1.);
	col = col*col*.5;
    col *= sin(length(p)*9.+time*5.)*0.35+0.65;
	return col;
}

vec3 cubem(in vec3 p, in float ofst)
{
    p = abs(p);
    if (p.x > p.y && p.x > p.z) return tex( vec2(p.z,p.y)/p.x,ofst );
    else if (p.y > p.x && p.y > p.z) return tex( vec2(p.z,p.x)/p.y,ofst );
    else return tex( vec2(p.y,p.x)/p.z,ofst );
}

const float PI = 3.14159265358979323846;

//important to do: load env texture here
vec3 SampleEnvironment(in vec3 reflVec)
{
    //if(reflVec.y>-0.5) return vec3(0,0,0);
    //else return vec3(1,1,1);//cubem(reflVec, 0);//texture(TextureEnv, reflVec).rgb;
    //here we have the problem reflVec is in eyespace but we need it in world space
    vec3 r = inverse(transpose(inverse(mat3(mView[0].xyz, mView[1].xyz, mView[2].xyz))))*reflVec;
    return texture(skybox, r).rgb;
}

/**
 * Performs the Riemann Sum approximation of the IBL lighting integral.
 *
 * The ambient IBL source hits the surface from all angles. We average
 * the lighting contribution from a number of random light directional
 * vectors to approximate the total specular lighting.
 *
 * The number of steps is controlled by the 'IBL Steps' global.
 */
 //Geometry for IBL uses a different k than direct lighting
 //GGX and Schlick-Beckmann
float geometry(float cosTheta, float k){
	return (cosTheta)/(cosTheta*(1.0-k)+k);
}
float smithsIBL(float NdotV, float NdotL, float roughness){
    float k = (roughness * roughness);
    k = k*k; 
	return geometry(NdotV, k) * geometry(NdotL, k);
}
vec3 CalculateSpecularIBL(
    in    vec3  surfNorm,
    in    vec3  toView,
    in    vec3  fresnel0,
    inout vec3  sfresnel,
    in    float roughness)
{
    vec3 totalSpec = vec3(0.0);
    vec3 toSurfaceCenter = reflect(-toView, surfNorm);
    int IBLSteps = 64;
    for(int i = 0; i < IBLSteps; ++i)
    {
        // The 2D hemispherical sampling vector
    	vec2 xi = Hammersley(i, IBLSteps);
        
        // Bias the Hammersley vector towards the specular lobe of the surface roughness
        vec3 H = ImportanceSample(xi, surfNorm, roughness);
        
        // The light sample vector
        vec3 L = normalize((2.0 * dot(toView, H) * H) - toView);
        
        float NoV = clamp(dot_c(surfNorm, toView), 0.0, 1.0);
        float NoL = clamp(dot_c(surfNorm, L), 0.0, 1.0);
        float NoH = clamp(dot_c(surfNorm, H), 0.0, 1.0);
        float VoH = clamp(dot_c(toView, H), 0.0, 1.0);
        
        if(NoL > 0.0)
        {
            vec3 color = SampleEnvironment(L);
            
            float geoAtten = smithsIBL(NoV, NoL, roughness);
            vec3  fresnel = CalculateFresnel(surfNorm, toView, fresnel0);
            
            sfresnel += fresnel;
            totalSpec += (color * fresnel * geoAtten * VoH) / (NoH * NoV);
        }
    }
    
    sfresnel /= float(IBLSteps);
    
    return (totalSpec / float(IBLSteps));
}
float DistributionGGX(vec3 N, vec3 H, float roughness)
{
    float a = roughness*roughness;
    float a2 = a*a;
    float NdotH = max(dot(N, H), 0.0);
    float NdotH2 = NdotH*NdotH;

    float nom   = a2;
    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = 3.1415926 * denom * denom;

    return nom / denom;
}
// ----------------------------------------------------------------------------
float GeometrySchlickGGX(float NdotV, float roughness)
{
    float r = (roughness + 1.0);
    float k = (r*r) / 8.0;

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
vec3 fresnelSchlick(float cosTheta, vec3 F0)
{
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

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

vec3 CalculateLightingIBLToon(
    in vec3  N,
    in vec3  V,
    in vec3  albedo,
    in float roughness,
    in float metallic)
{
    mat3 m = inverse(transpose(inverse(mat3(mView[0].xyz, mView[1].xyz, mView[2].xyz))));
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
    vec3 prefilteredColor2 = textureLod(prefilterMap, m*R,  max(roughness, 0.5) * MAX_REFLECTION_LOD).rgb;
    prefilteredColor = clamp(smoothstep(0.5,0.5,length(prefilteredColor)), 0,1)*vec3(1,1,1);
    vec3 specularColor = mix(prefilteredColor+0.2, prefilteredColor2, prefilteredColor);
    vec2 brdf  = texture(brdfLUT, vec2(max(dot(N, V), 0.0), roughness)).rg;
    brdf.r = (floor(brdf.r/0.33)+0.165)*0.33;
    vec3 specular = specularColor * (F * brdf.r + smoothstep(0.7,0.7,brdf.y));

    return (kD * diffuse + specular);

}

vec3 ACESToneMapping(vec3 color, float adapted_lum)
{
	const float A = 2.51f;
	const float B = 0.03f;
	const float C = 2.43f;
	const float D = 0.59f;
	const float E = 0.14f;

	color *= adapted_lum;
	return (color * (A * color + B)) / (color * (C * color + D) + E);
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

float toonSpecular(vec3 V, vec3 L, vec3 N, float roughness)
{
    float NoV = dot(N,V);
    float _SpecularSize = pow((1-roughness),5);
    float specularFalloff = NoV;
    specularFalloff = pow(specularFalloff, 2);
    vec3 reflectionDirection = reflect(L, N);
    float towardsReflection = dot(V, -reflectionDirection);
    float specularChange = fwidth(towardsReflection);
    float specularIntensity = smoothstep(1.0 - _SpecularSize, 1.0 - _SpecularSize + specularChange, towardsReflection);
    return clamp(specularIntensity,0,1);
}
vec3 histThings(vec3 s)
{
    vec3 norms = s/(length(s)+0.00001);
    float ls = length(s);
    ls = ceil(ls/0.2)*0.2;
    return norms * ls;
}
)" + R"(
float V_Kelemen(float LoH) {
    return 0.25 / (LoH * LoH);
}
vec3 ToonBRDF(vec3 baseColor, float metallic, float subsurface, 
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
    float NoL = dot(N,L);
    float shad1 = smoothstep(0.3, 0.31, NoL);
    float shad2 = smoothstep(0.0,0.01, NoL);
    vec3 diffuse = mon2lin(baseColor)/PI;
    vec3 shadowC1 = diffuse * 0.4;
    vec3 C1 = mix(shadowC1, diffuse, shad1);
    vec3 shadowC2 = shadowC1 * 0.4;
    vec3 C2 = mix(shadowC2, C1, shad2);

    vec3 H = normalize(L+V);
    vec3 F0 = vec3(0.04); 
    F0 = mix(F0, baseColor, metallic);
    // Cook-Torrance BRDF
    float NDF = DistributionGGX(N, H, roughness);   
    float G   = GeometrySmith(N, V, L, roughness);    
    vec3 F    = fresnelSchlick(max(dot(H, V), 0.0), F0);        
    
    vec3 numerator    = NDF * G * F;
    float denominator = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.0001; // + 0.0001 to prevent divide by zero
    vec3 s = numerator / denominator;
    
    // kS is equal to Fresnel
    vec3 kS = F;
    // for energy conservation, the diffuse and specular light can't
    // be above 1.0 (unless the surface emits light); to preserve this
    // relationship the diffuse component (kD) should equal 1.0 - kS.
    vec3 kD = vec3(1.0) - kS;
    // multiply kD by the inverse metalness such that only non-metals 
    // have diffuse lighting, or a linear blend if partly metal (pure metals
    // have no diffuse light).
    kD *= 1.0 - metallic;	                

    vec3 norms = s/(length(s)+0.00001);
    float ls = length(s);
    ls = ceil(ls/0.4)*0.4;


    return (kD*C2 + norms * ls * toonSpecular(V, L, N, roughness));
}
vec3 ToonDisneyBRDF(vec3 baseColor, float metallic, float subsurface, 
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
    float viewIndp = mix(1.0, Fd90, FL);
    float Fd = (floor(viewIndp/0.33)+0.165) * 0.33 * mix(1.0, Fd90, FV);

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

    float shad1 = smoothstep(0.3, 0.31, NdotL);
    float shad2 = smoothstep(0.0,0.01, NdotL);
    vec3 shadowC1 = vec3(1,1,1) * 0.4;
    vec3 C1 = mix(shadowC1, vec3(1,1,1), shad1);
    vec3 shadowC2 = shadowC1 * 0.4;
    vec3 C2 = mix(shadowC2, C1, shad2);
    //c1 *= C2.x;
    
    Fsheen = Fsheen/(length(Fsheen)+1e-5) * (floor(length(Fsheen)/0.2)+0.1)*0.2;
    vec3 fspecularTerm = (Gs*Fs*Ds);
    vec3 fcoatTerm =  vec3(.25*clearcoat*Gr*Fr*Dr);

    return ((c1 * Cdlin  + Fsheen)
        * (1-metallic)
        + (normalize(fspecularTerm) * ceil(length(fspecularTerm)/0.3) * 0.3 + fcoatTerm)* toonSpecular(V, L, N, roughness)) * C2 ;
        
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
float softLight0(float a, float b)
{
float G;
float res;
if(b<=0.25)
    G = ((16*b-12)*b+4)*b;
else
   G = sqrt(b);
if(a<=0.5)
   res = b - (1-2*a)*b*(1-b);
else
   res = b+(2*a-1)*(G-b);
return res;
}
float linearLight0(float a, float b)
{
    if(a>0.5)
        return b + 2 * (a-0.5);
   else
       return b + a - 1;
}
float brightness(vec3 c)
{
    return sqrt(c.x * c.r * 0.241 + c.y * c.y * 0.691 + c.z * c.z * 0.068);
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
//layout (std140, binding = 0) uniform LightSpaceMatrices
//{
uniform mat4 lightSpaceMatrices[128];
//};
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
            vec3 noise = random3(pos+vec3(x, y,0)) *0.01*softness / pow(2,layer);
            float pcfDepth = sampleShadowArray(lightNo, coord + (vec2(x, y) * softness + noise.xy * 0) * texelSize, layer)  * (far1-near1) + near1 ; 
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
            vec3 noise = random3(pos+vec3(x, y,0)) * 0.01*softness ;
            float pcfDepth1 = sampleShadowArray(lightNo, coord1 + (vec2(x, y) * softness + noise.xy  * 0/ pow(2,layer)) * texelSize, layer) * (far1-near1) + near1;
            
            float pcfDepth2 = sampleShadowArray(lightNo, coord2 + (vec2(x, y) * softness + noise.xy  * 0 / pow(2,layer+1)) * texelSize, layer+1) * (far2-near2) + near2; 
            float s1 = ((currentDepth1 * (far1-near1) + near1 - bias) > pcfDepth1)?1.0 : 0.0;
            float s2 = ((currentDepth2 * (far2-near2) + near2 - bias) > pcfDepth2)?1.0 : 0.0;
            shadow += mix(s1, s2, 0.5);        
        }    
    }
    float size = 2.0*float(k)+1.0;
    return shadow /= (size*size);
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
    float bias = (1-pow(slop,0.1)) * 0.1 + pow(slop,0.1) * 0.001;
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
float PCFAttLayer(int lightNo, float currentDepth, float bias, vec3 pos, int layer, int k, float softness, vec2 coord, float near, float far)
{
    
    float length = 0.0;
    float res = far;
    vec2 texelSize = 1.0 / vec2(textureSize(shadowMap[lightNo * (cascadeCount + 1) + 0], 0));
    for(int x = -k; x <= k; ++x)
    {
        for(int y = -k; y <= k; ++y)
        {
            vec3 noise = random3(pos+vec3(x, y,0)*0.01*softness);
            float pcfDepth = sampleShadowArray(lightNo, coord + (vec2(x, y) * softness + noise.xy) * softness * texelSize, layer) * (far-near) + near; 
            res = min(res, abs(currentDepth - pcfDepth));
        }    
    }
    return res;
}

float lightAttenuation(int lightNo, vec3 fragPosWorldSpace, float softness)
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
    float nearPlane = near[lightNo * (cascadeCount + 1) + layer];
    float farPlane = far[lightNo * (cascadeCount + 1) + layer];
    float currentDepth = projCoords.z * (farPlane - nearPlane) + nearPlane;

    float avgL = PCFAttLayer(lightNo, currentDepth, 0, fragPosWorldSpace, layer, 5, 1, projCoords.xy, nearPlane, farPlane);

    return avgL;
    
}

)" + R"(
uniform mat4 reflectMVP[16];
uniform sampler2DRect reflectionMap0;
uniform sampler2DRect reflectionMap1;
uniform sampler2DRect reflectionMap2;
uniform sampler2DRect reflectionMap3;
uniform sampler2DRect reflectionMap4;
uniform sampler2DRect reflectionMap5;
uniform sampler2DRect reflectionMap6;
uniform sampler2DRect reflectionMap7;
uniform sampler2DRect reflectionMap8;
uniform sampler2DRect reflectionMap9;
uniform sampler2DRect reflectionMap10;
uniform sampler2DRect reflectionMap11;
uniform sampler2DRect reflectionMap12;
uniform sampler2DRect reflectionMap13;
uniform sampler2DRect reflectionMap14;
uniform sampler2DRect reflectionMap15;
vec4 sampleReflectRectID(vec2 coord, int id)
{
    if(id==0) return texture2DRect(reflectionMap0, coord);
    if(id==1) return texture2DRect(reflectionMap1, coord);
    if(id==2) return texture2DRect(reflectionMap2, coord);
    if(id==3) return texture2DRect(reflectionMap3, coord);
    if(id==4) return texture2DRect(reflectionMap4, coord);
    if(id==5) return texture2DRect(reflectionMap5, coord);
    if(id==6) return texture2DRect(reflectionMap6, coord);
    if(id==7) return texture2DRect(reflectionMap7, coord);
    if(id==8) return texture2DRect(reflectionMap8, coord);
    if(id==9) return texture2DRect(reflectionMap9, coord);
    if(id==10) return texture2DRect(reflectionMap10, coord);
    if(id==11) return texture2DRect(reflectionMap11, coord);
    if(id==12) return texture2DRect(reflectionMap12, coord);
    if(id==13) return texture2DRect(reflectionMap13, coord);
    if(id==14) return texture2DRect(reflectionMap14, coord);
    if(id==15) return texture2DRect(reflectionMap15, coord);
}
float mad(float a, float b, float c)
{
    return c + a * b;
}
vec3 mad3(vec3 a, vec3 b, vec3 c)
{
    return c + a * b;
}
vec3 skinBRDF(vec3 normal, vec3 light, float curvature)
{
    float NdotL = dot(normal, light) * 0.5 + 0.5; // map to 0 to 1 range
    float curva = (1.0/mad(curvature, 0.5 - 0.0625, 0.0625) - 2.0) / (16.0 - 2.0); 
    float oneMinusCurva = 1.0 - curva;
    vec3 curve0;
    {
        vec3 rangeMin = vec3(0.0, 0.3, 0.3);
        vec3 rangeMax = vec3(1.0, 0.7, 0.7);
        vec3 offset = vec3(0.0, 0.06, 0.06);
        vec3 t = clamp( mad3(vec3(NdotL), 1.0 / (rangeMax - rangeMin), (offset + rangeMin) / (rangeMin - rangeMax)  ), vec3(0), vec3(1));
        vec3 lowerLine = (t * t) * vec3(0.65, 0.5, 0.9);
        lowerLine.r += 0.045;
        lowerLine.b *= t.b;
        vec3 m = vec3(1.75, 2.0, 1.97);
        vec3 upperLine = mad3(vec3(NdotL), m, vec3(0.99, 0.99, 0.99) -m );
        upperLine = clamp(upperLine, vec3(0), vec3(1));
        vec3 lerpMin = vec3(0.0, 0.35, 0.35);
        vec3 lerpMax = vec3(1.0, 0.7 , 0.6 );
        vec3 lerpT = clamp( mad3(vec3(NdotL), vec3(1.0)/(lerpMax-lerpMin), lerpMin/ (lerpMin - lerpMax) ), vec3(0), vec3(1));
        curve0 = mix(lowerLine, upperLine, lerpT * lerpT);
    }
    vec3 curve1;
    {
        vec3 m = vec3(1.95, 2.0, 2.0);
        vec3 upperLine = mad3( vec3(NdotL), m, vec3(0.99, 0.99, 1.0) - m);
        curve1 = clamp(upperLine,0,1);
    }
    float oneMinusCurva2 = oneMinusCurva * oneMinusCurva;
    vec3 brdf = mix(curve0, curve1, mad(oneMinusCurva2, -1.0 * oneMinusCurva2, 1.0) );
    return brdf;
}
vec3 reflectionCalculation(vec3 worldPos, int id)
{
    vec4 fragPosReflectSpace = reflectMVP[id] * vec4(worldPos, 1.0);
    // perform perspective divide
    vec3 projCoords = fragPosReflectSpace.xyz / fragPosReflectSpace.w;
    // transform to [0,1] range
    projCoords = projCoords * 0.5 + 0.5;
    if (projCoords.x>=0&&projCoords.x<=1&&projCoords.y>=0&&projCoords.y<=1)
    {
        return sampleReflectRectID(projCoords.xy * vec2(textureSize(reflectionMap0,0)), id ).xyz;
    }
    return vec3(0,0,0);
}
uniform float reflectPass;
uniform float reflectionViewID;
uniform float depthPass;
uniform sampler2DRect depthBuffer;
uniform vec3 reflect_normals[16];
uniform vec3 reflect_centers[16];
vec3 studioShading(vec3 albedo, vec3 view_dir, vec3 normal, vec3 old_tangent) {
    vec4 projPos = mView * vec4(position.xyz, 1.0);
    //normal = normalize(normal);
    vec3 L1 = light[0];
    vec3 att_pos = position;
    vec3 att_clr = iColor;
    vec3 att_nrm = normal;
    vec3 att_uv = iTexCoord;
    vec3 att_tang = old_tangent;
    float att_NoL = dot(normal, L1);
    //if(depthPass<=0.01)
    //{
    //    
    //    float d = texture2DRect(depthBuffer, gl_FragCoord.xy).r;
    //    if(d==0 || abs(projPos.z)>abs(d) )
    //        discard;
    //}
    /* custom_shader_begin */
)" + mtl->frag + R"(
    if(reflectPass==1.0 && mat_reflection==1.0 )
        discard;
    if(reflectPass==1.0 && dot(reflect_normals[int(reflectionViewID)], position-reflect_centers[int(reflectionViewID)])<0)
        discard;
    /* custom_shader_end */
    if(mat_opacity>=0.99 && mat_reflection!=1.0)
        discard;
    
    //if(depthPass>=0.99)
    //{
    //    return abs(projPos.zzz);
    //}
    
    vec3 colorEmission = mat_emission;
    mat_metallic = clamp(mat_metallic, 0, 1);
    vec3 new_normal = normal; /* TODO: use mat_normal to transform this */
    vec3 color = vec3(0,0,0);
    vec3 light_dir;
    vec3 albedo2 = mat_basecolor;
    float roughness = mat_roughness;
    vec3 tan = normalize(old_tangent - dot(normal, old_tangent)*normal);
    mat3 TBN = mat3(tan, cross(normal, tan), normal);

    new_normal = TBN*normalize(mat_normal);
    mat3 eyeinvmat = transpose(inverse(mat3(mView[0].xyz, mView[1].xyz, mView[2].xyz)));
    new_normal = eyeinvmat*new_normal;
    mat3 eyemat = mat3(mView[0].xyz, mView[1].xyz, mView[2].xyz);
    //vec3 up        = abs(new_normal.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
    vec3 tangent = eyemat*tan;//   = normalize(cross(up, new_normal));
    vec3 bitangent = eyemat*TBN[1];// = cross(new_normal, tangent);
    //pixarONB(new_normal, tangent, bitangent);
    color = vec3(0,0,0);
    vec3 realColor = vec3(0,0,0);
    vec3 shadowAcc = vec3(0,0,0);
    float lightsNo = 0;
    for(int lightId=0; lightId<lightNum; lightId++){
        light_dir = eyemat*lightDir[lightId];
        vec3 photoReal = BRDF(mat_basecolor, mat_metallic,mat_subsurface,mat_specular,mat_roughness,mat_specularTint,mat_anisotropic,mat_sheen,mat_sheenTint,mat_clearcoat,mat_clearcoatGloss,normalize(light_dir), normalize(view_dir), normalize(new_normal),normalize(tangent), normalize(bitangent)) * lightIntensity[lightId];// * vec3(1, 1, 1) * mat_zenxposure;
        vec3 NPR = ToonDisneyBRDF(mat_basecolor, mat_metallic,0,mat_specular,mat_roughness,mat_specularTint,mat_anisotropic,mat_sheen,mat_sheenTint,mat_clearcoat,mat_clearcoatGloss,normalize(light_dir), normalize(view_dir), normalize(new_normal),normalize(tangent), normalize(bitangent)) * lightIntensity[lightId];// * vec3(1, 1, 1) * mat_zenxposure;

        vec3 sss =  vec3(0);
        if(mat_subsurface>0)
        {
            vec3 vl = light_dir + new_normal * mat_sssParam.x;

            float ltDot = pow(clamp(dot(normalize(view_dir), -vl),0,1), 12.0) * mat_sssParam.y;
            float lthick = lightAttenuation(lightId, position, shadowSoftness[lightId]);
            sss = mat_thickness * exp(-lthick * mat_sssParam.z) * ltDot * mat_sssColor * lightIntensity[lightId];
        }
        if(mat_foliage>0)
        {
            if(dot(new_normal, light_dir)<0)
            {
                sss += mat_foliage * clamp(dot(-new_normal, light_dir)*0.6+0.4, 0,1) * mon2lin(mat_basecolor)/PI;
            }
        }
        if(mat_skin>0)
        {
            sss += mat_skin * skinBRDF(new_normal, light_dir, mat_curvature) * lightIntensity[lightId] * mon2lin(mat_basecolor)/PI;
        }

        vec3 lcolor = mix(photoReal, NPR, mat_toon) + mat_subsurface * sss;
    //   color +=  
    //       CalculateLightingAnalytical(
    //           new_normal,
    //           normalize(light_dir),
    //           normalize(view_dir),
    //           albedo2,
    //           roughness,
    //           mat_metallic) * vec3(1, 1, 1) * mat_zenxposure;
    //    color += vec3(0.45, 0.47, 0.5) * pbr(mat_basecolor, mat_roughness,
    //             mat_metallic, mat_specular, new_normal, light_dir, view_dir);

    //    light_dir = vec3(0,1,-1);
    //    color += vec3(0.3, 0.23, 0.18) * pbr(mat_basecolor, mat_roughness,
    //             mat_metallic, mat_specular, new_normal, light_dir, view_dir);
    //    color +=  
    //        CalculateLightingAnalytical(
    //            new_normal,
    //            light_dir,
    //            view_dir,
    //            albedo2,
    //            roughness,
    //            mat_metallic) * vec3(0.3, 0.23, 0.18)*5;
    //    light_dir = vec3(0,-0.2,-1);
    //    color +=  
    //        CalculateLightingAnalytical(
    //            new_normal,
    //            light_dir,
    //            view_dir,
    //            albedo2,
    //            roughness,
    //            mat_metallic) * vec3(0.15, 0.2, 0.22)*6;
    //    color += vec3(0.15, 0.2, 0.22) * pbr(mat_basecolor, mat_roughness,
    //             mat_metallic, mat_specular, new_normal, light_dir, view_dir);


        
        
        float shadow = ShadowCalculation(lightId, position, shadowSoftness[lightId], tan, TBN[1],3);
        vec3 sclr = clamp(vec3(1.0-shadow) + shadowTint[lightId], vec3(0), vec3(1));
        color += lcolor * sclr;
        realColor += photoReal * sclr;
    }
    
    
    vec3 iblPhotoReal =  CalculateLightingIBL(new_normal,view_dir,albedo2,roughness,mat_metallic);
    vec3 iblNPR = CalculateLightingIBLToon(new_normal,view_dir,albedo2,roughness,mat_metallic);
    vec3 ibl = mat_ao * mix(iblPhotoReal, iblNPR,mat_toon);
    color += ibl;
    realColor += iblPhotoReal;
    float brightness0 = brightness(realColor)/(brightness(mon2lin(mat_basecolor))+0.00001);
    float brightness1 = smoothstep(mat_shape.x, mat_shape.y, dot(new_normal, light_dir));
    float brightness = mix(brightness1, brightness0, mat_style);
    
    float brightnessNoise = softLight0(mat_strokeNoise, brightness);
    brightnessNoise = smoothstep(mat_shad.x, mat_shad.y, brightnessNoise);
    float stroke = linearLight0(brightnessNoise, mat_stroke);
    stroke = clamp(stroke + mat_stroke, 0,1);
    vec3 strokeColor = clamp(vec3(stroke) + mat_strokeTint, vec3(0), vec3(1));

    //strokeColor = pow(strokeColor,vec3(2.0));
    color = mix(color, mix(color*strokeColor, color, strokeColor), mat_toon);

    color = ACESFitted(color.rgb, 2.2);
    if(mat_reflection==1.0 && mat_reflectID>-1 && mat_reflectID<16)
        color = mix(color, reflectionCalculation(position, int(mat_reflectID)), mat_reflection);
    return color + colorEmission;


}
)") + R"(

vec3 calcRayDir(vec3 pos)
{
  vec4 vpos = mView * vec4(pos, 1);
//   vec2 uv = vpos.xy / vpos.w;
//   vec4 ro = mInvVP * vec4(uv, -1, 1);
//   vec4 re = mInvVP * vec4(uv, +1, 1);
//   vec3 rd = normalize(re.xyz / re.w - ro.xyz / ro.w);
  return normalize(vpos.xyz);
}
uniform float msweight;
void main()
{
  
  if (mRenderWireframe) {
    fColor = vec4(0.89, 0.57, 0.15, 1.0);
    return;
  }
  vec3 normal;
  if (mSmoothShading) {
    normal = normalize(iNormal);
  } else {
    normal = normalize(cross(dFdx(position), dFdy(position)));
  }
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

  vec3 color = studioShading(albedo, viewdir, normal, tangent);
  
  fColor = vec4(color*msweight, 1);
  
  if (mNormalCheck) {
      float intensity = clamp((mView * vec4(normal, 0)).z, 0, 1) * 0.4 + 0.6;
      if (gl_FrontFacing) {
        fColor = vec4(0.42 * intensity*msweight, 0.42 * intensity*msweight, 0.93 * intensity*msweight, 1);
      } else {
        fColor = vec4(0.87 * intensity*msweight, 0.22 * intensity*msweight, 0.22 * intensity*msweight, 1);
      }
  }
}
)";
    }

//printf("!!!!%s!!!!\n", frag.c_str());
    return compile_program(vert, frag);
  }
}