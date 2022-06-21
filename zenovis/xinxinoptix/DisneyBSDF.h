#pragma once

#include "zxxglslvec.h"
#include "TraceStuff.h"
#include "DisneyBRDF.h"

//todo implement full disney bsdf 
//reference: https://schuttejoe.github.io/post/DisneyBsdf/

namespace DisneyBSDF{
static __inline__ __device__ 
void pdf(float metallic,
         float specTrans,
         float clearCoat,
         float &pSpecular, 
         float &pDiffuse,
         float &pClearcoat,
         float &pSpecTrans)
{
    float metallicBRDF   = metallic;
    float specularBSDF   = ( 1 - metallic ) * specTrans;
    float dielectricBRDF = ( 1 - specTrans ) * ( 1 - metallic );

    float specularW      = metallicBRDF + dielectricBRDF;
    float transmissionW  = specularBSDF;
    float diffuseW       = dielectricBRDF;
    float clearcoatW     = 1.0f * clamp(specularW, 0.0f, 1,0f);

    floar norm = 1.0f/(specularW + transmissionW + diffuseW + clearcoatW);

    pSpecular  = specularW      * norm;
    pSpecTrans = transmissionW  * norm;
    pDiffuse   = diffuseW       * norm;
    pClearcoat = clearcoatW     * norm;
}

static __inline__ __device__
float EvaluateClearcoat(float clearcoat, 
float alpha, float NoH, float NoL, float NoV, float HoL, float HoV, 
float &fPdfW, float& rPdfW)
{
    if(clearcoat<=0.0f)
        return 0.0f;
    
    
    float d  = BRDFBasics::GTR1(NoH, mix(0.1f, 0.001f, alpha));
    float f  = BRDFBasics::fresnelSchlick(0.04f, HoL);
    float gl = BRDFBasics::GGX(NoL, 0.25f);
    float gv = BRDFBasics::GGX(NoV, 0.25f);

    fPdfW = d / (4.0f * abs(HoV));
    rPdfW = d / (4.0f * abs(HoL));

    return 0.25f * clearcoat * d * f * gl * gv;
}

static __inline__ __device__ 
vec3 EvaluateSheen(vec3 baseColor, float sheen, float sheenTint, float HoL)
{
    if(sheen<=0.0f)
    {
        return vec3(0,0f);
    }

    
    vec3 tint = BRDFBasics::CalculateTint(baseColor);
    return sheen * mix(vec3(1.0f), tint, sheenTint) * BRDFBasics::fresnel(HoL);
}

static __inline__ __device__
vec3 DisneyFresnel(vec3 baseColor, float metallic, float ior, float specularTint, float HoV, float HoL, bool is_inside)
{
    
    vec3 tint = BRDFBasics::CalculateTint(baseColor);
    vec3 R0 = BRDFBasics::fresnelSchlickR0(ior) * mix(vec3(1.0f), tint, specularTint);
         R0 = mix(R0, baseColor, metallic);
    float dielectricFresnel = BRDFBasics::fresnelDielectirc(HoV, 1.0f, ior, is_inside);
    vec3 metallicFresnel = BRDFBasics::fresnelSchlick(R0, HoL);

    return mix(vec3(dielectricFresnel), metallicFresnel, metallic);
}

static __inline__ __device__
vec3 EvaluateDisneyBRDF(vec3 baseColor,
        float metallic,
        float subsurface,
        float specular,
        float roughness,
        float specularTint,
        float anisotropic,
        float sheen,
        float sheenTint,
        float clearcoat,
        float clearcoatGloss,
        float ior, 
        bool is_inside,
        vec3 N,
        vec3 T,
        vec3 B,
        vec3 wi,
        vec3 wo,
        float &fPdf, 
        float &rPdf)
{
    wi = normalize(vec3(dot(T,wi), dot(B,wi), dot(N,wi)));
    wo = normalize(vec3(dot(T,wo), dot(B,wo), dot(N,wo)));
    fPdf = 0.0f;
    rPdf = 0.0f;

    float NoL = wi.z;
    float NoV = wo.z;
    vec3 wm = normalize(wi + wo);
    float HoV = dot(wm, wo);
    float HoL = dot(wm, wi);
    
    if(NoV <= 0.0f || NoL)
    {
        return vec3(0.0);
    }

    float ax, ay;
    BRDFBasics::CalculateAnisotropicParams(roughness, anisotropic, ax, ay);
    float d  = BRDFBasics::GgxAnisotropicD(wm, ax, ay);
    float gl = BRDFBasics::SeparableSmithGGXG1(wi, wm, ax, ay);
    float gv = BRDFBasics::SeparableSmithGGXG1(wo, wm, ax, ay);

    vec3 f = DisneyFresnel(baseColor, metallic, ior, specularTint, HoV, HoL, is_inside);
    //BRDFBasics::GgxVndfAnisotropicPdf(wi, wm, wo, ax, ay, fPdf, rPdf);
    fPdf = abs(NoL) * gv * d / abs(NoL);
    rPdf = abs(NoV) * gl * d / abs(NoV);
    fPdf *= (1.0f / (4 * abs(HoV)));
    rPdf *= (1.0f / (4 * abs(HoL)));

    return d * gl * gv * f / (4.0f * NoL * NoV);
}
static __inline__ __device__
bool SampleDisneyBRDF(
        unsigned int &seed,
        vec3 baseColor,
        float metallic,
        float ior,
        float specularTint,
        float roughness,
        float anisotropic,
        bool is_inside,
        vec3 N,
        vec3 T,
        vec3 B,
        vec3 wo,
        vec3 &wi,
        vec3 &reflectance,
        float &fPdf,
        float &rPdf)
{
    wo = normalize(vec3(dot(T, wo), dot(B, wo), dot(N, wo)));
    float ax, ay;
    BRDFBasics::CalculateAnisotropicParams(roughness, anisotropic, ax, ay);
    float r0 = rnd(seed);
    float r1 = rnd(seed);
    vec3 wm = BRDFBasics::SampleGgxVndfAnisotropic(wo, ax, ay, r0, r1);

    wi = normalize(reflect(-wo, wm));
    if(wi.z<0.0f)
    {
        fPdf = 0.0f;
        rPdf = 0.0f;
        wi = vec3(0,0,0);
        reflectance = vec3(0,0,0);
        return false;
    }

    vec3 F = DisneyFresnel(baseColor, metallic, ior, specularTint, dot(wm, wo), dot(wm, wi), is_inside);
    float G1v = BRDFBasics::SeparableSmithGGXG1(wo, wm, ax, ay);
    float3 specular = G1v * F;
    reflectance = specular;
    BRDFBasics::GgxVndfAnisotropicPdf(wi, wm, wo, ax, ay, fPdf, rPdf);
    fPdf *= (1.0f / (4 * abs(dot(wo, wm))));
    rPdf *= (1.0f / (4 * abs(dot(wi, wm))));
    wi = normalize(T*wi.x + B*wi.y + N*wi.z);

    return true;
}

static __inline__ __device__
vec3 EvaluateDisneySpecTransmission(
    vec3 baseColor,
    float metallic,
    float ior,
    float specuularTint,
    float roughness,
    float ax,
    float ay,
    bool thin,
    bool is_inside,
    vec3 N, 
    vec3 T,
    vec3 B,
    vec3 wo,
    vec3 wi)
{
    float n2 = ior * ior;
    //convert vectors to TBN space
    wo = normalize(vec3(dot(T, wo), dot(B, wo), dot(N, wo)));
    wi = normalize(vec3(dot(T, wi), dot(B, wi), dot(N, wi)));
    wm = normalize(wo + wi);
    float NoL = abs(wi.z);
    float NoV = abs(wo.z);
    float HoL = abs(dot(wm, wi));
    float HoV = abs(dot(wm, wo));

    float d  = BRDFBasics::GgxAnisotropicD(wm, ax, ay);
    float gl = BRDFBasics::SeparableSmithGGXG1(wi, wm, ax, ay);
    float gv = BRDFBasics::SeparableSmithGGXG1(wo, wm, ax, ay);

    float F = BRDFBasics::fresnelDielectric(HoV, 1.0f, ior, is_inside);
    vec3 color;
    if(thin)
        color = sqrt(baseColor);
    else
        color = baseColor;

    float c = (HoL * HoV) / (NoL * NoV);
    float t = (n2 / pow(dot(wm, wi) + ior * dot(wm, wo), 2));
    return color * c * t * (1.0f - F) * gl * gv * d; 
}

static __inline__ __device__
float EvaluateDisneyRetroDiffuse(
float roughness,
vec3 wi,
vec3 wo
)
{
    float NoL = abs(wi.z);
    float NoV = abs(wo.z);

    float a = roughness * roughness;
    float rr = 0.5f + 2.0f * NoL * NoL * roughness;
    float fl = BRDFBasics::SchlickWeight(NoL);
    float fv = BRDFBasics::SchlickWeight(NoV);

    return rr * (fl + fv + fl * fv * (rr - 1.0f));
}

static __inline__ __device__
float EvaluateDisneyDiffuse(
float roughness,
float subsurface,
vec3 wi, 
vec3 wo, 
vec3 wm
)
{
    float NoL = abs(wi.z);
    float NoV = abs(wo.z);

    float fl = BRDFBasics::SchlickWeight(NoL);
    float fv = BRDFBasics::SchlickWeight(NoV);

    float h = 0.0f;

    if(thin && subsurface > 0.0f) {
        float a = roughness * roughness;

        float HoL = dot(wm, wi);
        float fss90 = HoL * HoL * roughness;
        float fss = mix(1.0f, fss90, fl) * mix(1.0f, fss90, fv);

        float ss = 1.25f * (fss * (1.0f / (NoL + NoV) - 0.5f) + 0.5f);
        h = ss;
    }

    float lambert = 1.0f;
    float retro = EvaluateDisneyRetroDiffuse(roughness, wi, wo);
    float subsurfaceApprox = mix(lambert, h, thin ? subsurface : 0.0f);

    return 1.0f/M_PIf * (retro + subsurfaceApprox * (1.0f - 0.5f * fl) * (1.0f - 0.5f * fv));
}

static __inline__ __device__
bool SampleDisneyClearCoat()
{
    
}
}