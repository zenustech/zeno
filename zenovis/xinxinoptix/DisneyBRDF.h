#pragma once

#include "zxxglslvec.h"
#include "TraceStuff.h"

namespace BRDFBasics{
static __inline__ __device__  float fresnel(float cosT){
    float v = clamp(1-cosT,0.0f,1.0f);
    float v2 = v *v;
    return v2 * v2 * v;
}
static __inline__ __device__ vec3 fresnelSchlick(vec3 r0, float radians)
{
    float exponential = powf( 1.0f - radians, 5.0f);
    return r0 + (vec3(1.0f) - r0) * exponential;
}
static __inline__ __device__ float fresnelSchlick(float r0, float radians)
{
    return mix(1.0f, fresnel(radians), r0);
}
static __inline__ __device__ float SchlickWeight(float u)
{
    float m = clamp(1.0f - u, 0.0f, 1.0f);
    float m2 = m * m;
    return m * m2 * m2;
}
static __inline__ __device__ float fresnelSchlickR0(float eta)
{
    return pow(eta - 1.0f, 2.0f) / pow(eta + 1.0f, 2.0f);
}
static __inline__ __device__ float SchlickDielectic(float cosThetaI, float relativeIor)
{
    float r0 = fresnelSchlickR0(relativeIor);
    return r0 + (1.0f - r0) * SchlickWeight(cosThetaI);
}

static __inline__ __device__ float fresnelDielectric(float cosThetaI, float ni, float nt, bool is_inside)
{
    cosThetaI = clamp(cosThetaI, -1.0f, 1.0f);

    if(is_inside)
    {
        float temp = ni;
        ni = nt;
        nt = temp;
    }

    float sinThetaI = sqrtf(max(0.0f, 1.0f - cosThetaI * cosThetaI));
    float sinThetaT = ni / nt * sinThetaI;

    if(sinThetaT >= 1)
    {
        return 1;
    }

    float cosThetaT = sqrtf(max(0.0f, 1.0f - sinThetaT * sinThetaT));

    float rParallel     = ((nt * cosThetaI) - (ni * cosThetaT)) / ((nt * cosThetaI) + (ni * cosThetaT));
    float rPerpendicuar = ((ni * cosThetaI) - (nt * cosThetaT)) / ((ni * cosThetaI) + (nt * cosThetaT));
    return (rParallel * rParallel + rPerpendicuar * rPerpendicuar) / 2;
}
static __inline__ __device__  float GTR1(float cosT,float a){
    if(a >= 1.0f) return 1/M_PIf;
    float t = (1+(a*a-1)*cosT*cosT);
    return (a*a-1.0f) / (M_PIf*logf(a*a)*t);
}
static __inline__ __device__  float GTR2(float cosT,float a){
    float t = (1+(a*a-1)*cosT*cosT);
    return (a*a) / (M_PIf*t*t);
}
static __inline__ __device__  float GGX(float cosT, float a){
    float a2 = a*a;
    float b = cosT*cosT;
    return 2.0f/ (1.0f + sqrtf(a2 + b - a2*b));
}
static __inline__ __device__  vec3 sampleOnHemisphere(unsigned int &seed, float roughness)
{
    float x = rnd(seed);
    float y = rnd(seed);

    float a = roughness*roughness;

	float phi = 2.0f * M_PIf * x;
	float cosTheta = sqrtf((1.0f - y) / (1.0f + (a*a - 1.0f) * y));
	float sinTheta = sqrtf(1.0f - cosTheta*cosTheta);


    return vec3(cos(phi) * sinTheta,  sin(phi) * sinTheta, cosTheta);
}
static __inline__ __device__ float pdfDiffuse(vec3 wi, vec3 n)
{
    return abs(dot(n, wi)/M_PIf);
}
static __inline__ __device__ float pdfMicrofacet(float NoH, float roughness)
{
    float a2 = roughness * roughness;
    a2 *= a2;
    float cos2Theta = NoH * NoH;
    float denom = cos2Theta * (a2 - 1.) + 1;
    if(denom == 0 ) return 0;
    float pdfDistrib = a2 / (M_PIf * denom * denom);
    return pdfDistrib;
}
static __inline__ __device__ float pdfClearCoat(float NoH, float ccAlpha)
{
    float Dr = GTR1(NoH, ccAlpha);
    return Dr;
}
static __inline__ __device__ 
float ThinTransmissionRoughness(float ior, float roughness)
{
    return clamp((0.65f * ior - 0.35f)*roughness, 0.0f, 1.0f);
}
static __inline__ __device__
void CalculateAnisotropicParams(float roughness, float anisotropic, float &ax, float &ay)
{
    float aspect = sqrtf(1.0f - 0.9f * anisotropic);
    ax = max(0.001f, roughness*roughness / aspect);
    ay = max(0.001f, roughness*roughness * aspect);
}
static __inline__ __device__
vec3 CalculateTint(vec3 baseColor)
{
    float luminance = dot(vec3(0.3f, 0.6f,1.0f), baseColor);
    return luminance>0.0f?baseColor * (1.0f/luminance) : vec3(1.0f);
}
static __inline__ __device__ float  SeparableSmithGGXG1(vec3 w, vec3 wm, float ax, float ay)
{

    if(abs(w.z)<1e-7) {
        return 0.0f;
    }
    float sinTheta = sqrtf(1.0f - w.z * w.z);
    float absTanTheta = abs( sinTheta / w.z);
    float Cos2Phi = (sinTheta == 0.0f)? 1.0f:clamp(w.x / sinTheta, -1.0f, 1.0f);
    Cos2Phi *= Cos2Phi;
    float Sin2Phi = (sinTheta == 0.0f)? 1.0f:clamp(w.y / sinTheta, -1.0f, 1.0f);
    Sin2Phi *= Sin2Phi;
    float a = sqrtf(Cos2Phi * ax * ax + Sin2Phi * ay * ay);
    float a2Tan2Theta = pow(a * absTanTheta, 2.0f);

    float lambda = 0.5f * (-1.0f + sqrtf(1.0f + a2Tan2Theta));
    return 1.0f / (1.0f + lambda);
}
static __inline__ __device__ float GgxAnisotropicD(vec3 wm, float ax, float ay)
{
    float dotHX2 = wm.x * wm.x;
    float dotHY2 = wm.y * wm.y;
    float cos2Theta = wm.z * wm.z;
    float ax2 = ax * ax;
    float ay2 = ay * ay;

    return 1.0f / (M_PIf * ax * ay * powf(dotHX2 / ax2 + dotHY2 / ay2 + cos2Theta, 2.0f));
}

static __inline__ __device__ void GgxVndfAnisotropicPdf(vec3 wi, vec3 wm, vec3 wo, float ax, float ay,
                                   float& forwardPdfW, float& reversePdfW)
{
    float D = GgxAnisotropicD(wm, ax, ay);

    float absDotNL = abs(wi.z);
    float absDotHL = abs(dot(wm, wi));
    float G1v = SeparableSmithGGXG1(wo, wm, ax, ay);
    forwardPdfW = G1v * absDotHL * D / absDotNL;

    float absDotNV = abs(wo.z);
    float absDotHV = abs(dot(wm, wo));
    float G1l = SeparableSmithGGXG1(wi, wm, ax, ay);
    reversePdfW = G1l * absDotHV * D / absDotNV;
}
static __inline__ __device__ 
vec3 SampleGgxVndfAnisotropic(vec3 wo, float ax, float ay, float u1, float u2)
{
    // -- Stretch the view vector so we are sampling as though roughness==1
    vec3 v = normalize(vec3(wo.x * ax, wo.y * ay,  wo.z));

    // -- Build an orthonormal basis with v, t1, and t2
    vec3 t1 = (v.z < 0.9999f) ? normalize(cross(v, vec3(0,0,1))) : vec3(1,0,0);
    vec3 t2 = cross(t1, v);

    // -- Choose a point on a disk with each half of the disk weighted proportionally to its projection onto direction v
    float a = 1.0f / (1.0f + v.z);
    float r = sqrtf(u1);
    float phi = (u2 < a) ? (u2 / a) * M_PIf : M_PIf + (u2 - a) / (1.0f - a) * M_PIf;
    float p1 = r * cos(phi);
    float p2 = r * sin(phi) * ((u2 < a) ? 1.0f : v.z);

    // -- Calculate the normal in this stretched tangent space
    vec3 n = p1 * t1 + p2 * t2 + sqrtf(max(0.0f, 1.0f - p1 * p1 - p2 * p2)) * v;

    // -- unstretch and normalize the normal
    return normalize(vec3(ax * n.x, ay * n.y, n.z));
}
}
namespace DisneyBRDF
{   
static __inline__ __device__ float pdf(
        vec3 baseColor,
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
        vec3 N,
        vec3 T,
        vec3 B,
        vec3 wi,
        vec3 wo)
    {
        vec3 n = N;
        float spAlpha = max(0.001f, roughness);
        float ccAlpha = mix(0.1f, 0.001f, clearcoatGloss);
        float diffRatio = 0.5f*(1.0f - metallic);
        float spRatio = 1.0f - diffRatio;

        vec3 half = normalize(wi + wo);

        float cosTheta = abs(dot(n, half));
        float pdfGTR2 = BRDFBasics::GTR2(cosTheta, spAlpha) * cosTheta;
        float pdfGTR1 = BRDFBasics::GTR1(cosTheta, ccAlpha) * cosTheta;

        float ratio = 1.0f/(1.0f + clearcoat);
        float pdfSpec = mix(pdfGTR1, pdfGTR2, ratio)/(4.0f * abs(dot(wo, half)));
        float pdfDiff = abs(dot(wi, n)) * (1.0f/M_PIf);

        return diffRatio * pdfDiff + spRatio * pdfSpec;
    }

static __inline__ __device__ vec3 sample_f(
        unsigned int &seed, 
        vec3 baseColor,
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
        vec3 N,
        vec3 T,
        vec3 B,
        vec3 wo,
        float &is_refl)
    {
        
        float ratiodiffuse = (1.0f - metallic)/2.0f;
        float p = rnd(seed);
        
        Onb tbn = Onb(N);
        
        vec3 wi;
        
        if( p < ratiodiffuse){
            //sample diffuse lobe
            
            vec3 P = BRDFBasics::sampleOnHemisphere(seed, 1.0f);
            wi = P;
            tbn.inverse_transform(wi);
            wi = normalize(wi);
            is_refl = 0;
        }else{
            //sample specular lobe.
            float a = max(0.001f, roughness);
            
            vec3 P = BRDFBasics::sampleOnHemisphere(seed, a*a);
            vec3 half = normalize(P);
            tbn.inverse_transform(half);            
            wi = half* 2.0f* dot(normalize(wo), half) - normalize(wo); //reflection vector
            wi = normalize(wi);
            is_refl = 1;
        }
        
        return wi;
    }
static __inline__ __device__ vec3 eval(
        vec3 baseColor,
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
        vec3 N,
        vec3 T,
        vec3 B,
        vec3 wi,
        vec3 wo)
    {
        vec3 wh = normalize(wi+ wo);
        float ndoth = dot(N, wh);
        float ndotwi = dot(N, wi);
        float ndotwo = dot(N, wo);
        float widoth = dot(wi, wh);

        if(ndotwi <=0 || ndotwo <=0 )
            return vec3(0,0,0);

        vec3 Cdlin = baseColor;
        float Cdlum = 0.3f*Cdlin.x + 0.6f*Cdlin.y + 0.1f*Cdlin.z;

        vec3 Ctint = Cdlum > 0.0f ? Cdlin / Cdlum : vec3(1.0f,1.0f,1.0f);
        vec3 Cspec0 = mix(specular*0.08f*mix(vec3(1,1,1), Ctint, specularTint), Cdlin, metallic);
        vec3 Csheen = mix(vec3(1.0f,1.0f,1.0f), Ctint, sheenTint);

        //diffuse
        float Fd90 = 0.5f + 2.0f * ndoth * ndoth * roughness;
        float Fi = BRDFBasics::fresnel(ndotwi);
        float Fo = BRDFBasics::fresnel(ndotwo);
        
        float Fd = (1 +(Fd90-1)*Fi)*(1+(Fd90-1)*Fo);

        float Fss90 = widoth*widoth*roughness;
        float Fss = mix(1.0f, Fss90, Fi) * mix(1.0f,Fss90, Fo);
        float ss = 1.25f * (Fss *(1.0f / (ndotwi + ndotwo) - 0.5f) + 0.5f);

        float a = max(0.001, roughness);
        float Ds = BRDFBasics::GTR2(ndoth, a);
        float Dc = BRDFBasics::GTR1(ndoth, mix(0.1f, 0.001f, clearcoatGloss));

        float roughg = sqrtf(roughness*0.5f + 0.5f);
        float Gs = BRDFBasics::GGX(ndotwo, roughness) * BRDFBasics::GGX(ndotwi, roughness);

        float Gc = BRDFBasics::GGX(ndotwo, 0.25) * BRDFBasics::GGX(ndotwi, 0.25f);

        float Fh = BRDFBasics::fresnel(widoth);
        vec3 Fs = mix(Cspec0, vec3(1.0f,1.0f,1.0f), Fh);
        float Fc = mix(0.04f, 1.0f, Fh);

        vec3 Fsheen = Fh * sheen * Csheen;

        return ((1/M_PIf) * mix(Fd, ss, subsurface) * Cdlin + Fsheen) * (1.0f - metallic)
        + Gs*Fs*Ds + 0.25f*clearcoat*Gc*Fc*Dc;
    }
}




