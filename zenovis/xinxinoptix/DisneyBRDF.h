#pragma once

#include "zxxglslvec.h"

namespace BRDFBasics{
static __device__  float fresnel(float cosT){
    float v = clamp(1-cosT,0.0f,1.0f);
    float v2 = v *v;
    return v2 * v2 * v;
}
static __device__  float GTR1(float cosT,float a){
    if(a >= 1.0f) return 1/M_PIf;
    float t = (1+(a*a-1)*cosT*cosT);
    return (a*a-1.0f) / (M_PIf*logf(a*a)*t);
}
static __device__  float GTR2(float cosT,float a){
    float t = (1+(a*a-1)*cosT*cosT);
    return (a*a) / (M_PIf*t*t);
}
static __device__  float GGX(float cosT, float a){
    float a2 = a*a;
    float b = cosT*cosT;
    return 1.0f/ (cosT + sqrtf(a2 + b - a2*b));
}
static __device__  float3 sampleOnHemisphere(unsigned int &seed, float roughness)
{
    float x = rnd(seed);
    float y = rnd(seed);

    float a = roughness*roughness;

	float phi = 2.0f * M_PIf * x;
	float cosTheta = sqrtf((1.0f - y) / (1.0f + (a*a - 1.0f) * y));
	float sinTheta = sqrtf(1.0f - cosTheta*cosTheta);


    return make_float3(cos(phi) * sinTheta,  sin(phi) * sinTheta, cosTheta);
}
};
namespace DisneyBRDF
{   
static __device__ float pdf(
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
        float pdfSpec = glsl::mix(pdfGTR1, pdfGTR2, ratio)/(4.0f * abs(dot(wi, half)));
        float pdfDiff = abs(dot(wi, n)) * (1.0f/M_PIf);

        return diffRatio * pdfDiff + spRatio * pdfSpec;
    }

static __device__ vec3 sample_f(
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
static __device__ vec3 eval(
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
            return make_vec3(0,0,0);

        vec3 Cdlin = baseColor;
        float Cdlum = 0.3f*Cdlin.x + 0.6f*Cdlin.y + 0.1f*Cdlin.z;

        vec3 Ctint = Cdlum > 0.0f ? Cdlin / Cdlum : make_vec3(1.0f,1.0f,1.0f);
        vec3 Cspec0 = mix(specular*0.08f*mix(make_vec3(1,1,1), Ctint, specularTint), Cdlin, metallic);
        vec3 Csheen = mix(make_vec3(1.0f,1.0f,1.0f), Ctint, sheenTint);

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
        float Gs = BRDFBasics::GGX(ndotwo, roughg) * BRDFBasics::GGX(ndotwi, roughg);

        float Gc = BRDFBasics::GGX(ndotwo, 0.25) * BRDFBasics::GGX(ndotwi, 0.25f);

        float Fh = BRDFBasics::fresnel(widoth);
        vec3 Fs = mix(Cspec0, make_vec3(1.0f,1.0f,1.0f), Fh);
        float Fc = mix(0.04f, 1.0f, Fh);

        vec3 Fsheen = Fh * sheen * Csheen;

        return ((1/M_PIf) * mix(Fd, ss, subsurface) * Cdlin + Fsheen) * (1.0f - metallic)
        + Gs*Fs*Ds + 0.25f*clearcoat*Gc*Fc*Dc;
    }
};

//////////////////////////////////////////
///here inject common code in glsl style
static __device__ vec3 perlin_hash22(vec3 p)
{
    p = vec3( dot(p,vec3(127.1f,311.7f,284.4f)),
              dot(p,vec3(269.5f,183.3f,162.2f)),
	      	  dot(p,vec3(228.3f,164.9f,126.0f)));
    return -1.0f + 2.0f * fract(sin(p)*43758.5453123f);
}

static __device__ float perlin_lev1(vec3 p)
{
    vec3 pi = vec3(floor(p));
    vec3 pf = p - pi;
    vec3 w = pf * pf * (3.0f - 2.0f * pf);
    return .08f + .8f * (mix(
			            mix(
                            mix(
                            dot(perlin_hash22(pi + 0), pf - 0),
                            dot(perlin_hash22(pi + 0), pf - 0),
                            w.x),
                            mix(
                            dot(perlin_hash22(pi + vec3(0, 1, 0)), pf - vec3(0, 1, 0)),
                            dot(perlin_hash22(pi + vec3(1, 1, 0)), pf - vec3(1, 1, 0)),
                            w.x),
				        w.y),
			            mix(
				            mix(
                            dot(perlin_hash22(pi + vec3(0, 0, 1)), pf - vec3(0, 0, 1)),
                            dot(perlin_hash22(pi + vec3(1, 0, 1)), pf - vec3(1, 0, 1)),
                            w.x),
				            mix(
                            dot(perlin_hash22(pi + vec3(0, 1, 1)), pf - vec3(0, 1, 1)),
                            dot(perlin_hash22(pi + vec3(1, 1, 1)), pf - vec3(1, 1, 1)),
                            w.x),
				        w.y),
			          w.z));
}

static __device__ float perlin(float p,int n,vec3 a)
{
    float total = 0;
    for(int i=0; i<n; i++)
    {
        float frequency = pow(2.0f,i*1.0f);
        float amplitude = pow(p,i*1.0f);
        total = total + perlin_lev1(a * frequency) * amplitude;
    }

    return total;
}

///end example of common code injection in glsl style





