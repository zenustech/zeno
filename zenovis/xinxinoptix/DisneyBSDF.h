#pragma once
#include "zxxglslvec.h"
#include "TraceStuff.h"


#include "DisneyBRDF.h"




//todo implement full disney bsdf 
//reference: https://schuttejoe.github.io/post/DisneyBsdf/
//list of component:
//Sheen
//Clearcoat
//Specular BRDF
//Specular BSDF
//Diffuse BRDF
//
//params
//
//vec3 baseColor,
//float metallic,
//float subsurface,
//float specular,
//float roughness,
//float specularTint,
//float anisotropic,
//float sheen,
//float sheenTint,
//float clearCoat,
//float clearcoatGloss,

//vec3 transmiianceColor
//float flatness
//float specTrans,
//float scatterDistance,
//float ior,

namespace DisneyBSDF{
    enum SurfaceEventFlags{
        scatterEvent = 0x01,
        transmissionEvent = 0x02,
        diracEvent = 0x04
    };

    enum PhaseFunctions{
        vacuum,
        isotropic,
        decideLate // Unknow while hitting from inside. 
    };
    static __inline__ __device__ 
    float bssrdf_dipole_compute_Rd(float a, float fourthirdA)
    {
        float s = sqrt(max(3.0f * (1.0f - a), 0.0f));
        return 0.5f * a * (1.0f + exp(-fourthirdA * s)) * exp(-s);
    }
    static __inline__ __device__
    float bssrdf_dipole_compute_alpha_prime(float rd, float fourthirdA)
    {
        /* Little Newton solver. */
        if (rd < 1e-4f) {
            return 0.0f;
        }
        if (rd >= 0.995f) {
            return 0.999999f;
        }

        float x0 = 0.0f;
        float x1 = 1.0f;
        float xmid, fmid;

        constexpr const int max_num_iterations = 12;
        for (int i = 0; i < max_num_iterations; ++i) {
            xmid = 0.5f * (x0 + x1);
            fmid = bssrdf_dipole_compute_Rd(xmid, fourthirdA);
            if (fmid < rd) {
                x0 = xmid;
            }
            else {
                x1 = xmid;
            }
        }

        return xmid;
    }
    static __inline__ __device__ void
    setup_subsurface_radius(float eta, vec3 albedo, vec3 &radius)
    {
        float inv_eta = 1.0f/eta;
        float F_dr = inv_eta * (-1.440f * inv_eta + 0.710f) + 0.668f + 0.0636f * eta;
        float fourthirdA = (4.0f / 3.0f) * (1.0f + F_dr) /
                             (1.0f - F_dr + 1e-7f ); /* From Jensen's `Fdr` ratio formula. */
        vec3 alpha_prime;
        alpha_prime.x = bssrdf_dipole_compute_alpha_prime(albedo.x, fourthirdA);
        alpha_prime.y = bssrdf_dipole_compute_alpha_prime(albedo.y, fourthirdA);
        alpha_prime.z = bssrdf_dipole_compute_alpha_prime(albedo.z, fourthirdA);
        radius = radius * sqrt(3.0f * abs(vec3(1.0) - alpha_prime));

    }
    static __inline__ __device__ void 
    subsurface_random_walk_remap(const float albedo,
                                 const float radius,
                                 float g,
                                 float &sigma_t,
                                 float &alpha)
    {
        /* Compute attenuation and scattering coefficients from albedo. */
        float g2 = g * g;
        float g3 = g2 * g;
        float g4 = g3 * g;
        float g5 = g4 * g;
        float g6 = g5 * g;
        float g7 = g6 * g;

        float A = 1.8260523782f + -1.28451056436f * g + -1.79904629312f * g2 +
                  9.19393289202f * g3 + -22.8215585862f * g4 + 32.0234874259f * g5 +
                  -23.6264803333f * g6 + 7.21067002658f * g7;
        float B = 4.98511194385f +
                  0.127355959438f *
                      exp(31.1491581433f * g + -201.847017512f * g2 + 841.576016723f * g3 +
                          -2018.09288505f * g4 + 2731.71560286f * g5 + -1935.41424244f * g6 +
                          559.009054474f * g7);
        float C = 1.09686102424f + -0.394704063468f * g + 1.05258115941f * g2 +
                  -8.83963712726f * g3 + 28.8643230661f * g4 + -46.8802913581f * g5 +
                  38.5402837518f * g6 + -12.7181042538f * g7;
        float D = 0.496310210422f + 0.360146581622f * g + -2.15139309747f * g2 +
                  17.8896899217f * g3 + -55.2984010333f * g4 + 82.065982243f * g5 +
                  -58.5106008578f * g6 + 15.8478295021f * g7;
        float E = 4.23190299701f +
                  0.00310603949088f *
                      exp(76.7316253952f * g + -594.356773233f * g2 + 2448.8834203f * g3 +
                          -5576.68528998f * g4 + 7116.60171912f * g5 + -4763.54467887f * g6 +
                          1303.5318055f * g7);
        float F = 2.40602999408f + -2.51814844609f * g + 9.18494908356f * g2 +
                  -79.2191708682f * g3 + 259.082868209f * g4 + -403.613804597f * g5 +
                  302.85712436f * g6 + -87.4370473567f * g7;

        float blend = pow(albedo, 0.25f);

        alpha = (1.0f - blend) * A * powf(atanf(B * albedo), C) +
                blend * D * powf(atanf(E * albedo), F);
        alpha = clamp(alpha, 0.0f, 0.999999f);  // because of numerical precision

        float sigma_t_prime = 1.0f / fmaxf(radius, 1e-16f);
        sigma_t = sigma_t_prime / (1.0f - g);
    }

    static __inline__ __device__
    void CalculateExtinction2(vec3 albedo, vec3 radius, vec3 &sigma_t, vec3 &alpha)
    {
        vec3 r = radius;
        setup_subsurface_radius(3.0, albedo, r);
        subsurface_random_walk_remap(albedo.x, r.x, 0, sigma_t.x, alpha.x);
        subsurface_random_walk_remap(albedo.y, r.y, 0, sigma_t.y, alpha.y);
        subsurface_random_walk_remap(albedo.z, r.z, 0, sigma_t.z, alpha.z);
        //sigma_s = sigma_t * alpha;
    }

    static __inline__ __device__
    int volume_sample_channel(vec3 a, float rand, vec3 &pdf)
    {
        /* Sample color channel proportional to throughput and single scattering
        * albedo, to significantly reduce noise with many bounce, following:
        *
        * "Practical and Controllable Subsurface Scattering for Production Path
        *  Tracing". Matt Jen-Yuan Chiang, Peter Kutz, Brent Burley. SIGGRAPH 2016. */
        vec3 weights = abs(a);
        float sum_weights = dot(weights, vec3(1,1,1));

        if (sum_weights > 0.0f) {
            pdf = weights / sum_weights;
        }
        else {
            pdf = vec3(1.0f / 3.0f);
        }

        float pdf_sum = 0.0f;
    
        pdf_sum += pdf.x;
        if (rand < pdf_sum) {
            return 0;
        }
        pdf_sum += pdf.y;
        if (rand < pdf_sum)
        {
            return 1;
        }
    
        return 2;
    }

    static __inline__ __device__ 
    vec3 CalculateExtinction(vec3 apparantColor, float scatterDistance)
    {
        return 1.0/(max(apparantColor * scatterDistance,vec3(0.000001)));
    }
    
    static __inline__ __device__
    float SampleDistance2(unsigned int &seed, vec3 a/*throughput * alpha*/, const vec3& sigma_t, vec3& channelPDF)
    {
        float r0 = rnd(seed);
        int channel = volume_sample_channel(a, r0, channelPDF);
        channel = clamp(channel, 0, 2);
        float c = sigma_t[channel];
        
        float s = -log(max(1.0f-rnd(seed), _FLT_MIN_)) / max(c, 1e-5f);
        return s;
    }

    static __inline__ __device__
    void world2local(vec3& v, vec3 T, vec3 B, vec3 N){
        v = normalize(vec3(dot(T,v), dot(B,v), dot(N,v)));
    }

    static __inline__ __device__
    void rotateTangent(vec3& _T, vec3& _B, vec3 N, float rotInRadian) {
        vec3 T = normalize(cos(rotInRadian) * _T - sin(rotInRadian) * _B);
        vec3 B = normalize(sin(rotInRadian) * _T + cos(rotInRadian) * _B);
        _T = T;
        _B = B;
    }

    static __inline__ __device__ 
    void pdf(
        float metallic,
        float specTrans,
        float clearCoat,
        float &pSpecular, 
        float &pDiffuse,
        float &pClearcoat,
        float &pSpecTrans,
        float &totalp)
    {
        float metallicBRDF   = metallic;
        float specularBSDF   = ( 1.0f - metallic ) * specTrans;
        float dielectricBRDF = ( 1.0f - specTrans ) * ( 1.0f - metallic );
        float specularW      = metallicBRDF + dielectricBRDF;
        float transmissionW  = specularBSDF;
        float diffuseW       = dielectricBRDF;
        float clearcoatW     = clearCoat;

        float norm = 1.0f/(specularW + transmissionW + diffuseW + clearcoatW);
        totalp = (specularW + transmissionW + diffuseW + clearcoatW);

        pSpecular  = specularW      * norm;
        pSpecTrans = transmissionW  * norm;
        pDiffuse   = diffuseW       * norm;
        pClearcoat = clearcoatW     * norm;
    }
    static __inline__ __device__
    float D_Charlie (float NH, float roughness)
    {
      float invR = 1.0f / roughness;
      float cos2h = NH * NH;
      float sin2h = 1 - cos2h;
      return (2.0f + invR) * pow(sin2h, invR * 0.5f) / 2.0f / M_PIf;
    }
    static __inline__ __device__
    float CharlieL (float x, float r)
    {
      r = clamp(r,0.0f,1.0f);
      r = 1 - (1 - r) * (1 - r);
      float a = mix(25.3245, 21.5473, r);
      float b = mix(3.32435, 3.82987, r);
      float c = mix(0.16801, 0.19823, r);
      float d = mix(-1.27393, -1.97760, r);
      float e = mix(-4.85967, -4.32054, r);
      return a / (1 + b * pow(x, c)) + d * x + e;
    }
    static __inline__ __device__
    float V_Charlie (float NL, float NV, float roughness)
    {
      float lambdaV = NV < 0.5 ? exp(CharlieL(NV, roughness)) : exp(2 * CharlieL(0.5, roughness) - CharlieL(1 - NV, roughness));
      float lambdaL = NL < 0.5 ? exp(CharlieL(NL, roughness)) : exp(2 * CharlieL(0.5, roughness) - CharlieL(1 - NL, roughness));
      return 1 / ((1 + lambdaV + lambdaL) * (4 * NV * NL));
    }


    static __inline__ __device__ 
    vec3 EvaluateSheen(vec3 baseColor, float sheen, float sheenTint, float HoL)
    {
        if(sheen<=0.0f)
        {
            return vec3(0.0f);
        }
        vec3 tint = BRDFBasics::CalculateTint(baseColor);
        return sheen * mix(vec3(1.0f), tint, sheenTint) ;
    }

    static __inline__ __device__
    float EvaluateClearcoat(
        float clearcoat, 
        float alpha,
        float clearCoatRoughness,
        float clearCoatIOR,
        float NoH,
        float NoL,
        float NoV,
        float HoL,
        float HoV,
        float& fPdfW, 
        float& rPdfW)
    {
        if(clearcoat<=0.0f){
            return 0.0f;
        }
        float d  = BRDFBasics::GTR1(NoH, mix(0.1f, 0.001f, alpha));
        d = clamp(d, 0.0f, 1.0f);
        float f  = BRDFBasics::fresnelDielectric(HoL, 1.0f, clearCoatIOR, false);
        float gl = BRDFBasics::GGX(NoL, clearCoatRoughness);
        float gv = BRDFBasics::GGX(NoV, clearCoatRoughness);

//        fPdfW = d / (4.0f * abs(HoV));
//        rPdfW = d / (4.0f * abs(HoL));

        return 0.25f * clearcoat * d * f * gl * gv ;
    }

    static __inline__ __device__
    vec3 DisneyFresnel(
            vec3 baseColor,
            float metallic,
            float ior,
            float specularTint,
            float HoV,
            float HoL,
            bool is_inside)
    {
        vec3 tint = BRDFBasics::CalculateTint(baseColor);
        vec3 R0 = BRDFBasics::fresnelSchlickR0(ior) * mix(vec3(1.0f), tint, specularTint);
             R0 = mix(R0, baseColor, metallic);
        float dielectricFresnel = BRDFBasics::fresnelDielectric(HoV, 1.0f, ior, is_inside);
        vec3 metallicFresnel = BRDFBasics::fresnelSchlick(R0, HoL);

        return mix(vec3(dielectricFresnel), metallicFresnel, metallic);
    }

    static __inline__ __device__
    vec3 EvaluateDisneyBRDF(
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
            float ior, 
            bool is_inside,
            vec3 wi,
            vec3 wo,
            float &fPdf, 
            float &rPdf)
    {
        fPdf = 0.0f;
        rPdf = 0.0f;

        float NoL = wi.z;
        float NoV = wo.z;

        if(NoV <= 0.0f || NoL <= 0.0f)
        {
            return vec3(0.0);
        }

        vec3 wm = normalize(wi + wo);
        float HoV = dot(wm, wo);
        //float HoL = dot(wm, wi);
        float HoL = HoV;

        float ax, ay;
        BRDFBasics::CalculateAnisotropicParams(roughness, anisotropic, ax, ay);
        float d  = BRDFBasics::GgxD(wm, ax, ay);
        d = clamp(d, 0.0f, 1.0f);

        //float gl = BRDFBasics::SeparableSmithGGXG1(wi, wm, ax, ay);
        //float gv = BRDFBasics::SeparableSmithGGXG1(wo, wm, ax, ay);
        float g = BRDFBasics::GgxG(wo, wi, ax, ay);
        vec3 f = DisneyFresnel(baseColor, metallic, ior, specularTint, HoV, HoL, is_inside);
        //BRDFBasics::GgxVndfAnisotropicPdf(wi, wm, wo, ax, ay, fPdf, rPdf);
        //fPdf = abs(NoL) * gv * d / abs(NoL);
        //rPdf = abs(NoV) * gl * d / abs(NoV);
        //fPdf *= (1.0f / (4 * abs(HoV)));
        //rPdf *= (1.0f / (4 * abs(HoL)));
        return d * g * f  / (4.0f * NoL * NoV) ;
        //return d * g * f / (4.0f * NoL * NoV) ;
    }


    static __inline__ __device__
    vec3 EvaluateDisneySpecTransmission(
        vec3 baseColor,
        float metallic,
        float ior,
        float specularTint,
        float roughness,
        float ax,
        float ay,
        bool thin,
        bool is_inside,
        vec3 wo,
        vec3 wi)
    {
        //shall the end result shall consider both reflectio and refraction term
        vec3 reflectance;
        if(wo.z * wi.z>0)//on same side
        {
            //this is a light in the reflection direction
            vec3 wm = normalize(wi + wo);
            float NoL = abs(wi.z);
            float NoV = abs(wo.z);
            float HoL = abs(dot(wm, wi));
            float HoV = abs(dot(wm, wo));
            float d = BRDFBasics::GgxD(wm, ax, ay);
            d = clamp(d, 0.0f, 1.0f);
            float g = BRDFBasics::GgxG(wo, wi, ax, ay);
            float F = BRDFBasics::fresnelDielectric(dot(wm, wo), 1.0f, ior, false);
            vec3 color;
            if(thin)
                color = sqrt(baseColor);
            else
                color = baseColor;

            reflectance = (F * g * d * color  )/ (4.0f * abs(wi.z) * abs(wo.z));
        }
        else {
            float n2 = ior * ior;

            vec3 wm = normalize(wi + ior*wo);

            float NoL = abs(wi.z);
            float NoV = abs(wo.z);
            float HoL = abs(dot(wm, wi));
            float HoV = abs(dot(wm, wo));

            //float d  = BRDFBasics::GgxAnisotropicD(wm, ax, ay);
            float d = BRDFBasics::GgxD(wm, ax, ay);
            d = clamp(d, 0.0f, 1.0f);
            float g = BRDFBasics::GgxG(wo, wi, ax, ay);
            //float gl = BRDFBasics::SeparableSmithGGXG1(wi, wm, ax, ay);
            //float gv = BRDFBasics::SeparableSmithGGXG1(wo, wm, ax, ay);


            float F = BRDFBasics::fresnelDielectric(dot(wm, wo), 1.0f, ior, false);
            vec3 color;
            if(thin)
                color = sqrt(baseColor);
            else
                color = baseColor;

            //float c = (HoL * HoV) / (NoL * NoV + 1e-7);
            float t = (n2 / pow(dot(wm, wi) + ior * dot(wm, wo), 2.0f));
            //if(length(wm) < 1e-5){
            //    return color * (1.0f - F);
            //}
            //+ color *  t *  (1.0f - F) * g * d * 4.0f * HoL * HoV
            reflectance = ( color *  t *  (1.0f - F) * g * d * 4.0f * HoL * HoV ) / (4.0f * abs(wi.z) * abs(wo.z));
        }
        return reflectance;
//        float n2 = ior * ior;
//
//        vec3 wm = normalize(wi + ior*wo);
//
//        float NoL = abs(wi.z);
//        float NoV = abs(wo.z);
//        float HoL = abs(dot(wm, wi));
//        float HoV = abs(dot(wm, wo));
//
//        float d  = BRDFBasics::GgxAnisotropicD(wm, ax, ay);
//
//
//        float gl = BRDFBasics::SeparableSmithGGXG1(wi, wm, ax, ay);
//        float gv = BRDFBasics::SeparableSmithGGXG1(wo, wm, ax, ay);
//
//
//        float F = BRDFBasics::fresnelDielectric(dot(wm, wo), 1.0f, ior, false);
//        vec3 color;
//        if(thin)
//            color = sqrt(baseColor);
//        else
//            color = baseColor;
//
//        float c = (HoL * HoV) / (NoL * NoV + 1e-7);
//        float t = (n2 / pow(dot(wm, wi) + ior * dot(wm, wo), 2.0f));
//        //if(length(wm) < 1e-5){
//        //    return color * (1.0f - F);
//        //}
//
//        return color * c * t *  (1.0f - F) * gl * gv * d;
    }

    static __inline__ __device__
    float EvaluateDisneyRetroDiffuse(
        float roughness,
        vec3 wi,
        vec3 wo
        )
    {
        // float NoL = abs(wi.z);
        // float NoV = abs(wo.z);

        // float a = roughness * roughness;
        // float rr = 0.5f + 2.0f * NoL * NoL * roughness;
        // float fl = clamp(BRDFBasics::SchlickWeight(NoL),0.0f,1.0f);
        // float fv = clamp(BRDFBasics::SchlickWeight(NoV),0.0f,1.0f);

        // //return rr * (fl + fv + fl * fv * (rr - 1.0f));
        // return mix(1.0f, rr, fl) * mix(1.0f, rr, fv);
        float LH2 = abs(dot(wi, wo)) + 1;
        float RR = roughness * LH2;
        return RR;
    }

    static __inline__ __device__
    float EvaluateDisneyDiffuse(
        float roughness,
        float flatness,
        vec3 wi, 
        vec3 wo, 
        vec3 wm,
        bool thin)
    {
        float NoL = abs(wi.z);
        float NoV = abs(wo.z);

        float fl = BRDFBasics::SchlickWeight(NoL);
        float fv = BRDFBasics::SchlickWeight(NoV);

        float h = 0.0f;

        // if(thin && flatness > 0.0f) {
        //     float a = roughness * roughness;

        //     float HoL = dot(wm, wi);
        //     float fss90 = HoL * HoL * roughness;
        //     float fss = mix(1.0f, fss90, fl) * mix(1.0f, fss90, fv);

        //     float ss = 1.25f * (fss * (1.0f / (NoL + NoV + 1e-6) - 0.5f) + 0.5f);
        //     h = ss;
        // }

        float f = (1.0f - 0.5f * fl) * (1.0f - 0.5f * fv);
        float rr = EvaluateDisneyRetroDiffuse(roughness, wi, wo);
        f += rr*(fl + fv + fl * fv * (rr - 1.0f));
        return f / M_PIf ;
    }
    static __inline__ __device__
        float3 EvaluateDisney2(
            vec3 illum,
            vec3 baseColor,
            vec3 sssColor,
            float metallic,
            float subsurface,
            float specular,
            float roughness,
            float specularTint,
            float anisotropic,
            float anisoRotation,
            float sheen,
            float sheenTint,
            float clearCoat,
            float clearcoatGloss,
            float ccRough,
            float ccIor,
            float specTrans,
            float scatterDistance,
            float ior,
            float flatness,

            vec3 wi, //in world space
            vec3 wo, //in world space
            vec3 T,
            vec3 B,
            vec3 N,
            vec3 N2,
            bool thin,
            bool is_inside,
            float& fPdf,
            float& rPdf,
            float nDl,
            vec3 &dterm,
            vec3 &sterm,
            vec3 &tterm)

    {
        bool sameside = (dot(wo, N)*dot(wo, N2))>0.0f;
        if(sameside == false)
        {
          wo = normalize(wo - 1.01f * dot(wo, N) * N);
        }
        float eta = dot(wo, N)>0?ior:1.0f/ior;
        vec3 f = vec3(0.0f);
        fPdf = 0.0f;
        rotateTangent(T, B, N, anisoRotation * 2 * 3.1415926f);
        // Onb tbn = Onb(N);
        world2local(wi, T, B, N);
        world2local(wo, T, B, N);

        bool reflect = wi.z * wo.z > 0.0f;

        vec3 Csheen, Cspec0;
        float F0;

        vec3 wm = reflect? normalize(wi + wo):normalize(wi + wo * eta);

        wm = wm.z<0.0f?-wm:wm;
        BRDFBasics::TintColors(mix(baseColor, sssColor, subsurface), eta, specularTint, sheenTint, F0, Csheen, Cspec0);
        Cspec0 = Cspec0 * specular;
        //material layer mix weight
        float dielectricWt = (1.0 - metallic) * (1.0 - specTrans);
        float metalWt = metallic;
        float glassWt = (1.0 - metallic) * specTrans;

        float schlickWt = BRDFBasics::SchlickWeight(abs(wo.z));
        float psss = subsurface/(1.0f + subsurface);
        //event probability
        float diffPr = dielectricWt ;
        float sssPr = dielectricWt   * subsurface;
        float dielectricPr = dielectricWt * Luminance(mix(Cspec0, vec3(1.0), schlickWt));
        float metalPr = metalWt * Luminance(mix(baseColor, vec3(1.0), schlickWt));
        float glassPr = glassWt;
        float clearCtPr = 0.25 * clearCoat;

        float invTotalWt = 1.0 / (diffPr + sssPr + dielectricPr + metalPr + glassPr + clearCtPr);
        diffPr       *= invTotalWt;
        sssPr        *= invTotalWt;
        dielectricPr *= invTotalWt;
        metalPr      *= invTotalWt;
        glassPr      *= invTotalWt;
        clearCtPr    *= invTotalWt;
        float p0 = diffPr;
        float p1 = p0 + sssPr;
        float p2 = p0 + dielectricPr;
        float p3 = p2 + metalPr;
        float p4 = p3 + glassPr;
        float p5 = p4 + clearCtPr;


        float tmpPdf = 0.0f;
        wo = normalize(wo);
        wi = normalize(wi);
        wm = normalize(wm);
        float HoV = abs(dot(wm, wo));
        dterm = vec3(0.0f);
        sterm = vec3(0.0f);
        tterm = vec3(0.0f);

        if(diffPr > 0.0 && reflect)
        {

            vec3 d = BRDFBasics::EvalDisneyDiffuse(mix(baseColor,sssColor,subsurface)*(1.0f - psss), subsurface, roughness, sheen,
                                             Csheen, wo, wi, wm, tmpPdf) * dielectricWt   * illum;
            dterm = dterm + d;
            f = f + d;
            fPdf += tmpPdf * diffPr ;
        }
        if(dielectricPr>0.0 && reflect)
        {
            float F = clamp((BRDFBasics::DielectricFresnel(HoV, 1.0 / ior) - F0) / (1.0 - F0),0.0f,1.0f);
            float ax, ay;
            BRDFBasics::CalculateAnisotropicParams(roughness,anisotropic,ax,ay);
            vec3 s = BRDFBasics::EvalMicrofacetReflection(ax, ay, wo, wi, wm,
                                          mix(Cspec0, vec3(1.0f), F), tmpPdf) * dielectricWt  * illum;
            sterm = sterm + s;
            f = f + s;
            fPdf += tmpPdf * dielectricPr;
        }
        if(metalPr>0.0 && reflect)
        {
            vec3 F = mix(baseColor, vec3(1.0), BRDFBasics::SchlickWeight(HoV));
            float ax, ay;
            BRDFBasics::CalculateAnisotropicParams(roughness,anisotropic,ax,ay);
            vec3 s = BRDFBasics::EvalMicrofacetReflection(ax, ay, wo, wi, wm, F, tmpPdf) * metalWt  * illum;
            sterm = sterm + s;
            f = f + s;
            fPdf += tmpPdf * metalPr;
        }
        if(glassPr>0.0)
        {
            bool entering = wo.z>0?true:false;

            //float F = BRDFBasics::DielectricFresnel(, eta);
            float ax, ay;
            BRDFBasics::CalculateAnisotropicParams(roughness,anisotropic,ax,ay);
            if (reflect) {

              vec3 wm = normalize(wi + wo);
              float F = BRDFBasics::DielectricFresnel(abs(dot(wm, wo)), entering?ior:1.0/ior);
              vec3 s = BRDFBasics::EvalMicrofacetReflection(ax, ay, wo, wi, wm, vec3(F),
                                            tmpPdf) * glassWt;
              sterm = sterm + s;
              f = f + s;
              fPdf += tmpPdf * glassPr * F;
            } else {
              if(thin || ior<1.01f)
              {

                vec3 t = sqrt(sssColor) * glassWt;
                tterm = tterm + t;
                f = f + t;
                fPdf += 1.0f * glassPr;
              }else {
                vec3 wm = entering?-normalize(wo + ior * wi) : normalize(wo + 1.0f/ior * wi);
                float F = BRDFBasics::DielectricFresnel(abs(dot(wm, wo)), entering?ior:1.0/ior);
                float tmpPdf;
                vec3 brdf = BRDFBasics::EvalMicrofacetRefraction(sssColor,
                                                                 ax, ay,
                                                                 entering? ior:1.0f/ior,
                                                                 wo, wi, wm,
                                                                 vec3(F), tmpPdf);
//                float tmpPdf1;
//                wm = normalize(wi + wo * ior);
//                vec3 brdf1 = BRDFBasics::EvalMicrofacetRefraction(sssColor,
//                                                      ax, ay,
//                                                      ior,
//                                                      wo, wi, wm,
//                                                      vec3(F), tmpPdf1);
//                float tmpPdf2;
//                wm = normalize(wo + wi * ior);
//                vec3 brdf2 = BRDFBasics::EvalMicrofacetRefraction(sssColor,
//                                                      ax, ay,
//                                                      ior,
//                                                      wi, wo, wm,
//                                                      vec3(F), tmpPdf2);
                vec3 t = brdf * glassWt  * illum;
                tterm = tterm + t;
                f = f + t;
                fPdf += tmpPdf * glassPr * (1.0 - F);

              }
            }

        }
        if(clearCtPr>0.0 && reflect)
        {
            vec3 s = BRDFBasics::EvalClearcoat(ccRough, wo, wi,
                                         wm, tmpPdf) * 0.25 * clearCoat  * illum;
            sterm = sterm + s;
            f =  f + s;
            fPdf += tmpPdf * clearCtPr;
        }
        if(sssPr>0.0)
        {
          bool trans = wo.z * wi.z < 0.0f;
          float FL = BRDFBasics::SchlickWeight(abs(wi.z));
          float FV = BRDFBasics::SchlickWeight(abs(wo.z));
          float term = wo.z>0?FV:FL;
          float tmpPdf = trans?0.5/M_PIf:0.0f;
          vec3 d = 1.0f/M_PIf * (1.0f - 0.5f * term) * (trans?vec3(1.0f):vec3(0.0f))  * dielectricWt * subsurface;
          dterm = dterm + d;
          f = f + d;
          fPdf += tmpPdf * sssPr;

        }
        dterm = dterm * abs(wi.z);
        sterm = sterm * abs(wi.z);
        tterm = tterm * abs(wi.z);
        return float3( f * abs(wi.z));
    }
    static __inline__ __device__
    float3 EvaluateDisney(
        vec3 illum,
        vec3 baseColor,
        vec3 sssColor,
        float metallic,
        float subsurface,
        float specular,
        float roughness,
        float specularTint,
        float anisotropic,
        float anisoRotation,
        float sheen,
        float sheenTint,
        float clearCoat,
        float clearcoatGloss,
        float ccRough,
        float ccIor,
        float specTrans,
        float scatterDistance,
        float ior,
        float flatness,

        vec3 wi, //in world space
        vec3 wo, //in world space
        vec3 T,
        vec3 B,
        vec3 N,
        bool thin,
        bool is_inside,
        float& fPdf,
        float& rPdf,
        float nDl)

    {
        rotateTangent(T, B, N, anisoRotation * 2 * 3.1415926f);
        //Onb tbn = Onb(N);
        world2local(wi, T ,B, N);
        world2local(wo, T ,B, N);
        vec3 wm = normalize(wo+wi);

        float NoL = wi.z;
        float NoV = wo.z;
        float NoH = wm.z;
        float HoL = dot(wm,wi);
        
        float3 reflectance = make_float3(0.0f,0.0f,0.0f);
        fPdf = 0.0f;
        rPdf = 0.0f;

        float pSpecular,pDiffuse,pClearcoat,pSpecTrans, totalp;
        pdf(metallic,specTrans,clearCoat,pSpecular,pDiffuse,pClearcoat,pSpecTrans,totalp);

        // calculate all of the anisotropic params 
        float ax,ay;
        BRDFBasics::CalculateAnisotropicParams(roughness, anisotropic, ax, ay);

        float diffuseW = pDiffuse;
        float transmissionW = pSpecTrans;


        // Clearcoat
     
        bool upperHemisphere = NoL > 0.0f && NoV > 0.0f;

        if(upperHemisphere && clearCoat > 0.0f) {
            float forwardClearcoatPdfW;
            float reverseClearcoatPdfW;
            float clearcoat = EvaluateClearcoat(clearCoat,clearcoatGloss,max(0.1f,ccRough), ccIor, NoH,NoL,NoV,HoL,HoL,forwardClearcoatPdfW,reverseClearcoatPdfW);
            fPdf += pClearcoat * forwardClearcoatPdfW;
            rPdf += pClearcoat * reverseClearcoatPdfW;
            reflectance += pClearcoat * make_float3(clearcoat,clearcoat,clearcoat) * float3(illum);
        }
        // Diffuse

        if(diffuseW > 0.0f){
            float forwardDiffusePdfW = abs(wi.z);
            float reverseDiffusePdfW = abs(wo.z);
            float diffuse = EvaluateDisneyDiffuse(roughness,flatness, wi, wo, wm, thin);
            auto c = mix(baseColor, sssColor, subsurface);
            //vec3 fr = abs(vec3(1.0) - 0.5 * BRDFBasics::fresnelSchlick(c, abs(wm.z)));
            //printf("fr: %f, %f, %f\n", fr.x, fr.y, fr.z);
            //float w = max(dot(fr, vec3(1.0f,1.0f,1.0f)) , 0.0f);
            float p_in = subsurface;
            //printf("w: %f\n", w);

            float ptotal = 1.0f + p_in ;
            float psss = subsurface>0? p_in/ptotal : 0; // /ptotal;
            sheen = wi.z * wo.z>0? sheen:0.0f;
            vec3 lobeOfSheen =  EvaluateSheen(baseColor,sheen,sheenTint, HoL);
            lobeOfSheen = lobeOfSheen * D_Charlie(abs(wm.z), roughness) * V_Charlie(abs(wi.z), abs(wo.z), roughness);
            //float sheenW = sheen / (1.0f + sheen);
            vec3 diffusepart = diffuse * baseColor + lobeOfSheen ;
            fPdf += pDiffuse * forwardDiffusePdfW;
            rPdf += pDiffuse * reverseDiffusePdfW;
            if(!thin && nDl<=0.0f)
                diffuse = 0;
            vec3 lighting = (1.0f - psss) * diffusepart;
            lighting = clamp(lighting * illum, vec3(0.0f), vec3(50.0f));
            reflectance += diffuseW * float3(lighting);
        }
        // Transsmission
        if(transmissionW > 0.0f) {
            float rscaled = thin ? BRDFBasics::ThinTransmissionRoughness(ior, roughness) : roughness;
            float tax, tay;
            BRDFBasics::CalculateAnisotropicParams(rscaled, anisotropic, tax, tay);

            float3 transmission = EvaluateDisneySpecTransmission(baseColor,metallic,ior,specularTint,roughness, tax, tay, thin, is_inside,wo,wi);
            reflectance += transmissionW * transmission * float3(illum);

            float forwardTransmissivePdfW;
            float reverseTransmissivePdfW;
            BRDFBasics::GgxVndfAnisotropicPdf(wi, wm, wo, tax, tay, forwardTransmissivePdfW, reverseTransmissivePdfW);

            fPdf += pSpecTrans * forwardTransmissivePdfW / (pow(HoL + ior * HoL,2.0f));
            rPdf += pSpecTrans * reverseTransmissivePdfW / (pow(HoL + ior * HoL,2.0f));
        }
        // Specular

        if(upperHemisphere) {
            float forwardMetallicPdfW;
            float reverseMetallicPdfW;
            vec3 Spec = EvaluateDisneyBRDF(baseColor,  metallic, subsurface,  specular, max(0.1f,roughness), specularTint, anisotropic, sheen, sheenTint, clearCoat, clearcoatGloss, ior, is_inside, wi, wo, forwardMetallicPdfW, reverseMetallicPdfW);

            reflectance += pSpecular * Spec * vec3(illum);
            fPdf += pSpecular * forwardMetallicPdfW / (4 * abs(HoL) );
            rPdf += pSpecular * reverseMetallicPdfW / (4 * abs(HoL));
        }

        reflectance = reflectance * abs(NoL) ;

        return reflectance * totalp;/// totalp;

    }
    static __inline__ __device__
    float brdf_pdf(vec3 wo, vec3 wm, float roughness, float anisotropic)
    {
      float ax, ay;
      BRDFBasics::CalculateAnisotropicParams(roughness, anisotropic, ax, ay);
      float d = BRDFBasics::GgxD(wm, ax, ay);
      return  d * abs(wm.z) / (4.0f * abs(dot(wo, wm)));
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
        Onb  tbn = Onb(N);
        tbn.m_tangent = T;
        tbn.m_binormal = B;
        float ax, ay;
        BRDFBasics::CalculateAnisotropicParams(roughness, anisotropic, ax, ay);
        //float2 r01 = sobolRnd(seed);
        float r0 = rnd(seed);
        float r1 = rnd(seed);
        vec3 wm = BRDFBasics::sampleGgxAnisotropic(wo, ax, ay, r0, r1);//BRDFBasics::SampleGgxVndfAnisotropic(wo, ax, ay, r0, r1);
        //vec3 wm = BRDFBasics::sampleOnHemisphere(seed, roughness, fPdf);
        wi = normalize(reflect(-wo, wm)); 
        if(wi.z<=0.0f)
        {
//            fPdf = 0.0f;
//            rPdf = 0.0f;
//            wi = vec3(0,0,0);
//            reflectance = vec3(0,0,0);
//            return false;
            wi.z = 1e-4;
        }

        vec3 F = DisneyFresnel(baseColor, metallic, ior, specularTint, dot(wm, wo), dot(wm, wi), is_inside);
        float g = BRDFBasics::GgxG(wo, wi, ax, ay);
        float d = BRDFBasics::GgxD(wm, ax, ay);
        //float G1v = BRDFBasics::SeparableSmithGGXG1(wo, wm, ax, ay);
        //float G1l = BRDFBasics::SeparableSmithGGXG1(wi, wm, ax, ay);
        float3 specular = g * F / (4 * abs(wi.z) * abs(wo.z)) * abs(wi.z);
        float p =  abs(wm.z) / (4.0f * abs(dot(wo, wm)));

        tbn.inverse_transform(wi);
        wi = normalize(wi);

        //BRDFBasics::GgxVndfAnisotropicPdf(wi, wm, wo, ax, ay, fPdf, rPdf);
        // / fPdf;
        //fPdf *= (1.0f / (4 * abs(dot(wo, wm))));
        //rPdf *= (1.0f / (4 * abs(dot(wi, wm))));
        reflectance = specular / p;

        return true;
    }

    static __inline__ __device__
    bool SampleDisneyClearCoat(
            unsigned int &seed,
            float clearCoat,
            float clearcoatGloss,
            float clearCoatRoughness,
            float clearCoatIOR,
            vec3 T,
            vec3 B,
            vec3 N,
            vec3 wo,
            vec3& wi,
            vec3& reflectance,
            float& fPdf,
            float& rPdf
            )

    {
        float ax, ay;
        BRDFBasics::CalculateAnisotropicParams(clearCoatRoughness, 0, ax, ay);
        //float2 r01 = sobolRnd(seed);
        float r0 = rnd(seed);
        float r1 = rnd(seed);
        vec3 wm = BRDFBasics::sampleGgxAnisotropic(wo, ax, ay, r0, r1);//BRDFBasics::SampleGgxVndfAnisotropic(wo, ax, ay, r0, r1);
        if(dot(wm,wo) < 0.0f){
            wm = -wm;
        }

        wi = normalize(reflect(-wo,wm));

//        if(dot(wi,wo) < 0.0f){ //removable?
//            return false;
//        }

        float NoH = wm.z;
        float LoH = dot(wm,wi);
        float NoL  = abs(wi.z);
        float NoV = abs(wo.z);

        //float d = BRDFBasics::GTR1(abs(NoH),lerp(0.1f, 0.001f, clearcoatGloss));
        float d = BRDFBasics::GTR1(abs(NoH),mix(0.1f, 0.001f, clearcoatGloss));
        //previous: float f = BRDFBasics::fresnelSchlick(LoH, 0.04); wrong
        float f = BRDFBasics::fresnelDielectric(LoH, 1.0f, clearCoatIOR, false);
        float g = BRDFBasics::GgxG(wo, wi, ax, ay);//BRDFBasics::SeparableSmithGGXG1(wi,  wm, clearCoatRoughness, clearCoatRoughness);
        //float gl = BRDFBasics::SeparableSmithGGXG1(wo,  wm, clearCoatRoughness, clearCoatRoughness);

        //fPdf = d / (4.0f * dot(wo,wm));
        //rPdf = d /(4.0f * LoH);
        float p = d * abs(wm.z) / (4.0f * abs(dot(wo, wm)));
        reflectance = vec3(0.25f * clearCoat * d * g * f )/ (4 * abs(wi.z) * abs(wo.z)) * abs(wi.z);//  * abs(wo.z) /(wm.z + 1e-8f) ;///fPdf ;
        reflectance = reflectance / p;

        Onb  tbn = Onb(N);
        tbn.m_tangent = T;
        tbn.m_binormal = B;
        tbn.inverse_transform(wi);
        wi = normalize(wi);
        return true;
        
    }
    static __inline__ __device__
    float refractionG(vec3 wm, vec3 wo, float n, float &c)
    {
        c = dot(wo, wm);
        if(c < 0.0f) {
            c = -c;
            wm = -wm;
        }
        return 1.0f / ( n * n) - (1.0f - c * c);
    }
    static __inline__ __device__ 
    bool Transmit(vec3 wm, vec3 wo, float n, vec3& wi)
    {
        float c = dot(wo, wm);
        if(c < 0.0f) {
            c = -c;
            wm = -wm;
        }
        float root = 1.0f - n * n * (1.0f - c * c);
        if(root <= 0){
            return false;
        }

        wi = normalize((n * c -sqrt(root)) * wm - n * wo);
        return true;
    }

    static __inline__ __device__
        bool SampleDisneySpecTransmission2(
            unsigned int& seed,
            float ior,
            float roughness,
            float anisotropic,

            vec3 baseColor,
            vec3 transmittanceColor,

            float scatterDistance,
            vec3 wo,
            vec3& wi,
            float& rPdf,
            float& fPdf,
            vec3& reflectance,
            SurfaceEventFlags& flag,
            int& medium,
            vec3& extinction,
            bool thin,
            bool is_inside,
            vec3 T,
            vec3 B,
            vec3 N,
            bool& isTrans

            )
    {
        //here just redo the transmission function
        if(wo.z == 0.0f){
            fPdf = 0.0f;
            rPdf = 0.0f;
            reflectance = vec3(0.0f);
            wi = vec3(0.0f);
            return false;
            //            wo.z = 1e-5;
        }
        float rscaled = thin ? BRDFBasics::ThinTransmissionRoughness(ior,  roughness) : roughness;
        float tax,tay;
        BRDFBasics::CalculateAnisotropicParams(rscaled,anisotropic,tax,tay);

        //float2 r01 = sobolRnd(seed);
        float r0 = rnd(seed);
        float r1 = rnd(seed);
        auto wx = wo;
        if(thin == false && wx.z<0)
        {
            wx.z = -wx.z;
        }
        vec3 wm = BRDFBasics::sampleGgxAnisotropic(wx, tax, tay, r0, r1);

        //determine refraction before Fresnel
        float VoH = dot(wm,wo);
        if(wm.z < 0.0f){
            VoH = -VoH;
        }

        float ni = wo.z > 0.0f ? 1.0f : ior;
        float nt = wo.z > 0.0f ? ior : 1.0f;
        float relativeIOR = ni / nt;
        float c1;
        float gg = refractionG(wm, wo, relativeIOR, c1);
        float R = 1.0f;
        if(gg < 0)
        {
            R = 1.0f;
        }
        else
        {
            float g = sqrt(gg);
            float x = (c1 * (g + c1) - 1) / (c1 * (g - c1) + 1);
            float y = (g - c1) / (g + c1);
            R = 0.5f * y*y * (1 + x * x);
        }
        if( rnd(seed) < 1 - R)
        {
            Transmit(wm, wo, relativeIOR, wi);
            if(thin)
            {
                wi = normalize(reflect(-wo,wm));
                wi.z = -wi.z;
                reflectance = sqrt(transmittanceColor);
                flag = scatterEvent;
            }
            else
            {
                flag = transmissionEvent;
                isTrans = true;
                //phaseFuncion = (!is_inside)  ? isotropic : vacuum;
                extinction = CalculateExtinction(transmittanceColor, scatterDistance);
                float g = BRDFBasics::GgxG(wo, wi, tax, tay);
                g = ior>1.01f? g:1.0f;
                reflectance = g * transmittanceColor
                              * abs(wi.z) / (abs(wi.z) * abs(wo.z));
                float LoH = abs(dot(wi,wm));
                float jacobian = LoH  / pow(LoH + relativeIOR * VoH, 2.0f);
                float p = abs(wm.z) / (abs(dot(wo, wm)));
                reflectance = ior>1.1f? reflectance / p * jacobian : g * transmittanceColor;
            }
        }
        else
        {

            wi = normalize(reflect(-wo,wm));
            //float G1l = BRDFBasics::SeparableSmithGGXG1(wi, wm, tax, tay);
            float g = BRDFBasics::GgxG(wo, wi, tax, tay);

            flag = scatterEvent; // scatter event
            reflectance = g * baseColor * abs(wi.z) / (4 * abs(wi.z) * abs(wo.z));
            float p = abs(wm.z) / (4.0f * abs(dot(wo, wm)));
            //fPdf *= (1.0f / (4 * abs(dot(wo, wm))));
            //float jacobian = 4 * abs(VoH);
            //pdf = F / jacobian;
            reflectance = reflectance  / p;
        }

        if(wi.z == 0.0f){
            fPdf = 0.0f;
            rPdf = 0.0f;
            reflectance = vec3(0.0f);
            wi = vec3(0.0f);
            return false;
            //            if(rnd(seed)>0.5)
            //            {
            //                wi.z = 1e-5;
            //            } else
            //            {
            //                wi.z = - (1e-5);
            //            }
        }

        //if(roughness < 0.01f){
        //    * (int*) (&flag) |= 0x04; // flag |= SurfaceEventFlags::diracEvent ?
        //}




        Onb  tbn = Onb(N);
        tbn.m_tangent = T;
        tbn.m_binormal = B;
        tbn.inverse_transform(wi);
        wi = normalize(wi);
        return true;
    }


    static __inline__ __device__ 
    bool SampleDisneySpecTransmission(
            unsigned int& seed,
            float ior,
            float roughness,
            float anisotropic,

            vec3 baseColor,
            vec3 transmittanceColor,

            float scatterDistance,
            vec3 wo,
            vec3& wi,
            float& rPdf,
            float& fPdf,
            vec3& reflectance,
            SurfaceEventFlags& flag,
            int& medium,
            vec3& extinction,
            bool thin,
            bool is_inside,
            vec3 T,
            vec3 B,
            vec3 N,
            bool& isTrans

            )
    {
        if(wo.z == 0.0f){
            fPdf = 0.0f;
            rPdf = 0.0f;
            reflectance = vec3(0.0f);
            wi = vec3(0.0f);
            return false;
            //            wo.z = 1e-5;
        }
        float rscaled = thin ? BRDFBasics::ThinTransmissionRoughness(ior,  roughness) : roughness;

        float tax,tay;
        BRDFBasics::CalculateAnisotropicParams(rscaled,anisotropic,tax,tay);

        //float2 r01 = sobolRnd(seed);
        float r0 = rnd(seed);
        float r1 = rnd(seed);
        auto wx = wo;
        if(thin == false && wx.z<0)
        {
            wx.z = -wx.z;
        }
        vec3 wm = BRDFBasics::sampleGgxAnisotropic(wx, tax, tay, r0, r1);//BRDFBasics::SampleGgxVndfAnisotropic(wx, tax, tay, r0, r1);


        float VoH = dot(wm,wo);
        if(wm.z < 0.0f){
            VoH = -VoH;
        }

        float ni = wo.z > 0.0f ? 1.0f : ior;
        float nt = wo.z > 0.0f ? ior : 1.0f;
        float relativeIOR = ni / nt;

        float F = BRDFBasics::fresnelDielectric(VoH, 1.0f, ior, is_inside);
        //float G1v = BRDFBasics::SeparableSmithGGXG1(wo, wm, tax, tay);


        float pdf;

        if(rnd(seed) <= F){
            wi = normalize(reflect(-wo,wm));
            //float G1l = BRDFBasics::SeparableSmithGGXG1(wi, wm, tax, tay);
            float g = BRDFBasics::GgxG(wo, wi, tax, tay);

            flag = scatterEvent; // scatter event
            reflectance = g * baseColor * abs(wi.z) / (4 * abs(wi.z) * abs(wo.z));
            float p = abs(wm.z) / (4.0f * abs(dot(wo, wm)));
            //fPdf *= (1.0f / (4 * abs(dot(wo, wm))));
            //float jacobian = 4 * abs(VoH);
            //pdf = F / jacobian;
            reflectance = reflectance  / p;

        }else{
            if(thin){
                wi = normalize(reflect(-wo,wm));
                wi.z = -wi.z;
                reflectance = sqrt(transmittanceColor);
                flag = scatterEvent;
            }else{
                if( Transmit(wm, wo,relativeIOR, wi)){
                    flag = transmissionEvent;
                    isTrans = true;
                    //phaseFuncion = (!is_inside)  ? isotropic : vacuum;
                    extinction = CalculateExtinction(transmittanceColor, scatterDistance);
                    //extinction = CalculateExtinction2(vec3 albedo, vec3 radius, vec3 &sigma_t, vec3 &alpha)
                }else{
                    flag = scatterEvent;
                    isTrans = true;
                    wi = normalize(reflect(-wo,wm));
                }
                float g = BRDFBasics::GgxG(wo, wi, tax, tay);
                g = ior>1.01? g:1.0f;
                reflectance = g * transmittanceColor
                    * abs(wi.z) / (abs(wi.z) * abs(wo.z));
                float LoH = abs(dot(wi,wm));
                float jacobian = LoH  / pow(LoH + relativeIOR * VoH, 2.0f);
                float p = abs(wm.z) / (abs(dot(wo, wm)));
                reflectance = ior>1.1? reflectance / p * jacobian : g * transmittanceColor;
            }
            //float G1l = BRDFBasics::SeparableSmithGGXG1(wi, wm, tax, tay);
            //float LoH = abs(dot(wi,wm));
            //float jacobian = LoH  / (pow(LoH + relativeIOR * VoH, 2.0f) + 1e-5);
            //pdf = (1.0f - F) / jacobian;
            //reflectance = reflectance * jacobian * G1l * abs(wi.z) / abs(wm.z) / (4.0f * abs(wi.z) * abs(wo.z));
        }

        if(wi.z == 0.0f){
            fPdf = 0.0f;
            rPdf = 0.0f;
            reflectance = vec3(0.0f);
            wi = vec3(0.0f);
            return false;
            //            if(rnd(seed)>0.5)
            //            {
            //                wi.z = 1e-5;
            //            } else
            //            {
            //                wi.z = - (1e-5);
            //            }
        }

        //if(roughness < 0.01f){
        //    * (int*) (&flag) |= 0x04; // flag |= SurfaceEventFlags::diracEvent ?
        //}

        BRDFBasics::GgxVndfAnisotropicPdf(wi, wm, wo, tax, tay, fPdf, rPdf);
        fPdf *= pdf;
        rPdf *= pdf;


        Onb  tbn = Onb(N);
        tbn.m_tangent = T;
        tbn.m_binormal = B;
        tbn.inverse_transform(wi);
        wi = normalize(wi);
        return true;
//        if(wo.z == 0.0f){
//            fPdf = 0.0f;
//            rPdf = 0.0f;
//            reflectance = vec3(0.0f);
//            wi = vec3(0.0f);
//            return false;
//            //            wo.z = 1e-5;
//        }
//        float rscaled = thin ? BRDFBasics::ThinTransmissionRoughness(ior,  roughness) : roughness;
//
//        float tax,tay;
//        BRDFBasics::CalculateAnisotropicParams(rscaled,anisotropic,tax,tay);
//
//        //float2 r01 = sobolRnd(seed);
//        float r0 = rnd(seed);
//        float r1 = rnd(seed);
//        auto wx = wo;
//        if(thin == false && wx.z<0)
//        {
//            wx.z = -wx.z;
//        }
//        vec3 wm = BRDFBasics::SampleGgxVndfAnisotropic(wx, tax, tay, r0, r1);
//
//
//        float VoH = dot(wm,wo);
//        if(wm.z < 0.0f){
//            VoH = -VoH;
//        }
//
//        float ni = wo.z > 0.0f ? 1.0f : ior;
//        float nt = wo.z > 0.0f ? ior : 1.0f;
//        float relativeIOR = ni / nt;
//
//        float F = BRDFBasics::fresnelDielectric(VoH, 1.0f, ior, is_inside);
//
//        float G1v = BRDFBasics::SeparableSmithGGXG1(wo, wm, tax, tay);
//
//        float pdf;
//
//        if(rnd(seed) <= F){
//            wi = normalize(reflect(-wo,wm));
//
//            flag = scatterEvent; // scatter event
//            reflectance = G1v * baseColor;
//
//            //fPdf *= (1.0f / (4 * abs(dot(wo, wm))));
//            float jacobian = 4 * abs(VoH)  + 1e-5;
//            pdf = F / jacobian;
//
//        }else{
//            if(thin){
//                wi = normalize(reflect(-wo,wm));
//                wi.z = -wi.z;
//                reflectance = G1v * sqrt(transmittanceColor);
//                flag = scatterEvent;
//            }else{
//                if( Transmit(wm, wo,relativeIOR, wi)){
//                    flag = transmissionEvent;
//                    isTrans = true;
//                    //phaseFuncion = (!is_inside)  ? isotropic : vacuum;
//                    extinction = CalculateExtinction(transmittanceColor, scatterDistance);
//                    //extinction = CalculateExtinction2(vec3 albedo, vec3 radius, vec3 &sigma_t, vec3 &alpha)
//                }else{
//                    flag = scatterEvent;
//                    isTrans = true;
//                    wi = normalize(reflect(-wo,wm));
//                }
//                reflectance = G1v * vec3(1.0f);
//            }
//            float LoH = abs(dot(wi,wm));
//            float jacobian = LoH  / (pow(LoH + relativeIOR * VoH, 2.0f) + 1e-5) + 1e-5;
//            pdf = (1.0f - F) / jacobian;
//
//        }
//
//        if(wi.z == 0.0f){
//            fPdf = 0.0f;
//            rPdf = 0.0f;
//            reflectance = vec3(0.0f);
//            wi = vec3(0.0f);
//            return false;
//            //            if(rnd(seed)>0.5)
//            //            {
//            //                wi.z = 1e-5;
//            //            } else
//            //            {
//            //                wi.z = - (1e-5);
//            //            }
//        }
//
//        //if(roughness < 0.01f){
//        //    * (int*) (&flag) |= 0x04; // flag |= SurfaceEventFlags::diracEvent ?
//        //}
//
//        BRDFBasics::GgxVndfAnisotropicPdf(wi, wm, wo, tax, tay, fPdf, rPdf);
//        fPdf *= pdf;
//        rPdf *= pdf;
//
//
//        Onb  tbn = Onb(N);
//        tbn.m_tangent = T;
//        tbn.m_binormal = B;
//        tbn.inverse_transform(wi);
//        wi = normalize(wi);
//        return true;
    }
    static __inline__ __device__ 
    bool SampleDisneyDiffuse(
        unsigned int& seed,
        vec3 baseColor,
        vec3 sssParam,
        vec3 sssColor,
        float scatterDistance,
        float sheen,
        float sheenTint,
        float roughness,
        float flatness,
        float subsurface,
        bool thin,
        vec3 wo,
        vec3 T,
        vec3 B,
        vec3 N,
        vec3& wi,
        float& fPdf,
        float& rPdf,
        vec3& reflectance,
        SurfaceEventFlags& flag,
        int& medium,
        vec3& extinction,
        bool is_inside,
        bool &isSS

            )
    {
        vec3 color = mix(baseColor, sssColor, subsurface);
        color = clamp(color, vec3(0.05), vec3(0.99));
        wi =  normalize(BRDFBasics::sampleUniformHemiSphere(seed));//normalize(BRDFBasics::sampleOnHemisphere(seed, 1.0f, fPdf));
        vec3 wm = normalize(wi+wo);
        float NoL = wi.z;
        if(NoL==0.0f ){
            fPdf = 0.0f;
            rPdf = 0.0f;
            reflectance = vec3(0.0f);
            wi = vec3(0.0f);
            //return false;
            wi.z = 1e-5;
        }

        float NoV = wo.z;

        vec3 sssRadius = sssParam * subsurface;
        vec3 scalerSS = sssColor * sssParam;
        
        RadiancePRD* prd = getPRD();
        prd->ss_alpha = color;

        flag = scatterEvent;
        
        //vec3 fr = abs(vec3(1.0) - 0.5f * BRDFBasics::fresnelSchlick(color, abs(wm.z)));
        //printf("fr: %f, %f, %f\n", fr.x, fr.y, fr.z);
        //float w = max(dot(fr, vec3(1.0f,1.0f,1.0f)) , 0.0f);
        float p_in = subsurface;
        //printf("w: %f\n", w);

        float ptotal = 1.0f + p_in ;

        float psss = subsurface>0? p_in/ptotal : 0; // /ptotal;
        float prnd = rnd(seed);
        //printf("weight: %f, rnd: %f\n", weight,prnd);
        bool trans = false;
        if(wo.z>0) //we are outside
        {
            if (prnd <= psss) {
                trans = true;
                wi = -wi;
                isSS = true;
                if (thin) {
                    color = sqrt(color);
                } else {
                    flag = transmissionEvent;
                    medium = PhaseFunctions::isotropic;
                    //extinction = CalculateExtinction(scalerSS, scatterDistance);
                    CalculateExtinction2(color, sssRadius, prd->sigma_t, prd->ss_alpha);
                    color = vec3(1.0);
                    //color = vec3(0.99f);
                    //color = baseColor;
                }
            } else {
                color = color;
            }
        }else //we are inside
        {
            //either go out or turn in
            if (prnd <= 1)
            {
                trans = true;
                //go out, flag change
                isSS = false;
                wi = wi;
                if (thin) {
                    color = sqrt(color);
                } else {
                    flag = transmissionEvent;
                    medium = PhaseFunctions::decideLate;
                    //extinction = CalculateExtinction(scalerSS, scatterDistance);
                    //CalculateExtinction2(color, sssRadius, prd->sigma_t, prd->ss_alpha);
                    color = vec3(1.0f);//no attenuation happen
                }
            }else
            {
                color = vec3(1.0f);
            }
        }

        float HoL = dot(wm,wi);
        vec3 diffpart = vec3(0.0f);
        if(!trans)
        {

            float psheen = sheen>0.0f? 0.5f:0.0f;
            float amp = sheen>0.0f? 2.0f:1.0f;


            if(rnd(seed)<psheen)
            {
                diffpart =
                  EvaluateSheen(baseColor,sheen,sheenTint, HoL);

                diffpart = diffpart * D_Charlie(abs(wm.z), roughness) * V_Charlie(abs(wi.z), abs(wo.z), roughness);
                diffpart = diffpart  * amp ;
            } else
            {
                diffpart = amp * color * vec3(EvaluateDisneyDiffuse(1.0, flatness, wi, wo, wm, thin)) ;
            }
        }
        //vec3 sheenTerm = EvaluateSheen(baseColor, sheen, sheenTint, HoL);
        //float diff = EvaluateDisneyDiffuse(1.0, flatness, wi, wo, wm, thin);
        if(wi.z<0)
            diffpart =  vec3(1.0f);
        //vec3 diffpart = color * diff * M_PIf;
        reflectance = trans? color  * vec3(1.0) * abs(wi.z) * 2.0f * M_PIf : diffpart * abs(wi.z) * 2.0f * M_PIf;
        //fPdf = abs(NoL) * pdf;
        //rPdf = abs(NoV) * pdf;
        Onb  tbn = Onb(N);
        tbn.m_tangent = T;
        tbn.m_binormal = B;
        tbn.inverse_transform(wi);
        wi = normalize(wi);
        return true;
    }

    static __inline__ __device__
    vec3 SampleScatterDirection(unsigned int &seed)
    {
        //float2 r01 = sobolRnd(seed);
        float r0 = rnd(seed);
        float r1 = rnd(seed);

        float theta = 2.0f * M_PIf * r0;
        float phi = acos(clamp(1 - 2 * r1, -0.9999f, 0.9999f));
        float x = sin(phi) * cos(theta);
        float y = sin(phi) * sin(theta);
        float z = cos(phi);

        return normalize(vec3(x, y, z));
    }
    
    static __inline__ __device__
    vec3 Transmission(const vec3& extinction, float distance)
    {
        return exp(-extinction * distance);
    }

    static __inline__ __device__
    vec3 sss_rw_pdf(const vec3& sigma_t, float t, bool hit, vec3& transmittance)
    {
        vec3 T = Transmission(sigma_t, t);
        transmittance = T;
        return hit? T : (sigma_t * T);
    }

    static __inline__ __device__
    vec3 Transmission2(const vec3& sigma_s, const vec3& sigma_t, const vec3& channelPDF, float t, bool hit)
    {
        vec3 transmittance;
        vec3 pdf = sss_rw_pdf(sigma_t, t, hit, transmittance);

        //printf("trans PDf= %f %f %f sigma_t= %f %f %f \n", pdf.x, pdf.y, pdf.z, sigma_t.x, sigma_t.y, sigma_t.z);
        auto result = hit? transmittance : ((sigma_s * transmittance) / (dot(pdf, channelPDF) + 1e-6f));
        result = clamp(result,vec3(0.0f),vec3(1.0f));
        return result;
    }





    static __inline__ __device__
    bool SampleDisney2(
        unsigned int& seed,
        unsigned int& eventseed,
        vec3 baseColor,
        vec3 transmiianceColor,
        vec3 sssColor,
        float metallic,
        float subsurface,
        float specular,
        float roughness,
        float specularTint,
        float anisotropic,
        float anisoRotation,
        float sheen,
        float sheenTint,
        float clearCoat,
        float clearcoatGloss,
        float ccRough,
        float ccIor,
        float flatness,
        float specTrans,
        float scatterDistance,
        float ior,

        vec3 T,
        vec3 B,
        vec3 N,
        vec3 N2,
        vec3 wo,
        bool thin,
        bool is_inside,
        vec3& wi,
        vec3& reflectance,
        float& rPdf,
        float& fPdf,
        SurfaceEventFlags& flag,
        int& medium,
        vec3& extinction,
        bool& isDiff,
        bool& isSS,
        bool& isTrans,
        float& minSpecRough
    )
    {
        RadiancePRD* prd = getPRD();
        bool sameside = (dot(wo, N)*dot(wo, N2))>0.0f;
        if(sameside == false)
        {
          wo = normalize(wo - 1.01f * dot(wo, N) * N);
        }
        float eta = dot(wo, N)>0?ior:1.0f/ior;
        rotateTangent(T, B, N, anisoRotation * 2 * 3.1415926f);
        world2local(wo, T, B, N);
//        float r1 = rnd(seed);
//        float r2 = rnd(seed);

        vec3 Csheen, Cspec0;
        float F0;

        BRDFBasics::TintColors(mix(baseColor, sssColor, subsurface), eta, specularTint, sheenTint, F0, Csheen, Cspec0);
        Cspec0 = Cspec0 * specular;

        //material layer mix weight
        float dielectricWt = (1.0 - metallic) * (1.0 - specTrans);
        float metalWt = metallic;
        float glassWt = (1.0 - metallic) * specTrans;

        float schlickWt = BRDFBasics::SchlickWeight(abs(wo.z));
        float psss = subsurface/(1.0f + subsurface);
        //dielectricWt *= 1.0f - psub;

        //event probability
        float diffPr = dielectricWt;
        float sssPr = dielectricWt  * subsurface;
        float dielectricPr = dielectricWt * Luminance(mix(Cspec0, vec3(1.0), schlickWt));
        float metalPr = metalWt * Luminance(mix(baseColor, vec3(1.0), schlickWt));
        float glassPr = glassWt;
        float clearCtPr = 0.25 * clearCoat;

        float invTotalWt = 1.0 / (diffPr + sssPr + dielectricPr + metalPr + glassPr + clearCtPr);
        diffPr       *= invTotalWt;
        sssPr        *= invTotalWt;
        dielectricPr *= invTotalWt;
        metalPr      *= invTotalWt;
        glassPr      *= invTotalWt;
        clearCtPr    *= invTotalWt;
        float p0 = diffPr;
        float p1 = p0 + sssPr;
        float p2 = p1 + dielectricPr;
        float p3 = p2 + metalPr;
        float p4 = p3 + glassPr;
        float p5 = p4 + clearCtPr;

        float r3 = rnd(seed);
        Onb  tbn = Onb(N);
        tbn.m_tangent = T;
        tbn.m_binormal = B;
        prd->fromDiff = false;
        if(r3<p1) // diffuse + sss
        {
            float2 r = sobolRnd(seed);
            float r1 = r.x;
            float r2 = r.y;
          auto first_hit_type = prd->first_hit_type;
          prd->first_hit_type = prd->depth==0?DIFFUSE_HIT:first_hit_type;
          if(wo.z<0 && subsurface>0)//inside, scattering, go out for sure
          {
            wi = BRDFBasics::UniformSampleHemisphere(r1, r2);
            flag = transmissionEvent;
            isSS = false;
          }
          else{
            //switch between scattering or diffuse reflection
            float diffp = p0/p1;
            if(rnd(seed)<diffp || prd->fromDiff==true)
            {
              prd->fromDiff = true;
              wi = BRDFBasics::CosineSampleHemisphere(r1, r2);
              isSS = false;
            }else
            {
              //go inside
              wi = -BRDFBasics::UniformSampleHemisphere(rnd(seed), rnd(seed));
              isSS = true;
              flag = transmissionEvent;
              vec3 color = mix(baseColor, sssColor, subsurface) * psss;
              color = clamp(color, vec3(0.05), vec3(0.99));
              vec3 sssRadius = transmiianceColor * subsurface;
              RadiancePRD *prd = getPRD();
              prd->ss_alpha = color;
              if (isSS) {
                medium = PhaseFunctions::isotropic;
                CalculateExtinction2(color, sssRadius, prd->sigma_t,
                                     prd->ss_alpha);
              }
            }
          }
//            prd->fromDiff = true;
//            //float psub = subsurface / (1.0f + subsurface);
//            //float r4 = rnd(seed);
//            //if(r4<psub)
//
//            //else
//            //{
//              wi = BRDFBasics::CosineSampleHemisphere(r1, r2);
//              tbn.inverse_transform(wi);
//              wi = normalize(wi);
//
//              bool sameside2 = (dot(wi, N) * dot(wi, N2)) > 0.0f;
//              if (sameside == false) {
//                wi = normalize(wi - 1.01f * dot(wi, N2) * N2);
//              }
//            //}
//
////        }else if(r3<p1 && prd->fromDiff == false)
////        {
//
//            wi = wo.z > 0 ? -BRDFBasics::UniformSampleHemisphere(r1, r2)
//                          : BRDFBasics::UniformSampleHemisphere(r1, r2);
//            bool trans = true;
//            isSS = wo.z > 0 ? true : false;
//            flag = transmissionEvent;
//            vec3 color = mix(baseColor, sssColor, subsurface);
//            color = clamp(color, vec3(0.05), vec3(0.99));
//            vec3 sssRadius = transmiianceColor * subsurface;
//            RadiancePRD *prd = getPRD();
//            prd->ss_alpha = color;
//            if (isSS) {
//              medium = PhaseFunctions::isotropic;
//              CalculateExtinction2(color, sssRadius, prd->sigma_t,
//                                   prd->ss_alpha);
//            }
            tbn.inverse_transform(wi);
            wi = normalize(wi);

            bool sameside2 = (dot(wi, N) * dot(wi, N2)) > 0.0f;
            if (sameside == false) {
              wi = normalize(wi - 1.01f * dot(wi, N2) * N2);
            }
            //reflectance = vec3(1.0f) * M_PIf ;
            //return true;

        }
        else if(r3<p3)//specular
        {
            float2 r = sobolRnd(seed);
            float r1 = r.x;
            float r2 = r.y;
            auto first_hit_type = prd->first_hit_type;
            prd->first_hit_type = prd->depth==0?SPECULAR_HIT:first_hit_type;
            float ax, ay;
            BRDFBasics::CalculateAnisotropicParams(roughness,anisotropic,ax,ay);

            vec3 wm = BRDFBasics::SampleGGXVNDF(wo.z>0?wo:-wo, ax, ay, r1, r2);

            if (wm.z < 0.0)
              wm = -wm;

            wm = wo.z>0? wm:-wm;

            wi = normalize(reflect(-wo, wm));
            tbn.inverse_transform(wi);
            wi = normalize(wi);

            if(dot(wi, N2)<0)
            {
              wi = normalize(wi - 1.01f * dot(wi, N2) * N2);
            }
        }else if(r3<p4)//glass
        {
            float2 r = sobolRnd(seed);
            float r1 = r.x;
            float r2 = r.y;

//          SampleDisneySpecTransmission2(seed, ior, roughness, anisotropic, baseColor, transmiianceColor, scatterDistance,
//                                        wo, wi, rPdf, fPdf, reflectance, flag, medium, extinction, thin, is_inside,
//                                        T, B, N, isTrans);
            bool entering = wo.z>0?true:false;
            float ax, ay;
            BRDFBasics::CalculateAnisotropicParams(roughness,anisotropic,ax,ay);
            vec3 swo = wo.z>0?wo:-wo;
            vec3 wm = BRDFBasics::SampleGGXVNDF(swo, ax, ay, r1, r2);
            wm = wm.z<0?-wm:wm;

            wm = entering?wm:-wm;

            float F = BRDFBasics::DielectricFresnel(abs(dot(wm, wo)), entering?ior:1.0f/ior);
            float p = rnd(seed);
            if(p<F)//reflection
            {
              wi = normalize(reflect(-normalize(wo),wm));
            }else //refraction
            {
              wi = normalize(refract(wo, wm, entering?1.0f/ior:ior));
              flag = transmissionEvent;
              isTrans = true;
              extinction = CalculateExtinction(transmiianceColor, scatterDistance);
              extinction = entering? extinction : vec3(0.0f);
            }

            tbn.inverse_transform(wi);
            wi = normalize(wi);
          minSpecRough = roughness;
          bool sameside2 = (dot(wi, N) * dot(wi, N2))>0.0f;
          if(sameside2 == false)
          {
            wi = normalize(wi - 1.01f * dot(wi, N2) * N2);
          }
          auto isReflection =  dot(wi, N2) * dot(wo, N2)>0?1:0;
          auto first_hit_type = prd->first_hit_type;
          prd->first_hit_type = prd->depth==0? (isReflection==1?SPECULAR_HIT:TRANSMIT_HIT):first_hit_type;
        }else if(r3<p5)//cc
        {
            float2 r = sobolRnd(seed);
            float r1 = r.x;
            float r2 = r.y;
            auto first_hit_type = prd->first_hit_type;
            prd->first_hit_type = prd->depth==0?SPECULAR_HIT:first_hit_type;
            vec3 wm = BRDFBasics::SampleGTR1(ccRough, r1, r2);

            if (wm.z < 0.0)
              wm = -wm;
            wm = wo.z>0?wm:-wm;
            wi = normalize(reflect(-wo, wm));
            tbn.inverse_transform(wi);
            wi = normalize(wi);
            if(dot(wi, N2)<0)
            {
              wi = normalize(wi - 1.01f * dot(wi, N2) * N2);
            }
        }
        tbn.inverse_transform(wo);
        float pdf, pdf2;
        vec3 rd, rs, rt;
        reflectance = EvaluateDisney2(vec3(1.0f), baseColor, sssColor, metallic, subsurface,
                                      specular, roughness, specularTint, anisotropic, anisoRotation, sheen,
                                      sheenTint, clearCoat, clearcoatGloss, ccRough, ccIor, specTrans,
                                      scatterDistance, ior, flatness, wi, wo, T, B, N, N2, thin,
                                      is_inside, pdf, pdf2, 0, rd, rs, rt);
        fPdf = pdf>1e-5?pdf:0.0f;
        reflectance = pdf>1e-5?reflectance:vec3(0.0f);
        return true;
    }

    static __inline__ __device__
    bool SampleDisney(
        unsigned int& seed,
        vec3 baseColor,
        vec3 transmiianceColor,
        vec3 sssColor,
        float metallic,
        float subsurface,
        float specular,
        float roughness,
        float specularTint,
        float anisotropic,
        float anisoRotation,
        float sheen,
        float sheenTint,
        float clearCoat,
        float clearcoatGloss,
        float ccRough,
        float ccIor,
        float flatness,
        float specTrans,
        float scatterDistance,
        float ior,

        vec3 T,
        vec3 B,
        vec3 N,
        vec3 N2,
        vec3 wo,
        bool thin,
        bool is_inside,
        vec3& wi,
        vec3& reflectance,
        float& rPdf,
        float& fPdf,
        SurfaceEventFlags& flag,
        int& medium,
        vec3& extinction,
        bool& isDiff,
        bool& isSS,
        bool& isTrans,
        float& minSpecRough
            )
    {
        bool sameside = (dot(wo, N)*dot(wo, N2))>0.0f;
        if(sameside == false)
        {
            wo = normalize(wo - 1.01f * dot(wo, N) * N);
        }
        rotateTangent(T, B, N, anisoRotation * 2 * 3.1415926f);
        world2local(wo, T, B, N);
        float pSpecular,pDiffuse,pClearcoat,pSpecTrans;
        float totalp;
        pdf(metallic, specTrans, clearCoat, pSpecular, pDiffuse, pClearcoat, pSpecTrans, totalp);

        bool success = false;
        isDiff = false;
        float pLobe = 0.0f;
        float p = rnd(seed);
        if( p<= pSpecular){
            //roughness = max(minSpecRough, roughness);
            success = SampleDisneyBRDF(
                    seed, 
                    baseColor,
                    metallic,
                    ior, 
                    specularTint, 
                    roughness, 
                    anisotropic, 
                    is_inside, 
                    N,
                    T,
                    B,
                    wo,
                    wi,
                    reflectance,
                    fPdf,
                    rPdf);
            minSpecRough = roughness;
            pLobe = pSpecular;
            if(dot(wi, N2)<0)
            {
                wi = normalize(wi - 1.01f * dot(wi, N2) * N2); 
            }

        }else if(pClearcoat >0.001f && p <= (pSpecular + pClearcoat)){
            //ccRough = max(minSpecRough, ccRough);
            success = SampleDisneyClearCoat(seed, clearCoat, clearcoatGloss, ccRough, ccIor, T, B, N, wo, wi, reflectance, fPdf, rPdf);
            pLobe = pClearcoat;
            if(dot(wi, N2)<0)
            {
                wi = normalize(wi - 1.01f * dot(wi, N2) * N2); 
            }
            minSpecRough = ccRough;
            isDiff = false;
        }else if(pSpecTrans > 0.001f && p <= (pSpecular + pClearcoat + pSpecTrans)){
            //roughness = max(minSpecRough, roughness);
            success = SampleDisneySpecTransmission2(seed, ior, roughness, anisotropic, baseColor, transmiianceColor, scatterDistance,
                                                    wo, wi, rPdf, fPdf, reflectance, flag, medium, extinction, thin, is_inside,
                                                    T, B, N, isTrans);
            minSpecRough = roughness;
            pLobe = pSpecTrans;
            bool sameside = (dot(wi, N) * dot(wi, N2))>0.0f;
            if(sameside == false)
            {
                wi = normalize(wi - 1.01f * dot(wi, N2) * N2);
            }

        }else {
            isDiff = true;
            minSpecRough = 0.2;
            success = SampleDisneyDiffuse(seed, baseColor, transmiianceColor, sssColor, scatterDistance, sheen, sheenTint, roughness, flatness, subsurface, thin, wo, T, B, N, wi, fPdf, rPdf, reflectance, flag, medium, extinction,is_inside, isSS);
            pLobe = pDiffuse;
            bool sameside = (dot(wi, N) * dot(wi, N2))>0.0f;
            if(sameside == false)
            {
                wi = normalize(wi - 1.01f * dot(wi, N2) * N2);
            }
        }
//        vec3 wi2 = wi;
//        world2local(wi2, T,B,N);
//        vec3 wm2 = normalize(wo + wi2);
//
//        float pp = pDiffuse * 0.5 / M_PIf + pSpecular * brdf_pdf(wo, wm2, roughness, anisotropic)
//            + pClearcoat * brdf_pdf(wo, wm2, roughness, anisotropic);
//        pp = pp / (pSpecular + pDiffuse + pClearcoat + pSpecTrans);

        reflectance = reflectance;// / pp;// * totalp;

        return true;

    }
}
static __inline__ __device__ float saturate(float num)
{
    return clamp(num,0.0f,1.0f);
}

static __inline__ __device__ float hash( float n )
{
    return fract(sin(n)*43758.5453f);
}


static __inline__ __device__ float noise( vec3 x )
{
    vec3 p = floor(x);
    vec3 f = fract(x);

    f = f*f*(3.0f-2.0*f);

    float n = p.x + p.y*57.0f + 113.0f*p.z;

    float res = mix(mix(mix( hash(n+  0.0f), hash(n+  1.0f),f.x),
                        mix( hash(n+ 57.0f), hash(n+ 58.0f),f.x),f.y),
                    mix(mix( hash(n+113.0f), hash(n+114.0f),f.x),
                        mix( hash(n+170.0f), hash(n+171.0f),f.x),f.y),f.z);
    return res;
}

static __inline__ __device__ float fbm( vec3 p , int layer=6)
{
    float f = 0.0;
    mat3 m = mat3( 0.00f,  0.80f,  0.60f,
                  -0.80f,  0.36f, -0.48f,
                  -0.60f, -0.48f,  0.64f );
    vec3 pp = p;
    float coef = 0.5f;
    for(int i=0;i<layer;i++) {
        f += coef * noise(pp);
        pp = m * pp *2.02f;
        coef *= 0.5f;
    }
    return f/0.9375f;
}
static __inline__ __device__
    mat3 rot(float deg){
    return mat3(cos(deg),-sin(deg),0,
                sin(deg), cos(deg),0,
                0,0,1);

}

static __inline__ __device__ vec3 proceduralSky2(vec3 dir, vec3 sunLightDir, float t)
{

    float bright = 1*(1.8f-0.55f);
    float color1 = fbm((dir*3.5f)-0.5f);  //xz
    float color2 = fbm((dir*7.8f)-10.5f); //yz

    float clouds1 = smoothstep(1.0f-0.55f,fmin((1.0f-0.55f)+0.28f*2.0f,1.0f),color1);
    float clouds2 = smoothstep(1.0f-0.55f,fmin((1.0f-0.55f)+0.28f,1.0f),color2);

    float cloudsFormComb = saturate(clouds1+clouds2);
    vec3 sunCol = vec3(258.0, 208.0, 100.0) / 15.0f;

    vec4 skyCol = vec4(0.6,0.8,1.0,1.0);
    float cloudCol = saturate(saturate(1.0-pow(color1,1.0f)*0.2f)*bright);
    vec4 clouds1Color = vec4(cloudCol,cloudCol,cloudCol,1.0f);
    vec4 clouds2Color = mix(clouds1Color,skyCol,0.25f);
    vec4 cloudColComb = mix(clouds1Color,clouds2Color,saturate(clouds2-clouds1));
    vec4 clouds = vec4(0.0);
    clouds = mix(skyCol,cloudColComb,cloudsFormComb);

    vec3 localRay = normalize(dir);
    float sunIntensity = 1.0f - (dot(localRay, sunLightDir) * 0.5f + 0.5f);
    sunIntensity = 0.2f / sunIntensity;
    sunIntensity = fmin(sunIntensity, 40000.0f);
    sunIntensity = fmax(0.0f, sunIntensity - 3.0f);
    //return vec3(0,0,0);
    return vec3(clouds)*0.5f + sunCol * (sunIntensity*0.0000075f);
}

// ####################################
#define sun_color vec3(1.f, .7f, .55f)
static __inline__ __device__ vec3 render_sky_color(vec3 rd, vec3 sunLightDir)
{
	float sun_amount = fmax(dot(rd, normalize(sunLightDir)), 0.0f);
	vec3 sky = mix(vec3(.0f, .1f, .4f), vec3(.3f, .6f, .8f), 1.0f - rd.y);
	sky = sky + sun_color * fmin(powf(sun_amount, 1500.0f) * 5.0f, 1.0f);
	sky = sky + sun_color * fmin(powf(sun_amount, 10.0f) * .6f, 1.0f);
	return sky;
}
struct ray {
	vec3 origin;
	vec3 direction;
};
struct sphere {
	vec3 origin;
	float radius;
	int material;
};
struct hit_record {
	float t;
	int material_id;
	vec3 normal;
	vec3 origin;
};
static __inline__ __device__ void intersect_sphere(
	ray r,
	sphere s,
    hit_record& hit
){
	vec3 oc = s.origin - r.origin;
    float a  = dot(r.direction, r.direction);
	float b = 2 * dot(oc, r.direction);
	float c = dot(oc, oc) - s.radius * s.radius;
    float discriminant = b*b - 4*a*c;
	if (discriminant < 0) return;

    float t = (-b - sqrt(discriminant) ) / (2.0f*a);

	hit.t = t;
	hit.material_id = s.material;
	hit.origin = r.origin + t * r.direction;
	hit.normal = (hit.origin - s.origin) / s.radius;
}
static __inline__ __device__ float softlight(float base, float blend, float c)
{
    return (blend < c) ? (2.0 * base * blend + base * base * (1.0f - 2.0f * blend)) : (sqrt(base) * (2.0f * blend - 1.0f) + 2.0f * base * (1.0f - blend));
}
static __inline__ __device__ float density(vec3 pos, vec3 windDir, float coverage, float t, float freq = 1.0f, int layer = 6)
{
	// signal
	vec3 p = 2.0f *  pos * .0212242f * freq; // test time
        vec3 pertb = vec3(noise(p*16), noise(vec3(p.x,p.z,p.y)*16), noise(vec3(p.y, p.x, p.z)*16)) * 0.05f;
	float dens = fbm(p + pertb + windDir * t, layer); //, FBM_FREQ);;

	float cov = 1.f - coverage;
//	dens = smoothstep (cov-0.1, cov + .1, dens);
//        dens = softlight(fbm(p*4 + pertb * 4  + windDir * t), dens, 0.8);
        dens *= smoothstep (cov, cov + .1f, dens);
	return pow(clamp(dens, 0.f, 1.f),0.5f);
}
static __inline__ __device__ float light(
	vec3 origin,
    vec3 sunLightDir,
    vec3 windDir,
    float coverage,
    float absorption,
    float t,
    float freq = 1.0
){
	const int steps = 4;
	float march_step = 0.5;

	vec3 pos = origin;
	vec3 dir_step = -sunLightDir * march_step;
	float T = 1.; // transmitance
        float coef = 1.0;
	for (int i = 0; i < steps; i++) {
		float dens = density(pos, windDir, coverage, t, freq,6);

		float T_i = exp(-absorption * dens * coef * march_step);
		T *= T_i;
		//if (T < .01) break;

		pos = vec3(
            pos.x + coef * dir_step.x,
            pos.y + coef * dir_step.y,
            pos.z + coef * dir_step.z
        );
            coef *= 2.0f;
	}

	return T;
}
#define SIMULATE_LIGHT
#define FAKE_LIGHT
#define max_dist 1e8f
static __inline__ __device__ vec4 render_clouds(
    ray r, 
    vec3 sunLightDir,
    vec3 windDir, 
    int steps, 
    float coverage, 
    float thickness, 
    float absorption, 
    float t
){
    //r.direction.x = r.direction.x * 2.0f;
    vec3 C = vec3(0, 0, 0);
    float alpha = 0.;
    float s = mix(30, 10, sqrtf(r.direction.y));
    float march_step = thickness / floor(s) / 2;
    vec3 dir_step = r.direction / sqrtf(r.direction.y)  * march_step ;

    sphere atmosphere = {
        vec3(0,-350, 0),
        500., 
        0
    };
    hit_record hit = {
        float(max_dist + 1e1),  // 'infinite' distance
        -1,                     // material id
        vec3(0., 0., 0.),       // normal
        vec3(0., 0., 0.)        // origin
    };

    intersect_sphere(r, atmosphere, hit);
	vec3 pos = hit.origin;
    float talpha = 0;
    float T = 1.; // transmitance
    float coef = 1.0;
    for (int i =0; i < int(s)/2; i++)
    {
        float freq = mix(0.5f, 1.0f, smoothstep(0.0f, 0.5f, r.direction.y));
        float dens = density(pos, windDir, coverage, t, freq);
        dens = mix(0.0f,dens, smoothstep(0.0f, 0.2f, r.direction.y));
        float T_i = exp(-absorption * dens * coef *  2.0f* march_step);
        T *= T_i;
        if (T < .01f)
            break;
        talpha += (1.f - T_i) * (1.f - talpha);
        pos = vec3(
            pos.x + coef * 2.0f* dir_step.x,
            pos.y + coef * 2.0f* dir_step.y,
            pos.z + coef * 2.0f* dir_step.z
        );
        coef *= 1.0f;
        if (length(pos) > 1e3f) break;
    }

        //vec3 pos = r.direction * 500.0f;
    pos = hit.origin;
        alpha = 0;
        T = 1.; // transmitance
        coef = 1.0;
    if (talpha > 1e-3f) {
        for (int i = 0; i < int(s); i++) {
            float h = float(i) / float(steps);
            float freq = mix(0.5f, 1.0f, smoothstep(0.0f, 0.5f, r.direction.y));
            float dens = density(pos, windDir, coverage, t, freq);
            dens = mix(0.0f, dens, smoothstep(0.0f, 0.2f, r.direction.y));
            float T_i =

                exp(-absorption * dens * coef * march_step);
            T *= T_i;
            if (T < .01f)
                break;
            float C_i;

                C_i = T *
#ifdef SIMULATE_LIGHT
                      light(pos, sunLightDir, windDir, coverage, absorption, t, freq) *
#endif
                      // #ifdef FAKE_LIGHT
                      // 			(exp(h) / 1.75) *
                      // #endif
                      dens * march_step;

                C = vec3(C.x + C_i, C.y + C_i, C.z + C_i);
                alpha += (1.f - T_i) * (1.f - alpha);
                pos = vec3(pos.x + coef * dir_step.x,
                           pos.y + coef * dir_step.y,
                           pos.z + coef * dir_step.z);
                coef *= 1.0f;
                if (length(pos) > 1e3f)
                    break;
            }
        }
    return vec4(C.x, C.y, C.z, alpha);
}

static __inline__ __device__ vec3 proceduralSky(
    vec3 dir, 
    vec3 sunLightDir, 
    vec3 windDir,
    int steps,
    float coverage, 
    float thickness,
    float absorption,
    float t
){
    vec3 col = vec3(0,0,0);

    vec3 r_dir = normalize(dir);
    ray r = {vec3(0,0,0), r_dir};
    
    vec3 sky = render_sky_color(r.direction, sunLightDir);
    if(r_dir.y<-0.001f) return sky; // just being lazy

    vec4 cld = render_clouds(r, sunLightDir, windDir, steps, coverage, thickness, absorption, t);
    col = mix(sky, vec3(cld)/(0.000001f+cld.w), cld.w);
    return col;
}

static __inline__ __device__ vec3 hdrSky2(
    vec3 dir
){
  dir = dir
            .rotY(to_radians(params.sky_rot_y))
            .rotX(to_radians(params.sky_rot_x))
            .rotZ(to_radians(params.sky_rot_z))
            .rotY(to_radians(params.sky_rot));
  float u = atan2(dir.z, dir.x)  / 3.1415926f * 0.5f + 0.5f;
  float v = asin(dir.y) / 3.1415926f + 0.5f;
  vec3 col = vec3(0);
  for(int jj=-2;jj<=2;jj++)
  {
    for(int ii=-2;ii<=2;ii++)
    {
      float dx = (float)ii / (float)(params.skynx);
      float dy = (float)jj / (float)(params.skyny);
      col = col + (vec3)texture2D(params.sky_texture, vec2(u + dx, v + dy)) * params.sky_strength;
    }
  }

  return col/9.0f;
}

static __inline__ __device__ vec3 hdrSky(
        vec3 dir, float upperBound,  float isclamp, float &pdf
){
    dir = dir
            .rotY(to_radians(params.sky_rot_y))
            .rotX(to_radians(params.sky_rot_x))
            .rotZ(to_radians(params.sky_rot_z))
            .rotY(to_radians(params.sky_rot));
    float u = atan2(dir.z, dir.x)  / 3.1415926f * 0.5f + 0.5f;
    float v = asin(dir.y) / 3.1415926f + 0.5f;
    vec3 col = (vec3)texture2D(params.sky_texture, vec2(u, v)) * params.sky_strength;
    vec3 col2 = clamp(col, vec3(0.0f), vec3(upperBound));
    int i = u * params.skynx;
    int j = v * params.skyny;
    float p = params.skycdf[params.skynx * params.skyny + j * params.skynx + i];
    pdf = p / (2.0f * M_PIf * M_PIf * sin(((float)j+0.5f)/(float)params.skyny*M_PIf));
    return mix(col, col2, isclamp);
}

static __inline__ __device__ vec3 colorTemperatureToRGB(float temperatureInKelvins)
{
    vec3 retColor;

    temperatureInKelvins = clamp(temperatureInKelvins, 1000.0f, 40000.0f) / 100.0f;

    if (temperatureInKelvins <= 66.0f)
    {
        retColor.x = 1.0f;
        retColor.y = saturate(0.39008157876901960784f * log(temperatureInKelvins) - 0.63184144378862745098f);
    }
    else
    {
        float t = temperatureInKelvins - 60.0f;
        retColor.x = saturate(1.29293618606274509804f * pow(t, -0.1332047592f));
        retColor.y = saturate(1.12989086089529411765f * pow(t, -0.0755148492f));
    }

    if (temperatureInKelvins >= 66.0f)
        retColor.z = 1.0;
    else if(temperatureInKelvins <= 19.0f)
        retColor.z = 0.0;
    else
        retColor.z = saturate(0.54320678911019607843f * log(temperatureInKelvins - 10.0f) - 1.19625408914f);

    return retColor;
}
static __inline__ __device__ vec3 envSky2(vec3 dir)
{
  return hdrSky2(dir);
}
static __inline__ __device__ vec3 envSky(
    vec3 dir,
    vec3 sunLightDir,
    vec3 windDir,
    int steps,
    float coverage,
    float thickness,
    float absorption,
    float t,
    float &pdf,
    float upperBound = 100.0f,
    float isclamp = 0.0f
){
    vec3 color;
    if (!params.usingHdrSky) {
        color = proceduralSky(
            dir,
            sunLightDir,
            windDir,
            steps,
            coverage,
            thickness,
            absorption,
            t
        );
    }
    else {
        color = hdrSky(
            dir, upperBound, isclamp, pdf
        );
    }
    if (params.colorTemperatureMix > 0) {
        vec3 colorTemp = colorTemperatureToRGB(params.colorTemperature);
        colorTemp = mix(vec3(1, 1, 1), colorTemp, params.colorTemperatureMix);
        color = color * colorTemp;
    }
    return color;
}
