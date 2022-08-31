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
        isotropic
    };

    static __inline__ __device__ 
    vec3 CalculateExtinction(vec3 apparantColor, float scatterDistance)
    {

        return -log(clamp(apparantColor, vec3(0.00001), vec3(0.99999)))/scatterDistance;

    }

    static __inline__ __device__
    void world2local(vec3& v, vec3 T, vec3 B, vec3 N){
        v = normalize(vec3(dot(T,v), dot(B,v), dot(N,v)));
    }

    static __inline__ __device__ 
    void pdf(
        float metallic,
        float specTrans,
        float clearCoat,
        float &pSpecular, 
        float &pDiffuse,
        float &pClearcoat,
        float &pSpecTrans)
    {
        float metallicBRDF   = metallic;
        float specularBSDF   = ( 1.0f - metallic ) * specTrans;
        float dielectricBRDF = ( 1.0f - specTrans ) * ( 1.0f - metallic );

        float specularW      = metallicBRDF + dielectricBRDF;
        float transmissionW  = specularBSDF;
        float diffuseW       = dielectricBRDF;
        float clearcoatW     = clearCoat;

        float norm = 1.0f/(specularW + transmissionW + diffuseW + clearcoatW);

        pSpecular  = specularW      * norm;
        pSpecTrans = transmissionW  * norm;
        pDiffuse   = diffuseW       * norm;
        pClearcoat = clearcoatW     * norm;
    }


    static __inline__ __device__ 
    vec3 EvaluateSheen(vec3 baseColor, float sheen, float sheenTint, float HoL)
    {
        if(sheen<=0.0f)
        {
            return vec3(0.0f);
        }
        vec3 tint = BRDFBasics::CalculateTint(baseColor);
        return sheen * mix(vec3(1.0f), tint, sheenTint) * BRDFBasics::fresnel(HoL);
    }

    static __inline__ __device__
    float EvaluateClearcoat(
        float clearcoat, 
        float alpha,
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
        float f  = BRDFBasics::fresnelSchlick(0.04f, HoL);
        float gl = BRDFBasics::GGX(NoL, 0.25f);
        float gv = BRDFBasics::GGX(NoV, 0.25f);

        fPdfW = d / (4.0f * abs(HoV));
        rPdfW = d / (4.0f * abs(HoL));

        return 0.25f * clearcoat * d * f * gl * gv;
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
        float d  = BRDFBasics::GgxAnisotropicD(wm, ax, ay);
        float gl = BRDFBasics::SeparableSmithGGXG1(wi, wm, ax, ay);
        float gv = BRDFBasics::SeparableSmithGGXG1(wo, wm, ax, ay);

        vec3 f = DisneyFresnel(baseColor, metallic, ior, specularTint, HoV, HoL, is_inside);
        BRDFBasics::GgxVndfAnisotropicPdf(wi, wm, wo, ax, ay, fPdf, rPdf);
        //fPdf = abs(NoL) * gv * d / abs(NoL);
        //rPdf = abs(NoV) * gl * d / abs(NoV);
        fPdf *= (1.0f / (4 * abs(HoV)));
        rPdf *= (1.0f / (4 * abs(HoL)));

        return d * gl * gv * f / (4.0f * NoL * NoV);
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


        float n2 = ior * ior;

        vec3 wm = normalize(wi + ior*wo);

        float NoL = abs(wi.z);
        float NoV = abs(wo.z);
        float HoL = abs(dot(wm, wi));
        float HoV = abs(dot(wm, wo));

        float d  = BRDFBasics::GgxAnisotropicD(wm, ax, ay);

        
        float gl = BRDFBasics::SeparableSmithGGXG1(wi, wm, ax, ay);
        float gv = BRDFBasics::SeparableSmithGGXG1(wo, wm, ax, ay);

        
        float F = BRDFBasics::fresnelDielectric(dot(wm, wo), 1.0f, ior, false);
        vec3 color;
        if(thin)
            color = sqrt(baseColor);
        else
            color = baseColor;

        float c = (HoL * HoV) / (NoL * NoV);
        float t = (n2 / pow(dot(wm, wi) + ior * dot(wm, wo), 2.0f));
        //if(length(wm) < 1e-5){
        //    return color * (1.0f - F);
        //}
        
        return color * c * t *  (1.0f - F) * gl * gv * d; 
        //return color ;
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

        if(thin && flatness > 0.0f) {
            float a = roughness * roughness;

            float HoL = dot(wm, wi);
            float fss90 = HoL * HoL * roughness;
            float fss = mix(1.0f, fss90, fl) * mix(1.0f, fss90, fv);

            float ss = 1.25f * (fss * (1.0f / (NoL + NoV) - 0.5f) + 0.5f);
            h = ss;
        }

        float lambert = 1.0f;
        float retro = EvaluateDisneyRetroDiffuse(roughness, wi, wo);
        float subsurfaceApprox = mix(lambert, h, thin ? flatness : 0.0f);

        return 1.0f/M_PIf * (retro + subsurfaceApprox * (1.0f - 0.5f * fl) * (1.0f - 0.5f * fv));
    }

    static __inline__ __device__
    float3 EvaluateDisney(
        vec3 baseColor,
        float metallic,
        float subsurface,
        float specular,
        float roughness,
        float specularTint,
        float anisotropic,
        float sheen,
        float sheenTint,
        float clearCoat,
        float clearcoatGloss,

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
        Onb tbn = Onb(N);
        world2local(wi, tbn.m_tangent ,tbn.m_binormal, N);
        world2local(wo, tbn.m_tangent ,tbn.m_binormal, N);
        vec3 wm = normalize(wo+wi);

        float NoL = wi.z;
        float NoV = wo.z;
        float NoH = wm.z;
        float HoL = dot(wm,wi);
        
        float3 reflectance = make_float3(0.0f,0.0f,0.0f);
        fPdf = 0.0f;
        rPdf = 0.0f;

        float pSpecular,pDiffuse,pClearcoat,pSpecTrans;
        pdf(metallic,specTrans,clearCoat,pSpecular,pDiffuse,pClearcoat,pSpecTrans);

        // calculate all of the anisotropic params 
        float ax,ay;
        BRDFBasics::CalculateAnisotropicParams(roughness, anisotropic, ax, ay);

        float diffuseW = (1.0f - metallic) * (1.0f - specTrans);
        float transmissionW = (1.0f - metallic) * specTrans;


        // Clearcoat
     
        bool upperHemisphere = NoL > 0.0f && NoV > 0.0f;

        if(upperHemisphere && clearCoat > 0.0f) {
            float forwardClearcoatPdfW;
            float reverseClearcoatPdfW;
            float clearcoat = EvaluateClearcoat(clearCoat,clearcoatGloss,NoH,NoL,NoV,HoL,HoL,forwardClearcoatPdfW,reverseClearcoatPdfW);
            fPdf += pClearcoat * forwardClearcoatPdfW;
            rPdf += pClearcoat * reverseClearcoatPdfW;
            reflectance += make_float3(clearcoat,clearcoat,clearcoat);
        }
        // Diffuse

        if(diffuseW > 0.0f){
            float forwardDiffusePdfW = abs(wi.z);
            float reverseDiffusePdfW = abs(wo.z);
            float diffuse = EvaluateDisneyDiffuse(roughness,flatness, wi, wo, wm, thin);

            vec3 lobeOfSheen =  EvaluateSheen(baseColor,sheen,sheenTint, HoL);

            fPdf += pDiffuse * forwardDiffusePdfW;
            rPdf += pDiffuse * reverseDiffusePdfW;
            if(!thin && nDl<=0.0f)
                diffuse = 0;
            reflectance += diffuseW * (diffuse * baseColor + lobeOfSheen);
        }
        // Transsmission
        if(transmissionW > 0.0f) {
            float rscaled = thin ? BRDFBasics::ThinTransmissionRoughness(ior, roughness) : roughness;
            float tax, tay;
            BRDFBasics::CalculateAnisotropicParams(rscaled, anisotropic, tax, tay);

            float3 transmission = EvaluateDisneySpecTransmission(baseColor,metallic,ior,specularTint,roughness, tax, tay, thin, is_inside,wo,wi);
            reflectance += transmissionW * transmission;

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
            vec3 Spec = EvaluateDisneyBRDF(baseColor,  metallic, subsurface,  specular, roughness, specularTint, anisotropic, sheen, sheenTint, clearCoat, clearcoatGloss, ior, is_inside, wi, wo, forwardMetallicPdfW, reverseMetallicPdfW);

            reflectance += Spec;
            fPdf += pSpecular * forwardMetallicPdfW / (4 * abs(HoL) );
            rPdf += pSpecular * reverseMetallicPdfW / (4 * abs(HoL));
        }

        reflectance = reflectance * abs(NoL);

        return reflectance;

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
        float ax, ay;
        BRDFBasics::CalculateAnisotropicParams(roughness, anisotropic, ax, ay);
        float2 r01 = sobolRnd(seed);
        float r0 = r01.x;//rnd(seed);
        float r1 = r01.y;//rnd(seed);
        vec3 wm = BRDFBasics::SampleGgxVndfAnisotropic(wo, ax, ay, r0, r1);

        wi = normalize(reflect(-wo, wm)); 
        if(wi.z<0.0f || abs(wi.z) < 1e-5)
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

        tbn.inverse_transform(wi);
        wi = normalize(wi);

        BRDFBasics::GgxVndfAnisotropicPdf(wi, wm, wo, ax, ay, fPdf, rPdf);
        fPdf *= (1.0f / (4 * abs(dot(wo, wm))));
        rPdf *= (1.0f / (4 * abs(dot(wi, wm))));


        return true;
    }

    static __inline__ __device__
    bool SampleDisneyClearCoat(
            unsigned int &seed,
            float clearCoat,
            float clearcoatGloss,
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
        float a2 = 0.0625; //0.25 * 0.25

        float2 r01 = sobolRnd(seed);
        float r0 = r01.x;//rnd(seed);
        float r1 = r01.y;//rnd(seed);

        float cosTheta = sqrt( max(0.0f, (1.0f - pow(a2, 1.0f - r0) ) / (1.0f -a2) ) );
        float sinTheta = sqrt( max(0.0f, 1.0f - cosTheta * cosTheta) );


        float phi = 2.0f * M_PIf * r1;

        vec3 wm = vec3(sinTheta * cosf(phi), sinTheta * sinf(phi), cosTheta);
        if(dot(wm,wo) < 0.0f){
            wm = -wm;
        }

        wi = normalize(reflect(-wo,wm));

        if(dot(wi,wo) < 0.0f){ //removable?
            return false;
        }

        float NoH = wm.z;
        float LoH = dot(wm,wi);
        float NoL  = abs(wi.z);
        float NoV = abs(wo.z);

        //float d = BRDFBasics::GTR1(abs(NoH),lerp(0.1f, 0.001f, clearcoatGloss));
        float d = BRDFBasics::GTR1(abs(NoH),(0.1f + clearcoatGloss * (0.001f-0.1f) ));
        float f = BRDFBasics::fresnelSchlick(LoH,0.04f);
        float g = BRDFBasics::SeparableSmithGGXG1(wi,  wm, 0.25f, 0.25f);

        fPdf = d / (4.0f * dot(wo,wm));
        rPdf = d /(4.0f * LoH);
        reflectance = vec3(0.25f * clearCoat * g * f *d ) / rPdf;

        Onb  tbn = Onb(N);
        tbn.inverse_transform(wi);
        wi = normalize(wi);
        return true;
        
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
            int& phaseFuncion,
            vec3& extinction,
            bool thin,
            bool is_inside,
            vec3 T,
            vec3 B,
            vec3 N

            )
    {
        if(abs(wo.z) <= 1e-6){
            fPdf = 0.0f;
            rPdf = 0.0f;
            reflectance = vec3(0.0f);
            wi = vec3(0.0f);
            return false;
        }
        float rscaled = thin ? BRDFBasics::ThinTransmissionRoughness(ior,  roughness) : roughness;

        float tax,tay;
        BRDFBasics::CalculateAnisotropicParams(rscaled,anisotropic,tax,tay);

        float2 r01 = sobolRnd(seed);
        float r0 = r01.x;//rnd(seed);
        float r1 = r01.y;//rnd(seed);
        vec3 wm = BRDFBasics::SampleGgxVndfAnisotropic(wo, tax, tay, r0, r1);

        float VoH = dot(wm,wo);
        if(wm.z < 0.0f){
            VoH = -VoH;
        }

        float ni = wo.z > 0.0f ? 1.0f : ior;
        float nt = wo.z > 0.0f ? ior : 1.0f;
        float relativeIOR = ni / nt;

        float F = BRDFBasics::fresnelDielectric(VoH, 1.0f, ior, is_inside);

        float G1v = BRDFBasics::SeparableSmithGGXG1(wo, wm, tax, tay);

        float pdf;

        if(rnd(seed) <= F){
            wi = normalize(reflect(-wo,wm));

            flag = scatterEvent; // scatter event
            reflectance = G1v * baseColor;

            //fPdf *= (1.0f / (4 * abs(dot(wo, wm))));
            float jacobian = 4 * abs(VoH)  + 1e-5;
            pdf = F / jacobian;

        }else{
            if(thin){
                wi = normalize(reflect(-wo,wm));
                wi.z = -wi.z;
                reflectance = G1v * sqrt(transmittanceColor);
                flag = scatterEvent;
            }else{
                if( Transmit(wm, wo,relativeIOR, wi)){
                    flag = transmissionEvent;
                    phaseFuncion = (!is_inside)  ? isotropic : vacuum;
                    extinction = CalculateExtinction(transmittanceColor, scatterDistance);
                }else{
                    flag = scatterEvent;
                    wi = normalize(reflect(-wo,wm));
                }
                reflectance = G1v * vec3(1.0f);    
            }
            float LoH = abs(dot(wi,wm));
            float jacobian = LoH  / (pow(LoH + relativeIOR * VoH, 2.0f) + 1e-5) + 1e-5;
            pdf = (1.0f - F) / jacobian;

        }

        if(abs(wi.z) <= 1e-6){
            fPdf = 0.0f;
            rPdf = 0.0f;
            reflectance = vec3(0.0f);
            wi = vec3(0.0f);
            return false;
        }

        //if(roughness < 0.01f){
        //    * (int*) (&flag) |= 0x04; // flag |= SurfaceEventFlags::diracEvent ? 
        //}

        BRDFBasics::GgxVndfAnisotropicPdf(wi, wm, wo, tax, tay, fPdf, rPdf);
        fPdf *= pdf;
        rPdf *= pdf;


        Onb  tbn = Onb(N);
        tbn.inverse_transform(wi);
        wi = normalize(wi);
        return true;
    }
    static __inline__ __device__ 
    bool SampleDisneyDiffuse(
        unsigned int& seed,
        vec3 baseColor,
        vec3 transmittanceColor,
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
        int& phaseFuncion,
        vec3& extinction,
        bool is_inside,
        bool &isSS

            )
    {

        // float2 r01 = sobolRnd(seed);
        // float r0 = r01.x;//rnd(seed);
        // float r1 = r01.y;//rnd(seed);
        wi =  normalize(BRDFBasics::sampleOnHemisphere(seed, 1.0f));
        vec3 wm = normalize(wi+wo);
        float NoL = wi.z;
        if(abs(NoL)<1e-5 ){
            fPdf = 0.0f;
            rPdf = 0.0f;
            reflectance = vec3(0.0f);
            wi = vec3(0.0f);
            return false;
        }

        float NoV = wo.z;

        vec3 color = baseColor;
        float pdf;

        flag = scatterEvent;

        if(rnd(seed) <= subsurface && subsurface > 0.001f){
            wi = -wi;
            pdf = subsurface;
            isSS = true;
            if(thin){
                color = sqrt(transmittanceColor);
            }else{
                flag = transmissionEvent;
                phaseFuncion = (!is_inside)  ? isotropic : vacuum;
                extinction = CalculateExtinction(sssColor, scatterDistance);
                color = transmittanceColor;

            }
        }else{
            pdf = 1.0 - subsurface;
        }

        float HoL = dot(wm,wo);
        vec3 sheenTerm = EvaluateSheen(baseColor, sheen, sheenTint, HoL);
        float diff = EvaluateDisneyDiffuse(roughness, flatness, wi, wo, wm, thin);

        reflectance = sheen + color * diff;
        fPdf = abs(NoL) * pdf;
        rPdf = abs(NoV) * pdf;
        Onb  tbn = Onb(N);
        tbn.inverse_transform(wi);
        wi = normalize(wi);
        return true;
    }
    static __inline__ __device__
    float SampleDistance(unsigned int &seed, float scatterDistance, float &pdf)
    {
        if(scatterDistance > 99){
            return 1e6f;
        }
        float s = -log(rnd(seed)) * scatterDistance;
        pdf = 1.0f;
        return s;
    }

    static __inline__ __device__
    vec3 SampleScatterDirection(unsigned int &seed)
    {
        float2 r01 = sobolRnd(seed);
        float r0 = r01.x;//rnd(seed);
        float r1 = r01.y;//rnd(seed);

        float u = 2.0f * r1 - 1.0f;
        float norm = sqrtf(max(0.0f, 1.0f - u * u));
        float theta = 2.0f * M_PIf * r0;

        return normalize(vec3(norm * cos(theta), norm * sin(theta), u));
    }

    static __inline__ __device__
    vec3 Transmission(vec3 extinction, float distance)
    {
        float tr = exp(-extinction.x * distance);
        float tg = exp(-extinction.y * distance);
        float tb = exp(-extinction.z * distance);

        return vec3(tr, tg, tb);
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
        float sheen,
        float sheenTint,
        float clearCoat,
        float clearcoatGloss,
        float flatness,
        float specTrans,
        float scatterDistance,
        float ior,

        vec3 T,
        vec3 B,
        vec3 N,
        vec3 wo,
        bool thin,
        bool is_inside,
        vec3& wi,
        vec3& reflectance,
        float& rPdf,
        float& fPdf,
        SurfaceEventFlags& flag,
        int& phaseFuncion,
        vec3& extinction,
        bool& isDiff,
        bool& isSS
            )
    {
        Onb  tbn = Onb(N);
        world2local(wo, tbn.m_tangent, tbn.m_binormal, N);
        float pSpecular,pDiffuse,pClearcoat,pSpecTrans;

        pdf(metallic, specTrans, clearCoat, pSpecular, pDiffuse, pClearcoat, pSpecTrans);

        bool success = false;
        isDiff = false;
        float pLobe = 0.0f;
        float p = rnd(seed);
        if( p<= pSpecular){
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
            pLobe = pSpecular;

        }else if(pClearcoat >0.001f && p <= (pSpecular + pClearcoat)){
            success = SampleDisneyClearCoat(seed, clearCoat, clearcoatGloss, T, B, N, wo, wi, reflectance, fPdf, rPdf);
            pLobe = pClearcoat;
            isDiff = true;
        }else if(pSpecTrans > 0.001f && p <= (pSpecular + pClearcoat + pSpecTrans)){
            success = SampleDisneySpecTransmission(seed, ior, roughness, anisotropic, baseColor, transmiianceColor, scatterDistance, wo, wi, rPdf, fPdf, reflectance, flag, phaseFuncion, extinction, thin, is_inside, T, B, N);
            pLobe = pSpecTrans;
        }else {
            isDiff = true;
            success = SampleDisneyDiffuse(seed, baseColor, transmiianceColor, sssColor, scatterDistance, sheen, sheenTint, roughness, flatness, subsurface, thin, wo, T, B, N, wi, fPdf, rPdf, reflectance, flag, phaseFuncion, extinction,is_inside, isSS);
            pLobe = pDiffuse;
        }
        //reflectance = clamp(reflectance, vec3(0,0,0), vec3(1,1,1));
        if(pLobe > 0.0f){
            //pLobe = clamp(pLobe, 0.001f, 0.999f);
            //reflectance = reflectance * (1.0f/(pLobe));
            rPdf *= pLobe;
            fPdf *= pLobe;
        }
        return success;

    }
}
