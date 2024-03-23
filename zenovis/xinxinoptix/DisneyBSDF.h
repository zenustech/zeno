#pragma once
#include "zxxglslvec.h"
#include "TraceStuff.h"
#include "IOMat.h"
#include "DisneyBRDF.h"
#include "HairBSDF.h"

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
    setup_subsurface_radius(float eta, vec3 albedo, vec3 &radius, bool fixedRadius)
    {
    	if (fixedRadius) {
			radius = radius * 0.25f / M_PIf;
			return;
		}

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
    void CalculateExtinction2(vec3 albedo, vec3 radius, vec3 &sigma_t, vec3 &alpha, float eta, bool fixedRadius)
    {
        vec3 r = radius;
        setup_subsurface_radius(eta, albedo, r, fixedRadius);
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
    vec3 CalculateExtinction(vec3 apparantColor, float scaler)
    {
        return 1.0/(max(apparantColor * scaler,vec3(0.000001)));
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
    float SampleDistance(unsigned int &seed, float scatterDistance){
        float r = rnd(seed);
        return -log(max(1.0f-rnd(seed),_FLT_MIN_)) * scatterDistance;

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
        float3 EvaluateDisney2(
            vec3 illum,
            struct MatOutput mat,
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
            vec3 &tterm,
            bool reflectance = false)

    {
        bool sameside = (dot(wo, N)*dot(wo, N2))>0.0f;
        if(sameside == false)
        {
          wo = normalize(wo - 1.02f * dot(wo, N) * N);
        }
        float eta = dot(wo, N)>0?mat.ior:1.0f/mat.ior;
        vec3 f = vec3(0.0f);
        fPdf = 0.0f;
        rotateTangent(T, B, N, mat.anisoRotation * 2 * 3.1415926f);
        // Onb tbn = Onb(N);
        world2local(wi, T, B, N);
        world2local(wo, T, B, N);
        world2local(N2, T, B, N);

        bool reflect = (dot(wi, N2) * dot(wo, N2) > 0.0f) || (wi.z * wo.z > 0.0f);

        vec3 Csheen, Cspec0;
        float F0;

        vec3 wm = reflect? normalize(wi + wo):normalize(wi + wo * eta);

        wm = wm.z<0.0f?-wm:wm;
        BRDFBasics::TintColors(mix(mat.basecolor, mat.sssColor, mat.subsurface), eta, mat.specularTint, mat.sheenTint, F0, Csheen, Cspec0);
        Cspec0 = Cspec0;
        //material layer mix weight
        float dielectricWt = (1.0 - mat.metallic) * (1.0 - mat.specTrans);
        float metalWt = mat.metallic;
        float glassWt = (1.0 - mat.metallic) * mat.specTrans;

        float schlickWt = BRDFBasics::SchlickWeight(abs(dot(wo, wm)));
        float F = BRDFBasics::DielectricFresnel(abs(dot(wo, wm)), mat.ior);
        float psss = mat.subsurface;
        float sssPortion = psss / (1.0 + psss);
        //event probability
        float diffPr = dielectricWt;
        float sssPr = dielectricWt  * psss;
        float dielectricPr = dielectricWt * luminance(mix(Cspec0, vec3(1.0f), F) * mat.specular);
        float metalPr = metalWt;
        float glassPr = glassWt;
        float clearCtPr = 0.25 * mat.clearcoat;

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
        if(mat.isHair>0.5f)
        {
          vec3 wo_t = normalize(vec3(0.0f,wo.y,wo.z));
          vec3 wi_t = normalize(vec3(0.0f,wi.y,wi.z));
          float Phi = acos(dot(wo_t, wi_t));
          vec3 extinction = CalculateExtinction(mat.sssParam,1.0f);
          vec3 h = HairBSDF::EvaluteHair(wi.x,dot(wi_t,wi),wo.x,
                                              dot(wo_t,wo),Phi,wi.z,1.55f,
                                              extinction,mat.basecolor,
                                              mat.roughness,0.7f,2.0f);
          fPdf += 1 / M_PIf / 4;
          dterm = dterm + h;
          f = f + h;
          return f * abs(wi.z);
        }
        if(diffPr > 0.0 && reflect)
        {

            vec3 d = BRDFBasics::EvalDisneyDiffuse(thin? mat.basecolor : mix(mat.basecolor,mat.sssColor,mat.subsurface), mat.subsurface, mat.roughness, mat.sheen,
                                             Csheen, wo, wi, wm, tmpPdf) * dielectricWt;
            dterm = dterm + d;
            f = f + d;
            fPdf += tmpPdf * diffPr ;
        }
        if(dielectricPr>0.0 && reflect)
        {
            float F = BRDFBasics::SchlickDielectic(abs(dot(wm, wo)), mat.ior);
            float ax, ay;
            BRDFBasics::CalculateAnisotropicParams(mat.roughness,mat.anisotropic,ax,ay);
            vec3 s = BRDFBasics::EvalMicrofacetReflection(ax, ay, wo, wi, wm,
                                          mix(mix(Cspec0, mat.diffractColor, mat.diffraction), vec3(1.0f), F) * mat.specular,
                                          tmpPdf) * dielectricWt;
            sterm = sterm + s;
            f = f + s;
            fPdf += tmpPdf * dielectricPr;
        }
        if(metalPr>0.0 && reflect)
        {
            vec3 F = mix(mix(mat.basecolor, mat.diffractColor, mat.diffraction), vec3(1.0), BRDFBasics::SchlickWeight(HoV));
            float ax, ay;
            BRDFBasics::CalculateAnisotropicParams(mat.roughness,mat.anisotropic,ax,ay);
            vec3 s = BRDFBasics::EvalMicrofacetReflection(ax, ay, wo, wi, wm, F, tmpPdf) * metalWt;
            tmpPdf *= (mat.roughness<=0.03 && reflectance==false)? 0.0f:1.0f;
            s = s * (tmpPdf>0.0f? 1.0f:0.0f);
            sterm = sterm + s;
            f = f + s;
            fPdf += tmpPdf * metalPr;
        }
        if(glassPr>0.0)
        {
            bool entering = wo.z>0?true:false;

            //float F = BRDFBasics::DielectricFresnel(, eta);
            float ax, ay;
            BRDFBasics::CalculateAnisotropicParams(mat.roughness,mat.anisotropic,ax,ay);
            if (reflect) {

              vec3 wm = normalize(wi + wo);
              float F = BRDFBasics::DielectricFresnel(abs(dot(wm, wo)), entering?mat.ior:1.0/mat.ior);
              vec3 s = BRDFBasics::EvalMicrofacetReflection(ax, ay, wo, wi, wm,
                                                            mix(mix(Cspec0, mat.diffractColor, mat.diffraction), vec3(1.0f), F) * mat.specular,
                                            tmpPdf) * glassWt;
              tmpPdf *= (mat.roughness<=0.03 && reflectance==false)? 0.0f:1.0f;
              s = s * (tmpPdf>0.0f? 1.0f:0.0f);
              sterm = sterm + s;
              f = f + s;
              fPdf += tmpPdf * glassPr;
            } else {
              if(thin)
              {
                vec3 t = sqrt(mix(mat.transColor, mat.diffractColor, mat.diffraction)) * glassWt;
                float tmpPdf = (reflectance==false)? 0.0f:1.0f;
                t = t * (tmpPdf>0.0f?1.0f:0.0f);
                tterm = tterm + t;
                f = f + t;
                fPdf += tmpPdf * glassPr;
              }else {
                vec3 wm = entering?-normalize(wo + mat.ior * wi) : normalize(wo + 1.0f/mat.ior * wi);
                float F = BRDFBasics::DielectricFresnel(abs(dot(wm, wo)), entering?mat.ior:1.0/mat.ior);
                float tmpPdf;
                vec3 brdf = BRDFBasics::EvalMicrofacetRefraction(mix(mat.transColor, mat.diffractColor, mat.diffraction),
                                                                 ax, ay,
                                                                 entering? mat.ior:1.0f/mat.ior,
                                                                 wo, wi, wm,
                                                                 vec3(F), tmpPdf);

                vec3 t = brdf * glassWt;
                tmpPdf *= (mat.roughness<=0.03 && reflectance==false)? 0.0f:1.0f;
                t = t * (tmpPdf>0.0f? 1.0f:0.0f);
                tterm = tterm + t;
                f = f + t;

                fPdf += tmpPdf * glassPr;

              }
            }

        }
        if(clearCtPr>0.0 && reflect)
        {
            vec3 wm = normalize(wi + wo);
            float ax, ay;
            BRDFBasics::CalculateAnisotropicParams(mat.clearcoatRoughness,0,ax,ay);
            //ior related clearCt
            float F = BRDFBasics::SchlickDielectic(abs(dot(wm, wo)), mat.clearcoatIOR);
            vec3 s = mix(vec3(0.04f), vec3(1.0f), F) *
                     BRDFBasics::EvalClearcoat(mat.clearcoatRoughness, wo, wi,
                                               wm, tmpPdf) * 0.25 * mat.clearcoat;
            sterm = sterm + s;
            f =  f + s;
            fPdf += tmpPdf * clearCtPr;
        }
        if((sssPr>0.0&&reflectance) || (sssPr>0.0 && dot(wo, N2)<0.0) || (sssPr>0.0 && (thin)))
        {
          bool trans = (dot(wi, N2) * dot(wo, N2)<0) && (wi.z * wo.z<0);
          float FL = BRDFBasics::SchlickWeight(abs(wi.z));
          float FV = BRDFBasics::SchlickWeight(abs(wo.z));
          float term = wo.z>0?FV:FL;
          float tmpPdf = trans? 1.0f : 0.0f;//0.5/M_PIf:0.0f;
          vec3 transmit = vec3(1.0f);
          if(thin) {
            vec3 color = mix(mat.basecolor, mat.sssColor, mat.subsurface);
            vec3 sigma_t, alpha;
            CalculateExtinction2(color, mat.subsurface * mat.sssParam, sigma_t, alpha, 1.4f, mat.sssFxiedRadius);
            vec3 channelPDF = vec3(1.0f/3.0f);
            transmit = Transmission2(sigma_t * alpha, sigma_t,
                                  channelPDF, 0.001 / (abs(wi.z) + 0.005f), true);
          }
          // vec3 d = 1.0f/M_PIf * (1.0f - 0.5f * term) * (trans?vec3(1.0f):vec3(0.0f))  * dielectricWt * subsurface;
          vec3 d = (trans? vec3(1.0f): vec3(0.0f)) * transmit  * dielectricWt;
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
    float brdf_pdf(vec3 wo, vec3 wm, float roughness, float anisotropic)
    {
      float ax, ay;
      BRDFBasics::CalculateAnisotropicParams(roughness, anisotropic, ax, ay);
      float d = BRDFBasics::GgxD(wm, ax, ay);
      return  d * abs(wm.z) / (4.0f * abs(dot(wo, wm)));
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
    bool SampleDisney2(
        unsigned int& seed,
        unsigned int& eventseed,
        struct MatOutput mat,
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
        float eta = dot(wo, N)>0?mat.ior:1.0f/mat.ior;
        rotateTangent(T, B, N, mat.anisoRotation * 2 * 3.1415926f);
        world2local(wo, T, B, N);
        float2 r = sobolRnd(eventseed);
//        float r1 = r.x;
//        float r2 = r.y;
        float r1 = rnd(seed);
        float r2 = rnd(seed);

        vec3 Csheen, Cspec0;
        float F0;

        BRDFBasics::TintColors(mix(mat.basecolor, mat.sssColor, mat.subsurface), eta, mat.specularTint, mat.sheenTint, F0, Csheen, Cspec0);

        //material layer mix weight
        float dielectricWt = (1.0 - mat.metallic) * (1.0 - mat.specTrans);
        float metalWt = mat.metallic;
        float glassWt = (1.0 - mat.metallic) * mat.specTrans;

        float ax, ay;
        BRDFBasics::CalculateAnisotropicParams(mat.roughness,mat.anisotropic,ax,ay);
        vec3 wm = BRDFBasics::SampleGGXVNDF(wo, ax, ay, r1, r2);
        float hov1 = abs(wo.z);
        float hov2 = abs(dot(wo, wm));
        float c = pow(smoothstep(0.0f,0.2f,mat.roughness),2.0f);

        float hov = mix(hov1, hov2, c);
        float schlickWt = BRDFBasics::SchlickWeight(hov);
        float F = BRDFBasics::DielectricFresnel(hov, mat.ior);
        float psss = mat.subsurface;
        float sssPortion = psss / (1.0f + psss);
        //dielectricWt *= 1.0f - psub;

        //event probability
        float diffPr = dielectricWt ;
        float sssPr = dielectricWt  * psss;
        float dielectricPr = dielectricWt * luminance(mix(Cspec0, vec3(1.0f), F) * mat.specular);
        float metalPr = metalWt;
        float glassPr = glassWt;
        float clearCtPr = 0.25 * mat.clearcoat;

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
        if(mat.isHair>0.5f){
          prd->fromDiff = true;
          wi = SampleScatterDirection(prd->seed) ;
          vec3 wo_t = normalize(vec3(0.0f,wo.y,wo.z));
          vec3 wi_t = normalize(vec3(0.0f,wi.y,wi.z));
          float Phi = acos(dot(wo_t,wi_t));
          vec3 extinction = CalculateExtinction(mat.sssParam,1.0f);
          reflectance = HairBSDF::EvaluteHair(wi.x,dot(wi_t,wi),wo.x,
                                              dot(wo_t,wo),Phi,wi.z,1.55f,
                                              extinction,mat.basecolor,
                                              mat.roughness,0.7f,2.0f);
                    
          isSS = false;
          tbn.inverse_transform(wi);
          wi = normalize(wi);


          float pdf, pdf2;
          pdf = 1 / M_PIf / 4;
          vec3 rd, rs, rt;
          fPdf = pdf>1e-5?pdf:0.0f;
          reflectance = pdf>1e-5?reflectance:vec3(0.0f);
          return true;
        }
       
       
        if(r3<p1){
          prd->hit_type = DIFFUSE_HIT;
          if(wo.z<0 && mat.subsurface>0)//inside, scattering, go out for sure
          {
            wi = BRDFBasics::UniformSampleHemisphere(r1, r2);
            flag = transmissionEvent;
            isSS = false;
            tbn.inverse_transform(wi);
            wi = normalize(wi);

            if (dot(wi, N2) < 0) {
              wi = normalize(wi - 1.01f * dot(wi, N2) * N2);
            }
          }
          else{
            //switch between scattering or diffuse reflection
            float diffp = p0/p1;
            if(rnd(seed)<diffp || prd->fromDiff==true)
            {
              prd->fromDiff = true;
              wi = BRDFBasics::CosineSampleHemisphere(r1, r2);
              isSS = false;
              tbn.inverse_transform(wi);
              wi = normalize(wi);

              if(dot(wi, N2)<0)
              {
                wi = normalize(wi - 1.01f * dot(wi, N2) * N2);
              }
            }else
            {

              //go inside
              wi = -BRDFBasics::UniformSampleHemisphere(r1, r2);
              wi.z = min(-0.2f, wi.z);
              wi = normalize(wi);
              isSS = true;
              flag = transmissionEvent;
              vec3 color = mix(mat.basecolor, mat.sssColor, mat.subsurface);
              color = clamp(color, vec3(0.05), vec3(0.99));
              vec3 sssRadius = mat.sssParam * mat.subsurface;
              RadiancePRD *prd = getPRD();
              prd->ss_alpha = color;
              if (isSS) {
                medium = PhaseFunctions::isotropic;
                CalculateExtinction2(color, sssRadius, prd->sigma_t, prd->ss_alpha, 1.4f, mat.sssFxiedRadius);
              }
              tbn.inverse_transform(wi);
              wi = normalize(wi);

              bool sameside2 = (dot(wi, N) * dot(wi, N2)) > 0.0f;
              if (sameside == false) {
                wi = normalize(wi - 1.01f * dot(wi, N2) * N2);
              }

            }
          }


            if(dot(wi, N2)>0)
            {
              isSS = false;
            }
            //reflectance = vec3(1.0f) * M_PIf ;
            //return true;

        }
        else if(r3<p3)//specular
        {
            prd->hit_type = SPECULAR_HIT;
            float ax, ay;
            BRDFBasics::CalculateAnisotropicParams(mat.roughness,mat.anisotropic,ax,ay);

            vec3 vtmp = wo;
            vtmp.z = wo.z>0?vtmp.z:-vtmp.z;
            vec3 wm = BRDFBasics::SampleGGXVNDF(vtmp, ax, ay, r1, r2);

            if (wm.z < 0.0)
              wm.z = -wm.z;

            wm.z = wo.z>0? wm.z:-wm.z;

            wi = normalize(reflect(-wo, wm));
            tbn.inverse_transform(wi);
            wi = normalize(wi);

            if(dot(wi, N2)<0)
            {
              wi = normalize(wi - 1.01f * dot(wi, N2) * N2);
            }
        }else if(r3<p4)//glass
        {

            bool entering = wo.z>0?true:false;
            float ax, ay;
            BRDFBasics::CalculateAnisotropicParams(mat.roughness,mat.anisotropic,ax,ay);
            vec3 swo = wo.z>0?wo:-wo;
            vec3 wm = BRDFBasics::SampleGGXVNDF(swo, ax, ay, r1, r2);
            wm = wm.z<0?-wm:wm;

            wm = entering?wm:-wm;

            float F = BRDFBasics::DielectricFresnel(abs(dot(wm, wo)), entering?mat.ior:1.0f/mat.ior);
            float p = rnd(seed);
            if(p<F)//reflection
            {
              wi = normalize(reflect(-normalize(wo),wm));
            }else //refraction
            {
              if(thin)
              {
                wi = -wo;
                extinction = vec3(0.0f);
              }else {
                wi = normalize(
                    refract(wo, wm, entering ? 1.0f / mat.ior : mat.ior));
                flag = transmissionEvent;
                isTrans = true;
                extinction =
                    CalculateExtinction(mat.transTint, mat.transTintDepth);
                extinction = entering ? extinction : vec3(0.0f);
              }
            }

            tbn.inverse_transform(wi);
            wi = normalize(wi);
          minSpecRough = mat.roughness;
          auto woo = wo;
          tbn.inverse_transform(woo);
          auto isReflection =  dot(wi, N) * dot(woo, N)>0?1:0;
          prd->hit_type = (isReflection==1?SPECULAR_HIT:TRANSMIT_HIT);
          bool sameside2 = (dot(wi, N) * dot(wi, N2))>0.0f;
          if(sameside2 == false)
          {
            wi = normalize(wi - 1.01f * dot(wi, N2) * N2);
          }

        }else if(r3<p5)//cc
        {
            prd->hit_type = SPECULAR_HIT;
            float ax, ay;
            BRDFBasics::CalculateAnisotropicParams(mat.clearcoatRoughness,0,ax,ay);
            vec3 swo = wo.z>0?wo:-wo;
            vec3 wm = BRDFBasics::SampleGGXVNDF(swo, ax, ay, r1, r2);
            wm = wm.z<0?-wm:wm;


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
        reflectance = EvaluateDisney2(vec3(1.0f), mat, wi, wo, T, B, N, N2, thin,
                                      is_inside, pdf, pdf2, 0, rd, rs, rt, true);
        fPdf = pdf>1e-5?pdf:0.0f;
        reflectance = pdf>1e-5?reflectance:vec3(0.0f);
        return true;
    }

}
