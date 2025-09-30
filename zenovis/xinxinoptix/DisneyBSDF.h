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
        float s = sqrtf(max(3.0f * (1.0f - a), 0.0f));
        return 0.5f * a * (1.0f + expf(-fourthirdA * s)) * expf(-s);
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
    static __inline__ __device__  float burley_fitting(float A)
    {
        return 1.9f - A + 3.5f * (A - 0.8f) * (A - 0.8f);
    }
    static __inline__ __device__  float burley_fitting5(float A)
    {
        return 1.85f - A + 7.0f * fabs((A - 0.8f) * (A - 0.8f) * (A - 0.8f));
    }
    static __inline__ __device__  void bssrdf_burley_setup(const vec3 &albedo, const vec3 &radius, const bool scale_mfp, const int mode, vec3 & radius_out)
    {
        vec3 l = scale_mfp ? 0.25f * radius / M_PIf : radius;
        vec3 A = albedo;
        vec3 s;
        if (mode ==0)
        {
            s = vec3(burley_fitting(A.x), burley_fitting(A.y), burley_fitting(A.z));
        } else {
            s = vec3(burley_fitting5(A.x), burley_fitting5(A.y), burley_fitting5(A.z));
        }

        radius_out = l/s;
    }
    static __inline__ __device__  void bssrdf_setup(const bool burley_radius, const bool scale_mfp,
                                        const bool use_eq5, vec3 & weight, vec3 &albedo,
                                        vec3 &radius, vec3 &diffuse_weight)
    {
        diffuse_weight = vec3(0);
        const float bssrdf_min_radius(1e-8f);
        vec3 kd(0);

        int bssrdf_channels = 3;
        for(int i=0;i<3;i++)
        {
            if(radius[i]<bssrdf_min_radius)
            {
                kd[i] = weight[i];
                weight[i] = 0;
                radius[i] = 0;
                bssrdf_channels--;
            }
        }

        if(bssrdf_channels < 3)
        {
            diffuse_weight = kd;
        }

        if(bssrdf_channels > 0)
        {
            if(burley_radius){
                vec3 updated_radius;
                bssrdf_burley_setup(albedo, radius, scale_mfp, use_eq5, updated_radius);
                radius = updated_radius;
            }
        }

    }
    static __inline__ __device__ void compute_scattering_coeff_from_albedo(float A, float d, float &sigma_s, float &sigma_t)
    {
        float a = 1.0f - expf(A * (-5.09406f + A * (2.61188f - A * 4.31805f)));
        float s = 1.9f - A + 3.5f * pow(A - 0.8f, 2.0f);

        sigma_t = 1.0f / max(d * s, 1e-16f);
        sigma_s = sigma_t * a;
    }
    static __inline__ __device__ void compute_scattering_coeff(const vec3 &weight,
                                                               const vec3& albedo,
                                                               const vec3& radius,
                                                               vec3& sigma_t,
                                                               vec3& sigma_s, vec3& throughput)
    {
        compute_scattering_coeff_from_albedo(albedo.x, radius.x, sigma_t.x,sigma_s.x);
        compute_scattering_coeff_from_albedo(albedo.y, radius.y, sigma_t.y,sigma_s.y);
        compute_scattering_coeff_from_albedo(albedo.z, radius.z, sigma_t.z,sigma_s.z);
        throughput = safe_divide_spectrum(weight, albedo);
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
//        vec3 r = radius;
//        vec3 sigma_s;
//        bssrdf_burley_setup(albedo, radius, true, true, r);
//        compute_scattering_coeff_from_albedo(albedo.x, r.x, sigma_t.x,sigma_s.x);
//        compute_scattering_coeff_from_albedo(albedo.y, r.y, sigma_t.y,sigma_s.y);
//        compute_scattering_coeff_from_albedo(albedo.z, r.z, sigma_t.z,sigma_s.z);
//        alpha = sigma_s/sigma_t;

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
        
        float s = -log(max(1.0f-rnd(seed), _FLT_MIN_)) / max(c, 1e-12f);
        return s;
    }

    static __inline__ __device__
    float SampleDistance(unsigned int &seed, float scatterDistance){
        return -logf(max(1.0f-rnd(seed),_FLT_MIN_)) * scatterDistance;

    }

    static __inline__ __device__
    void world2local(vec3& v, vec3 T, vec3 B, vec3 N){
        v = normalize(vec3(dot(T,v), dot(B,v), dot(N,v)));
    }

    static __inline__ __device__
    void rotateTangent(vec3& _T, vec3& _B, vec3 N, float rotInRadian) {
        vec3 T = normalize(cosf(rotInRadian) * _T - sinf(rotInRadian) * _B);
        vec3 B = normalize(sinf(rotInRadian) * _T + cosf(rotInRadian) * _B);
        _T = T;
        _B = B;
    }

    static __inline__ __device__
    float D_Charlie (float NH, float roughness)
    {
      float invR = 1.0f / roughness;
      float cos2h = NH * NH;
      float sin2h = 1.f - cos2h;
      return (2.0f + invR) * powf(sin2h, invR * 0.5f) / 2.0f / M_PIf;
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
      return a / (1 + b * powf(x, c)) + d * x + e;
    }
    static __inline__ __device__
    float V_Charlie (float NL, float NV, float roughness)
    {
      float lambdaV = NV < 0.5f ? expf(CharlieL(NV, roughness)) : expf(2.f * CharlieL(0.5f, roughness) - CharlieL(1.0f - NV, roughness));
      float lambdaL = NL < 0.5f ? expf(CharlieL(NL, roughness)) : expf(2.f * CharlieL(0.5f, roughness) - CharlieL(1.0f - NL, roughness));
      return 1.f / ((1.f + lambdaV + lambdaL) * (4.f * NV * NL));
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

      wi = normalize((n * c -sqrtf(root)) * wm - n * wo);
      return true;
    }

    static __inline__ __device__
        vec3 SampleScatterDirection(unsigned int &seed)
    {
      //float2 r01 = sobolRnd(seed);
      float r0 = rnd(seed);
      float r1 = rnd(seed);

      float theta = 2.0f * M_PIf * r0;
      float phi = acosf(clamp(1.f - 2.f * r1, -0.9999f, 0.9999f));
      float x = sinf(phi) * cosf(theta);
      float y = sinf(phi) * sinf(theta);
      float z = cosf(phi);

      return normalize(vec3(x, y, z));
    }

    static __inline__ __device__
        vec3 Transmission(const vec3& extinction, float distance)
    {
      return expf(-extinction * distance);
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
      auto result = (hit? transmittance : (sigma_s * transmittance)) / dot(pdf, channelPDF);
      //result = clamp(result,vec3(0.0f),vec3(1.0f));
      return result;
    }
    
    static __inline__ __device__
    vec3 EvaluateDiffuse(vec3 baseColor, float subsurface, float roughness, float sheen, vec3 Csheen, vec3 V, vec3 L, vec3 H, float &pdf){    
      pdf = 0.0f;
      if (L.z == 0.0f)
        return vec3(0.0f);

      float LDotH = abs(dot(L, H));
      float F90 = 0.5f + 2.0f * LDotH * LDotH * roughness;
      // Diffuse
      float FL = BRDFBasics::SchlickWeight(abs(L.z));
      float FV = BRDFBasics::SchlickWeight(abs(V.z));
      float Fd = mix(1.0f,F90,FL) * mix(1.0f,F90,FV);


      // Sheen
      float FH = BRDFBasics::SchlickWeight(LDotH);
      vec3 Fsheen = FH * sheen * Csheen;

      pdf =abs (L.z) * 1.0f / M_PIf;
      return 1.0f / M_PIf * baseColor * (Fd + Fsheen);
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
            bool reflectance = false,
            bool reflection_fromCC = false)

    {
        mat.roughness = reflectance==false?max(0.011f, mat.roughness):mat.roughness;
        //bool sameside = (dot(wo, N)*dot(wo, N2))>0.0f;
//        if(reflectance == true)
//        {
//          N = N2;
//          T = cross(B,N2);
//          B = cross(N2, T);
//        }
        float eta = dot(wo, N)>0?mat.ior:1.0f/mat.ior;
        vec3 f = vec3(0.0f);
        fPdf = 0.0f;
        rotateTangent(T, B, N, mat.anisoRotation * 2 * 3.1415926f);
        // Onb tbn = Onb(N);
        world2local(wi, T, B, N);
        world2local(wo, T, B, N);
        //world2local(N2, T, B, N);

        bool reflect = ( wi.z * wo.z > 0.0f);
        if(reflect && wi.z*wo.z<0)
        {
            wi.z = -wi.z;
        }
        vec3 Csheen, Cspec0;
        float F0;

        vec3 wm = reflect? normalize(wi + wo):normalize(wi + wo * eta);

        wm = wm.z<0.0f?-wm:wm;
        BRDFBasics::TintColors(mix(mat.basecolor, mat.sssColor, mat.subsurface), eta, mat.specularTint, mat.sheenTint, F0, Csheen, Cspec0);
        //material layer mix weight
        float dielectricWt = (1.0 - mat.metallic) * (1.0 - mat.specTrans);
        float metalWt = mat.metallic;
        float glassWt = (1.0 - mat.metallic) * mat.specTrans;

        float schlickWt = BRDFBasics::SchlickWeight(abs(dot(wo, wm)));
        float F = BRDFBasics::DielectricFresnel(abs(dot(wo, wm)), mat.ior);
        float F2 = BRDFBasics::DielectricFresnel(abs(dot(wo, wm)), mat.clearcoatIOR);
        float psss = mat.subsurface;
        float sssPortion = psss / (1.0 + psss);
        //event probability
        float ccweight = reflection_fromCC?1.0f:0.0f;
        float clearCtPr = ccweight * 0.25f * mat.clearcoat * F2;
        float other = (1.0f - clearCtPr) * (1.0 - ccweight);
        float diffPr = other * dielectricWt * (1.0f - luminance(mix(Cspec0, vec3(1.0f), F) * mat.specular)) ;
        float sssPr = other * dielectricWt * (1.0f - luminance(mix(Cspec0, vec3(1.0f), F) * mat.specular))  * psss;
        float dielectricPr = other * dielectricWt * luminance(mix(Cspec0, vec3(1.0f), F) * mat.specular);
        float metalPr = other * metalWt;
        float glassPr = other * glassWt;

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

        if(reflect){
            wm = normalize(wi + wo);
          if(diffPr > 0.0f){
            //vec3 d = EvaluateDiffuse(thin? mat.basecolor : mix(mat.basecolor,mat.sssColor,mat.subsurface), mat.subsurface, mat.roughness, mat.sheen,Csheen, wo, wi, wm, tmpPdf) * dielectricWt;
            vec3 d = BRDFBasics::EvalDisneyDiffuse(thin? mat.basecolor : mix(mat.basecolor,mat.sssColor,mat.subsurface), mat.subsurface, mat.roughness, mat.sheen,Csheen, wo, wi, wm, tmpPdf) * diffPr;
            dterm = dterm + d;
            f = f + d;
            fPdf += tmpPdf * diffPr ;
          }

          if(dielectricPr > 0.0f){
            float F = BRDFBasics::SchlickDielectic(abs(dot(wm, wo)), mat.ior);
            float ax, ay;
            BRDFBasics::CalculateAnisotropicParams(mat.roughness,mat.anisotropic,ax,ay);
            vec3 s = BRDFBasics::EvalMicrofacetReflection(ax, ay, wo, wi, wm,
                                          mix(mix(Cspec0, mat.diffractColor, mat.diffraction), vec3(1.0f), F) * mat.specular,
                                          tmpPdf) * dielectricPr;
            tmpPdf *= F;                              
            sterm = sterm + s;
            f = f + s;
            fPdf += tmpPdf * dielectricPr;
          }

          if(metalPr>0.0f){
            vec3 F = mix(mix(mat.basecolor, mat.diffractColor, mat.diffraction), vec3(1.0), BRDFBasics::SchlickWeight(HoV));
            float ax, ay;
            BRDFBasics::CalculateAnisotropicParams(mat.roughness,mat.anisotropic,ax,ay);
            vec3 s = BRDFBasics::EvalMicrofacetReflection(ax, ay, wo, wi, wm, F, tmpPdf) * metalPr;
            tmpPdf *= (mat.roughness<=0.01 && reflectance==false)? 0.0f:1.0f;
            s = s * (tmpPdf>0.0f? 1.0f:0.0f);
            sterm = sterm + s;
            f = f + s;
            fPdf += tmpPdf * metalPr;
          }

          if(clearCtPr>0.0f){
            vec3 wm = normalize(wi + wo);
            float ax, ay;
            BRDFBasics::CalculateAnisotropicParams(mat.clearcoatRoughness,0,ax,ay);
            //ior related clearCt
            float F = BRDFBasics::SchlickDielectic(abs(dot(wm, wo)), mat.clearcoatIOR);
//            vec3 s = mix(vec3(0.04f), vec3(1.0f), F) *
//                     BRDFBasics::EvalClearcoat(mat.clearcoatRoughness, wo, wi,
//                                               wm, tmpPdf) * clearCtPr;
            vec3 s = BRDFBasics::EvalMicrofacetReflection(ax, ay, wo, wi, wm,
                                          mix(vec3(0.04), vec3(1.0f), F),
                                          tmpPdf) * clearCtPr;
            tmpPdf *= F;
            tmpPdf *= (mat.clearcoatRoughness<=0.01 && reflectance==false)? 0.0f:1.0f;
            sterm = sterm + s;
            f =  f + s;
            fPdf += tmpPdf * clearCtPr;
          }

        }

        if(glassPr>0.0f)
        {
            bool entering = wo.z>0?true:false;

            float ax, ay;
            BRDFBasics::CalculateAnisotropicParams(mat.roughness,mat.anisotropic,ax,ay);
            if (reflect) {

              vec3 wm = normalize(wi + wo);
              float F = BRDFBasics::DielectricFresnel(abs(dot(wm, wo)), entering?mat.ior:1.0f/mat.ior);
              vec3 s = BRDFBasics::EvalMicrofacetReflection(ax, ay, wo, wi, wm,
                                                            mix(mix(Cspec0, mat.diffractColor, mat.diffraction), vec3(1.0f), F) * mat.specular,
                                            tmpPdf) * glassPr;
              tmpPdf *= (mat.roughness<=0.01 && reflectance==false)? 0.0f:F;
              s = s * (tmpPdf>0.0f? 1.0f:0.0f);
              sterm = sterm + s;
              f = f + s;
              fPdf += tmpPdf * glassPr;
            } else {
              if(thin)
              {
                vec3 t = sqrt(mix(mat.transColor, mat.diffractColor, mat.diffraction)) * glassPr / (1e-6+abs(wi.z));
                float F = BRDFBasics::DielectricFresnel(abs(wo.z), entering?mat.ior:1.0f/mat.ior);
                float tmpPdf = reflectance ? (1.0f-F) : 0.0f;
                t = t * (tmpPdf>0.0f?1.0f:0.0f);
                tterm = tterm + t;
                f = f + t;
                fPdf += tmpPdf * glassPr;
              }else {
                vec3 wm = entering?-normalize(wo + mat.ior * wi) : normalize(wo + 1.0f/mat.ior * wi);
                float F = BRDFBasics::DielectricFresnel(abs(dot(wm, wo)), entering?mat.ior:1.0f/mat.ior);
                float tmpPdf;
                vec3 brdf = BRDFBasics::EvalMicrofacetRefraction(mix(mat.transColor, mat.diffractColor, mat.diffraction),
                                                                 ax, ay,
                                                                 entering? mat.ior:1.0f/mat.ior,
                                                                 wo, wi, wm,
                                                                 vec3(F), tmpPdf);

                vec3 t = brdf * glassPr;
                tmpPdf *= (mat.roughness<=0.01 && reflectance==false)? 0.0f:(1.0f - F);
                t = t * (tmpPdf>0.0f? 1.0f:0.0f);
                tterm = tterm + t;
                f = f + t;

                fPdf += tmpPdf * glassPr;

              }
            }

        }
        if(sssPr > 0.0f && (reflectance || dot(wo,N2) < 0.0f || thin)){
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
                                  channelPDF, 0.001f / (abs(wi.z) + 0.005f), true);
          }
          // vec3 d = 1.0f/M_PIf * (1.0f - 0.5f * term) * (trans?vec3(1.0f):vec3(0.0f))  * dielectricWt * subsurface;
          vec3 d = (trans? vec3(1.0f): vec3(0.0f)) * transmit  * sssPr;
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
        float3 EvaluateDisney3(
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
            bool reflectance = false,
            bool reflection_fromCC = false)

    {
        mat.roughness = reflectance==false?max(0.011f, mat.roughness):mat.roughness;
        //bool sameside = (dot(wo, N)*dot(wo, N2))>0.0f;
//        if(reflectance == true)
//        {
//          N = N2;
//          T = cross(B,N2);
//          B = cross(N2, T);
//        }
        float eta = dot(wo, N)>0?mat.ior:1.0f/mat.ior;
        vec3 f = vec3(0.0f);
        fPdf = 0.0f;
        rotateTangent(T, B, N, mat.anisoRotation * 2 * 3.1415926f);
        // Onb tbn = Onb(N);
        vec3 wi_world = wi;
        vec3 wo_world = wo;
        world2local(wi, T, B, N);
        world2local(wo, T, B, N);
        //world2local(N2, T, B, N);

        bool reflect = ( wi.z * wo.z > 0.0f);
        if(reflect && wi.z*wo.z<0)
        {
            wi.z = -wi.z;
        }
        vec3 Csheen, Cspec0;
        float F0;

        vec3 wm = reflect? normalize(wi + wo):normalize(wi + wo * eta);

        wm = wm.z<0.0f?-wm:wm;
        BRDFBasics::TintColors(mix(mat.basecolor, mat.sssColor, mat.subsurface), eta, mat.specularTint, mat.sheenTint, F0, Csheen, Cspec0);
        //material layer mix weight
        float dielectricWt = (1.0f - mat.metallic) * (1.0f - mat.specTrans);
        float dispecular = (mat.specular==0&&mat.metallic==0)?0.0f:1.0f;
        float metalWt = (1.0f - mat.specTrans * (1.0f - mat.metallic))*dispecular;
        float glassWt = (1.0f - mat.metallic) * mat.specTrans;
        float ccWt = 0.25 * mat.clearcoat;

        float invTotalWt = 1.0f / ( dielectricWt + metalWt + glassWt + ccWt );
        float diffPr = dielectricWt * invTotalWt;
        float metalPr = metalWt * invTotalWt;
        float glassPr = glassWt * invTotalWt;
        float ccPr= ccWt * invTotalWt;


        float p0 = diffPr;
        float p3 = p0 + metalPr;
        float p4 = p3 + glassPr;
        float p5 = p4 + ccPr;




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
        float sssp=1.0f;
        if(reflect){
            wm = normalize(wi + wo);
          if(diffPr > 0.0f){

            //vec3 d = EvaluateDiffuse(thin? mat.basecolor : mix(mat.basecolor,mat.sssColor,mat.subsurface), mat.subsurface, mat.roughness, mat.sheen,Csheen, wo, wi, wm, tmpPdf) * dielectricWt;
            vec3 d = BRDFBasics::EvalDisneyDiffuse(mat.basecolor, mat.subsurface, mat.roughness, mat.sheen,Csheen, wo, wi, wm, tmpPdf);
            d = d * dielectricWt;
            dterm = dterm + d;
            f = f + d;
            fPdf += tmpPdf;
          }
          if(metalPr>0.0f){
            vec3 ks = vec3(1.0f);
            float r0 = (mat.ior - 1.0f) / (mat.ior + 1.0f);
            r0 = r0 * r0;
            float r0_eta = mix(BRDFBasics::SchlickWeight(HoV), 1.0f, r0);
            vec3 C0 = mat.specular * r0_eta * (1.0f - mat.metallic)*ks + mat.metallic * mat.basecolor;
            vec3 F = mix(C0, vec3(1.0), BRDFBasics::SchlickWeight(HoV));
            float ax, ay;
            BRDFBasics::CalculateAnisotropicParams(mat.roughness,mat.anisotropic,ax,ay);
            vec3 s = BRDFBasics::EvalMicrofacetReflection(ax, ay, wo, wi, wm, F, tmpPdf) * metalWt;
            tmpPdf *= (mat.roughness<=0.01 && reflectance==false)? 0.0f:1.0f;
            s = s * (tmpPdf>0.0f? 1.0f:0.0f);
            sterm = sterm + s;
            f = f + s;
            fPdf += tmpPdf * metalWt;
          }

          if(ccPr>0.0f){
            vec3 wm = normalize(wi + wo);
            float ax, ay;
            BRDFBasics::CalculateAnisotropicParams(mat.clearcoatRoughness,0,ax,ay);
            //ior related clearCt
            float F = BRDFBasics::SchlickDielectic(abs(dot(wm, wo)), mat.clearcoatIOR);
//            vec3 s = mix(vec3(0.04f), vec3(1.0f), F) *
//                     BRDFBasics::EvalClearcoat(mat.clearcoatRoughness, wo, wi,
//                                               wm, tmpPdf) * clearCtPr;
            vec3 s = BRDFBasics::EvalMicrofacetReflection(ax, ay, wo, wi, wm,
                                          mix(vec3(0.04), vec3(1.0f), F),
                                          tmpPdf) * ccWt;
            tmpPdf *= F;
            tmpPdf *= (mat.clearcoatRoughness<=0.01 && reflectance==false)? 0.0f:1.0f;
            sterm = sterm + s;
            f =  f + s;
            fPdf += tmpPdf * ccWt;
          }

        }

        if(glassPr>0.0f)
        {
            bool entering = wo.z>0?true:false;

            float ax, ay;
            BRDFBasics::CalculateAnisotropicParams(mat.roughness,mat.anisotropic,ax,ay);
            if (reflect) {

              vec3 wm = normalize(wi + wo);
              float F = BRDFBasics::DielectricFresnel(abs(dot(wm, wo)), entering?mat.ior:1.0f/mat.ior);
              vec3 s = BRDFBasics::EvalMicrofacetReflection(ax, ay, wo, wi, wm,
                                                            mix(mix(Cspec0, mat.diffractColor, mat.diffraction), vec3(1.0f), F) * mat.specular,
                                            tmpPdf) * glassWt;
              tmpPdf *= (mat.roughness<=0.01 && reflectance==false)? 0.0f:F;
              s = s * (tmpPdf>0.0f? 1.0f:0.0f);
              sterm = sterm + s;
              f = f + s;
              fPdf += tmpPdf * glassWt;
            } else {
              if(thin)
              {
                vec3 t = sqrt(mix(mat.transColor, mat.diffractColor, mat.diffraction)) * glassWt / (1e-6+abs(wi.z));
                //float F = BRDFBasics::DielectricFresnel(abs(wo.z), mat.ior);
                float tmpPdf = reflectance ? 1.0 : 0.0f;
                t = t * (tmpPdf>0.0f?1.0f:0.0f);
                tterm = tterm + t;
                f = f + t;
                fPdf += tmpPdf * glassWt;
              }else {
                vec3 wm = entering?-normalize(wo + mat.ior * wi) : normalize(wo + 1.0f/mat.ior * wi);
                float F = BRDFBasics::DielectricFresnel(abs(dot(wm, wo)), entering?mat.ior:1.0f/mat.ior);
                float tmpPdf;
                vec3 brdf = BRDFBasics::EvalMicrofacetRefraction(mix(sqrt(mat.transColor), mat.diffractColor, mat.diffraction),
                                                                 ax, ay,
                                                                 entering? mat.ior:1.0f/mat.ior,
                                                                 wo, wi, wm,
                                                                 vec3(F), tmpPdf);

                vec3 t = brdf * glassWt;
                tmpPdf *= (mat.roughness<=0.01 && reflectance==false)? 0.0f:(1.0f - F);
                t = t * (tmpPdf>0.0f? 1.0f:0.0f);
                tterm = tterm + t;
                f = f + t;

                fPdf += tmpPdf * glassWt;

              }
            }

        }
        if(mat.subsurface > 0.0f && (reflectance || dot(wo_world,N2) < 0.0f || thin)){
          bool trans = (wi.z * wo.z) < 0;
          float FL = BRDFBasics::SchlickWeight(abs(wi.z));
          float FV = BRDFBasics::SchlickWeight(abs(wo.z));
          float term = wo.z>0?FV:FL;
          float tmpPdf = trans? abs(wi.z) : 0.0f;//0.5/M_PIf:0.0f;
          vec3 transmit = vec3(1.0f);
          if(thin) {
//            vec3 color = mat.sssColor;
//            vec3 sigma_t, alpha;
//            CalculateExtinction2(color, mat.sssParam, sigma_t, alpha, 1.4f, mat.sssFxiedRadius);
//            vec3 channelPDF = vec3(1.0f/3.0f);
//            transmit = Transmission2(sigma_t * alpha, sigma_t,
//                                  channelPDF, 0.001f / (abs(wi.z) + 0.005f), true);
              transmit = clamp(mat.sssParam * mat.sssColor, vec3(0), vec3(1)) * abs(wi.z);
          }

          vec3 d = (trans? vec3(1.0f): vec3(0.0f)) * transmit  * sssp * dielectricWt;
          dterm = dterm + d;
          f = f + d;
          fPdf += tmpPdf  *  sssp * dielectricWt;

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
    void SampleNormal(vec3 wo, vec3& wm, float rough, float aniso, float r1, float r2)
    {
        float ax, ay;
        BRDFBasics::CalculateAnisotropicParams(rough,aniso,ax,ay);
        vec3 vtmp = wo;
        vtmp.z = abs(vtmp.z);
        wm = BRDFBasics::SampleGGXVNDF(vtmp, ax, ay, r1, r2);
    }

    static __inline__ __device__
    void SampleSpecular(vec3 wo, vec3& wi, float rough, float aniso, float r1, float r2){
      float ax, ay;
      BRDFBasics::CalculateAnisotropicParams(rough,aniso,ax,ay);
      vec3 vtmp = wo;
      vtmp.z = abs(vtmp.z);
      vec3 wm = BRDFBasics::SampleGGXVNDF(vtmp, ax, ay, r1, r2);

      wi = normalize(reflect(-wo, rough<0.02? vec3(0,0,1):wm));
    }


    static __inline__ __device__
    bool SampleDisney2(
        unsigned int& seed,
        unsigned int& eventseed,
        const MatOutput& mat,
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
        uint8_t& medium,
        vec3& extinction,
        bool& isDiff,
        bool& isSS,
        bool& isTrans,
        float& minSpecRough
    )
    {
        vec3 w_eval;
        bool reflection_fromCC = false;
        auto woo = normalize(wo);
        RadiancePRD* prd = getPRD();
//        bool sameside = (dot(woo, N)*dot(woo, N2))>0.0f;
//        if(sameside == false)
//        {
//          N = N2;
//          T = cross(B,N2);
//          B = cross(N2, T);
//        }
        float eta = dot(woo, N)>0?mat.ior:1.0f/mat.ior;
        rotateTangent(T, B, N, mat.anisoRotation * 2 * 3.1415926f);
        world2local(woo, T, B, N);
        //float2 r = sobolRnd(params.subframe_index,prd->depth+2, prd->eventseed);
        //float r1 = r.x;
        //float r2 = r.y;
        float r1 = rnd(seed);
        float r2 = rnd(seed);

        vec3 Csheen, Cspec0;
        float F0;

        BRDFBasics::TintColors(mix(mat.basecolor, mat.sssColor, mat.subsurface), eta, mat.specularTint, mat.sheenTint, F0, Csheen, Cspec0);

        //material layer mix weight
        float dielectricWt = (1.0f - mat.metallic) * (1.0f - mat.specTrans);
        float metalWt = mat.metallic;
        float glassWt = (1.0f - mat.metallic) * mat.specTrans;

        float ax, ay;
        BRDFBasics::CalculateAnisotropicParams(mat.roughness,mat.anisotropic,ax,ay);
        vec3 wm = mat.roughness<0.02?vec3(0,0,1):BRDFBasics::SampleGGXVNDF(woo, ax, ay, r1, r2);
        BRDFBasics::CalculateAnisotropicParams(mat.clearcoatRoughness,mat.anisotropic,ax,ay);
        vec3 wm2 = mat.clearcoatRoughness<0.02?vec3(0,0,1):BRDFBasics::SampleGGXVNDF(woo, ax, ay, r1, r2);
        float hov1 = abs(woo.z);
        float hov2 = abs(dot(woo, wm));
        float c = powf(smoothstep(0.0f,0.2f,mat.roughness),2.0f);

        float hov = mix(hov1, hov2, c);
        float schlickWt = BRDFBasics::SchlickWeight(hov);
        float F = BRDFBasics::DielectricFresnel(hov, mat.ior);
        float F2 = BRDFBasics::DielectricFresnel(abs(dot(woo, wm2)), mat.clearcoatIOR);
        float psss = mat.subsurface;
        float sssPortion = psss / (1.0f + psss);
        //dielectricWt *= 1.0f - psub;

        //event probability
        float clearCtPr = 0.25f * mat.clearcoat * F2;
        float other = 1.0f - clearCtPr;
        float diffPr = other * dielectricWt * (1.0f - luminance(mix(Cspec0, vec3(1.0f), F) * mat.specular));
        float sssPr = other * dielectricWt * (1.0f - luminance(mix(Cspec0, vec3(1.0f), F) * mat.specular))  * psss;
        float dielectricPr = other * dielectricWt * luminance(mix(Cspec0, vec3(1.0f), F) * mat.specular);
        float metalPr = other * metalWt;
        float glassPr = other * glassWt;


        float invTotalWt = 1.0f / (diffPr + sssPr + dielectricPr + metalPr + glassPr + clearCtPr);
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

        //finally got this clear, each seed can represent a whole different permutation of the whole van der corput sequence
        //and offset is the "index" to "take" the random number from van der corput
        //as a result, for i'th ray in a pixel, its offset shall be subframe_index, and scrumble seed shall change between
        //events
        float r3 = vdcrnd(prd->offset, prd->vdcseed);

        Onb  tbn = Onb(N);
        tbn.m_tangent = T;
        tbn.m_binormal = B;
        prd->fromDiff = false;
        if(mat.isHair>0.5f){
          prd->fromDiff = true;
          wi = SampleScatterDirection(prd->seed) ;
          vec3 wo_t = normalize(vec3(0.0f,woo.y,woo.z));
          vec3 wi_t = normalize(vec3(0.0f,wi.y,wi.z));
          float Phi = acos(dot(wo_t,wi_t));
          vec3 extinction = CalculateExtinction(mat.sssParam,1.0f);
          reflectance = HairBSDF::EvaluteHair(wi.x,dot(wi_t,wi),woo.x,
                                              dot(wo_t,woo),Phi,wi.z,1.55f,
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
          if(woo.z<0 && mat.subsurface>0)//inside, scattering, go out for sure
          {
            wi = BRDFBasics::CosineSampleHemisphere(r1, r2);
            flag = transmissionEvent;
            isSS = false;
            tbn.inverse_transform(wi);
            wi = normalize(wi);
            w_eval = wi;
            w_eval = dot(w_eval, N)<0?normalize(w_eval - 2.0f * dot(w_eval, N) * N):w_eval;
            if(dot(wi,N2)<0)
                wi = normalize(wi - 2.0f * dot(wi, N2) * N2);
          }
          else{
            //switch between scattering or diffuse reflection
            float diffp = p0/p1;
            if(rnd(prd->seed)<diffp || prd->fromDiff==true)
            {
              prd->fromDiff = true;
              wi = BRDFBasics::CosineSampleHemisphere(r1, r2);
              if(woo.z<0.0f){
                wi.z = -wi.z;
              }
              isSS = false;
              tbn.inverse_transform(wi);
              wi = normalize(wi);
              w_eval = wi;
              w_eval = (dot(wo,N)*dot(wi,N)<0)?normalize(wi - 2.0f * dot(wi, N) * N):w_eval;
              if(dot(wo,N2)*dot(wi,N2)<0)
                wi = normalize(wi - 2.0f * dot(wi, N2) * N2);
            }else
            {
              //go inside
              wi = -BRDFBasics::UniformSampleHemisphere(r1, r2);
              wi.z = min(-0.01f, wi.z);
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
              w_eval = wi;
              w_eval = (dot(wi,N)>0)?normalize(wi - 2.0f * dot(wi, N) * N):w_eval;
              if(dot(wi,N2)>0)
                wi = normalize(wi - 2.0f * dot(wi, N2) * N2);

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
            SampleSpecular(woo,wi,mat.roughness,mat.anisotropic,r1,r2);
            tbn.inverse_transform(wi);
            wi = normalize(wi);
            w_eval = wi;
            w_eval = (dot(wo,N)*dot(wi,N)<0)?normalize(wi - 2.0f * dot(wi, N) * N):w_eval;
            if(dot(wo,N2)*dot(wi,N2)<0)
                wi = normalize(wi - 2.0f * dot(wi, N2) * N2);

        }else if(r3<p4)//glass
        {
          bool entering = woo.z>0?true:false;
          float ax, ay;
          BRDFBasics::CalculateAnisotropicParams(mat.roughness,mat.anisotropic,ax,ay);
          vec3 swo = woo.z>0?woo:-woo;
          vec3 wm = mat.roughness<0.01?vec3(0,0,1):BRDFBasics::SampleGGXVNDF(swo, ax, ay, r1, r2);
          wm = wm.z<0?-wm:wm;

          wm = entering?wm:-wm;

          float F = BRDFBasics::DielectricFresnel(abs(dot(wm, woo)), entering?mat.ior:1.0f/mat.ior);

          if(rnd(prd->seed)<F)//reflection
          {
            wi = normalize(reflect(-normalize(woo),wm));
          }else //refraction
          {
            if(thin)
            {
              wi = -woo;
              extinction = vec3(0.0f);
            }else {
              wi = normalize(
                  refract(woo, wm, entering ? 1.0f / mat.ior : mat.ior));
              flag = transmissionEvent;
              isTrans = true;
              extinction =
                  CalculateExtinction(mat.transTint, mat.transTintDepth);
              extinction = entering ? extinction : vec3(0.0f);
            }
          }

          tbn.inverse_transform(wi);
          wi = normalize(wi);
          w_eval = wi;
          minSpecRough = mat.roughness;
          auto isReflection =  dot(wi, N) * dot(wo, N)>0?1:0;
          prd->hit_type = (isReflection==1?SPECULAR_HIT:TRANSMIT_HIT);
          bool sameside2 = (dot(wi, N) * dot(wi, N2))>0.0f;
          if(sameside2 == false)
          {
            wi = normalize(wi - 2.0f * dot(wi, N2) * N2);
          }

        }else if(r3<p5)//cc
        {
            prd->hit_type = SPECULAR_HIT;
            SampleSpecular(woo,wi,mat.clearcoatRoughness,0.0f,r1,r2);
            tbn.inverse_transform(wi);
            wi = normalize(wi);
            w_eval = wi;
            w_eval = (dot(wo,N)*dot(wi,N)<0)?normalize(wi - 2.0f * dot(wi, N) * N):w_eval;
            if(dot(wo,N2)*dot(wi,N2)<0)
                wi = normalize(wi - 2.0f * dot(wi, N2) * N2);
            reflection_fromCC = true;
        }

        float pdf, pdf2;
        vec3 rd, rs, rt;
        reflectance = EvaluateDisney2(vec3(1.0f), mat, w_eval, wo, T, B, N, N2, thin,
                                      is_inside, pdf, pdf2, 0, rd, rs, rt, true, reflection_fromCC);
        fPdf = pdf>1e-5f?pdf:0.0f;
        reflectance = pdf>1e-5f?reflectance:vec3(0.0f);
        return true;
    }


    static __inline__ __device__
    bool SampleDisney3(
        unsigned int& seed,
        unsigned int& eventseed,
        const MatOutput& mat,
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
        uint8_t& medium,
        vec3& extinction,
        bool& isDiff,
        bool& isSS,
        bool& isTrans,
        float& minSpecRough
    )
    {
        vec3 w_eval;
        bool reflection_fromCC = false;
        auto woo = normalize(wo);
        RadiancePRD* prd = getPRD();
//        bool sameside = (dot(woo, N)*dot(woo, N2))>0.0f;
//        if(sameside == false)
//        {
//          N = N2;
//          T = cross(B,N2);
//          B = cross(N2, T);
//        }
        float eta = dot(woo, N)>0?mat.ior:1.0f/mat.ior;
        rotateTangent(T, B, N, mat.anisoRotation * 2 * 3.1415926f);
        world2local(woo, T, B, N);
        //float2 r = sobolRnd(params.subframe_index,prd->depth+2, prd->eventseed);
        //float r1 = r.x;
        //float r2 = r.y;
        float r1 = rnd(seed);
        float r2 = rnd(seed);

        vec3 Csheen, Cspec0;
        float F0;

        BRDFBasics::TintColors(mix(mat.basecolor, mat.sssColor, mat.subsurface), eta, mat.specularTint, mat.sheenTint, F0, Csheen, Cspec0);

        //material layer mix weight
        float dielectricWt = (1.0f - mat.metallic) * (1.0f - mat.specTrans);
        float dispecular = (mat.specular==0&&mat.metallic==0)?0.0f:1.0f;
        float metalWt = (1.0f - mat.specTrans * (1.0f - mat.metallic))*dispecular;
        float glassWt = (1.0f - mat.metallic) * mat.specTrans;
        float ccWt = 0.25 * mat.clearcoat;

        float invTotalWt = 1.0f / ( dielectricWt + metalWt + glassWt + ccWt );
        float diffPr = dielectricWt * invTotalWt;
        float metalPr = metalWt * invTotalWt;
        float glassPr = glassWt * invTotalWt;
        float ccPr= ccWt * invTotalWt;


        float p0 = diffPr;
        float p3 = p0 + metalPr;
        float p4 = p3 + glassPr;
        float p5 = p4 + ccPr;

        //finally got this clear, each seed can represent a whole different permutation of the whole van der corput sequence
        //and offset is the "index" to "take" the random number from van der corput
        //as a result, for i'th ray in a pixel, its offset shall be subframe_index, and scrumble seed shall change between
        //events
        float r3 = vdcrnd(prd->offset, prd->vdcseed);
        Onb  tbn = Onb(N);
        tbn.m_tangent = T;
        tbn.m_binormal = B;
        prd->fromDiff = false;
        if(mat.isHair>0.5f){
          prd->fromDiff = true;
          wi = SampleScatterDirection(prd->seed) ;
          vec3 wo_t = normalize(vec3(0.0f,woo.y,woo.z));
          vec3 wi_t = normalize(vec3(0.0f,wi.y,wi.z));
          float Phi = acos(dot(wo_t,wi_t));
          vec3 extinction = CalculateExtinction(mat.sssParam,1.0f);
          reflectance = HairBSDF::EvaluteHair(wi.x,dot(wi_t,wi),woo.x,
                                              dot(wo_t,woo),Phi,wi.z,1.55f,
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


        if(r3<p0){
          prd->hit_type = DIFFUSE_HIT;
          isDiff = true;
          if(mat.thin<0.5 && woo.z<0 && mat.subsurface>0)//inside, scattering, go out for sure
          {
            wi = BRDFBasics::CosineSampleHemisphere(r1, r2);
            flag = transmissionEvent;
            isSS = false;
            tbn.inverse_transform(wi);
            wi = normalize(wi);
            w_eval = wi;
            //w_eval = dot(w_eval, N)<0?normalize(w_eval - 2.0f * dot(w_eval, N) * N):w_eval;
            if(dot(wi,N2)<0)
                wi = normalize(wi - 2.0f * dot(wi, N2) * N2);
            w_eval = wi;
            float pdf, pdf2;
            vec3 rd, rs, rt;
            reflectance = EvaluateDisney3(vec3(1.0f), mat, w_eval, prd->sssDirBegin, T, B, N, N2, thin,
                                          is_inside, pdf, pdf2, 0, rd, rs, rt, true, reflection_fromCC);
            fPdf = pdf>1e-5f?pdf:0.0f;
            reflectance = pdf>1e-5f?reflectance:vec3(0.0f);
            return true;
          }
          else{
            //switch between scattering or diffuse reflection
            float ax, ay;
            BRDFBasics::CalculateAnisotropicParams(mat.roughness,mat.anisotropic,ax,ay);
            vec3 swo = woo.z>0?woo:-woo;
            vec3 wm = mat.roughness<0.01?vec3(0,0,1):BRDFBasics::SampleGGXVNDF(swo, ax, ay, r1, r2);

            float F = mix(BRDFBasics::SchlickWeight(abs(dot(wm, woo))), 1.0f, 0.06f);
            float sss_wt = (1.0f - mat.subsurface);
            float diffp = mix(1.0f, 0.0f, mat.subsurface);
            if(rnd(prd->seed)<diffp || prd->fromDiff==true)
            {
              prd->fromDiff = true;
              wi = BRDFBasics::CosineSampleHemisphere(r1, r2);
              if(woo.z<0.0f){
                wi.z = -wi.z;
              }
              isSS = false;
              tbn.inverse_transform(wi);
              wi = normalize(wi);
              w_eval = wi;
              //w_eval = (dot(wo,N)*dot(wi,N)<0)?normalize(wi - 2.0f * dot(wi, N) * N):w_eval;
              if(dot(wo,N2)*dot(wi,N2)<0)
                wi = normalize(wi - 2.0f * dot(wi, N2) * N2);
              w_eval = wi;
            }else
            {
              //go inside
              wi = -BRDFBasics::UniformSampleHemisphere(r1, r2);
              wi.z = min(-0.1f, wi.z);
              wi = normalize(wi);
              isSS = true;
              flag = transmissionEvent;
              vec3 color = mat.sssColor;
              color = clamp(color, vec3(0.01), vec3(0.99));
              vec3 sssRadius = mat.subsurface * mat.sssParam;
              RadiancePRD *prd = getPRD();
              prd->ss_alpha = color;
              if (isSS) {
                medium = PhaseFunctions::isotropic;
                CalculateExtinction2(color, sssRadius, prd->sigma_t, prd->ss_alpha, 1.4f, mat.sssFxiedRadius);
              }
              tbn.inverse_transform(wi);
              wi = normalize(wi);
              w_eval = wi;
              //w_eval = (dot(wi,N)>0)?normalize(wi - 2.0f * dot(wi, N) * N):w_eval;
              if(dot(wi,N2)>0)
                wi = normalize(wi - 2.0f * dot(wi, N2) * N2);
              w_eval = wi;

              fPdf = 1.0f;
              reflectance = vec3(1.0f);
              return true;
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
            SampleSpecular(woo,wi,mat.roughness,mat.anisotropic,r1,r2);
            tbn.inverse_transform(wi);
            wi = normalize(wi);
            w_eval = wi;
            w_eval = (dot(wo,N)*dot(wi,N)<0)?normalize(wi - 2.0f * dot(wi, N) * N):w_eval;
            if(dot(wo,N2)*dot(wi,N2)<0)
                wi = normalize(wi - 2.0f * dot(wi, N2) * N2);

        }else if(r3<p4)//glass
        {
          bool entering = woo.z>0?true:false;
          float ax, ay;
          BRDFBasics::CalculateAnisotropicParams(mat.roughness,mat.anisotropic,ax,ay);
          vec3 swo = woo.z>0?woo:-woo;
          vec3 wm = mat.roughness<0.01?vec3(0,0,1):BRDFBasics::SampleGGXVNDF(swo, ax, ay, r1, r2);
          wm = wm.z<0?-wm:wm;

          wm = entering?wm:-wm;

          float F = BRDFBasics::DielectricFresnel(abs(dot(wm, woo)), entering?mat.ior:1.0f/mat.ior);

          if(rnd(prd->seed)<F)//reflection
          {
            wi = normalize(reflect(-normalize(woo),wm));
          }else //refraction
          {
            if(thin)
            {
              wi = -woo;
              extinction = vec3(0.0f);
            }else {
              wi = normalize(
                  refract(woo, wm, entering ? 1.0f / mat.ior : mat.ior));
              flag = transmissionEvent;
              isTrans = true;
              extinction =
                  CalculateExtinction(mat.transTint, mat.transTintDepth);
              extinction = entering ? extinction : vec3(0.0f);
            }
          }

          tbn.inverse_transform(wi);
          wi = normalize(wi);
          w_eval = wi;
          minSpecRough = mat.roughness;
          auto isReflection =  dot(wi, N) * dot(wo, N)>0?1:0;
          prd->hit_type = (isReflection==1?SPECULAR_HIT:TRANSMIT_HIT);
          bool sameside2 = (dot(wi, N) * dot(wi, N2))>0.0f;
          if(sameside2 == false)
          {
            wi = normalize(wi - 2.0f * dot(wi, N2) * N2);
          }

        }else if(r3<p5)//cc
        {
            prd->hit_type = SPECULAR_HIT;
            SampleSpecular(woo,wi,mat.clearcoatRoughness,0.0f,r1,r2);
            tbn.inverse_transform(wi);
            wi = normalize(wi);
            w_eval = wi;
            w_eval = (dot(wo,N)*dot(wi,N)<0)?normalize(wi - 2.0f * dot(wi, N) * N):w_eval;
            if(dot(wo,N2)*dot(wi,N2)<0)
                wi = normalize(wi - 2.0f * dot(wi, N2) * N2);
            reflection_fromCC = true;
        }

        float pdf, pdf2;
        vec3 rd, rs, rt;
        reflectance = EvaluateDisney3(vec3(1.0f), mat, w_eval, wo, T, B, N, N2, thin,
                                      is_inside, pdf, pdf2, 0, rd, rs, rt, true, reflection_fromCC);
        fPdf = pdf>1e-5f?pdf:0.0f;
        reflectance = pdf>1e-5f?reflectance:vec3(0.0f);
        return true;
    }




    //the bsdf part of surface behavior
    struct PrincipledBSDF{
      bool enable_diffuse   = false;
      vec3 diffuse_weight = vec3(0.f);

      bool enable_subsurface   = false;
      vec3 subsurface_weight = vec3(0.f);
      vec3 subsurface_albedo = vec3(0.f);
      vec3 subsurface_radius = vec3(0.f);

      // specular
      bool enable_specular   = false;
      vec3 specular_weight = vec3(0.f);
      float alpha_x          = 1.f;
      float alpha_y          = 1.f;
      float ior              = 1.5f;
      vec3 specular_color  = vec3(0.f);

      // clearcoat
      bool enable_clearcoat   = false;
      vec3 clearcoat_weight = vec3(0.f);
      float clearcoat_alpha_x = 1.f;
      float clearcoat_alpha_y = 1.f;
      float clearcoat_ior     = 1.5f;
      vec3 clearcoat_color  = vec3(0.0f);
    };
    struct BSDFSampleWeight {
        float diffuse_sample_weight;
        float subsurface_sample_weight;
        float specular_sample_weight;
        float clearcoat_sample_weight;
    };

    static __inline__ __device__  vec3 SpecularColor(const vec3 &wi, const vec3 &wo, const vec3 &specular_color, const float ior)
    {
        vec3 h = normalize(wo + wi);
        float f0 = BRDFBasics::DielectricFresnel(1.0, ior);
        float fh = (BRDFBasics::DielectricFresnel(dot(h, wo), ior) - f0) / (1.0f - f0);
        return mix(specular_color, vec3(1.0f), fh);
    }
    static __inline__ __device__ void FetchClosureSampleWeight(const vec3&wo, const PrincipledBSDF& bsdf, BSDFSampleWeight& w, MatOutput &mat)
    {
        w.diffuse_sample_weight = bsdf.enable_diffuse?RgbToY(bsdf.diffuse_weight):0;

        w.subsurface_sample_weight = bsdf.enable_subsurface?RgbToY(bsdf.subsurface_weight):0;

        w.specular_sample_weight = bsdf.enable_specular?RgbToY(bsdf.specular_weight * SpecularColor(wo*vec3(-1,-1,1), wo, bsdf.specular_color, bsdf.ior)):0;

        w.clearcoat_sample_weight = bsdf.enable_clearcoat?RgbToY(bsdf.clearcoat_weight * SpecularColor(wo*vec3(-1,-1,1), wo, bsdf.clearcoat_color, bsdf.clearcoat_ior)):0;

        float sum = 0.0f;
        sum += w.diffuse_sample_weight;
        sum += w.subsurface_sample_weight;
        sum += w.specular_sample_weight;
        sum += w.clearcoat_sample_weight;

        w.diffuse_sample_weight/=sum;
        w.subsurface_sample_weight/=sum;
        w.specular_sample_weight/=sum;
        w.clearcoat_sample_weight/=sum;
    }


    static __inline__ __device__ float sample_scatter_distance(const vec3 throughput,const vec3 sigma_s, const vec3 sigma_t,
                                                               unsigned int & seed, vec3 & channel_pdf)
    {
        vec3 albedo = safe_divide_spectrum(sigma_s, sigma_t);
        int channel = volume_sample_channel(albedo*throughput, rnd(seed), channel_pdf);
        const float sample_sigma_t = sigma_t[channel];
        float distance = -log(max(1.0f-rnd(seed), _FLT_MIN_)) / sample_sigma_t;
        return distance;
    }
//    static __inline__ __device__ bool randomwalk_subsurface(const vec3 &weight,
//                                                            const vec3 &albedo,
//                                                            const vec3 &radius,
//                                                            const vec3 &indir,
//                                                            const vec3 &inpos,
//                                                            unsigned int &seed,
//                                                            vec3 &wo, vec3 &throughput_out,
//                                                            int & scatter_bounces)
//    {
//        scatter_bounces = 0;
//        vec3 dir = indir;
//
//        vec3 sigma_t;
//        vec3 sigma_s;
//        vec3 throughput = vec3(1);
//
//        compute_scattering_coeff(weight,albedo, radius,sigma_t,sigma_s,throughput);
//
//        vec3 tmpdir = indir;
//        vec3 tmppos = inpos;
//        float max_distance = 1e16;
//        bool hit = false;
//        for(int bounce=0;bounce<128;bounce++)
//        {
//            if(bounce > 0)
//            {
//                if(false)
//                {
//                    //HG sample placeholder
//                }else{
//                    tmpdir = SampleScatterDirection(seed);
//                }
//            }
//
//            vec3 channel_pdf;
//            float max_dist = sample_scatter_distance(throughput,sigma_s, sigma_t,seed,channel_pdf);
//
//            hit = false;//traceFirstHit(ray)
//            float t = hit?optixGetRayTmax():max_dist;
//
//            vec3 transmittance = Transmission(sigma_t,t);
//
//            if(hit)
//            {
//                float pdf = dot(channel_pdf, transmittance);
//                throughput = throughput * transmittance / pdf;
//                break;
//            } else
//            {
//                float pdf = dot(channel_pdf, sigma_t * transmittance);
//                throughput = throughput * (sigma_s * transmittance) / pdf;
//            }
//
//            {
//                float p = saturate(max(max(throughput.x, throughput.y),throughput.z));
//                if(rnd(seed)>=p)
//                {
//                    break;
//                }
//                throughput = throughput / p;
//            }
//
//            tmppos = tmppos + t * tmpdir;
//            scatter_bounces++;
//        }//end of randomwalk tracing
//        if(!hit)
//        {
//            return false;
//        }
//        bool hit_back_face = false;//how to detect
//        if(hit_back_face)//hit a back face
//        {
//            return false;
//        }
//        wo = vec3(1);//from object space back to world space;
//        throughput_out = throughput;
//    }
    static __inline__ __device__  void MatToBsdf(const MatOutput &mat, PrincipledBSDF& bsdf)
    {
        const vec3 weight = vec3(1.0f);
        float closure_weight_cutoff = 1e-3;

        float _diffuse_weight  = (1.0f - saturate(mat.metallic)) * (1.0f - saturate(mat.specTrans));
        float _final_transmit  = saturate(mat.specTrans) * (1.0f - saturate(mat.metallic));
        float _specular_weight = (1.0f - _final_transmit);

        {
            vec3 mixed_ss_base_color = mix(mat.basecolor, mat.sssColor, mat.subsurface);
            bsdf.enable_diffuse = false;
            if(dot(mixed_ss_base_color, vec3(1.0f))/3.0f>closure_weight_cutoff){
                if(mat.subsurface < closure_weight_cutoff &&
                  _diffuse_weight > closure_weight_cutoff)
                {
                    bsdf.enable_diffuse = true;
                    bsdf.diffuse_weight = weight * mat.basecolor * _diffuse_weight;
                } else if(mat.subsurface > closure_weight_cutoff)
                {
                    bsdf.enable_subsurface = true;
                    vec3 subsurface_weight = weight * mixed_ss_base_color * _diffuse_weight;
                    bsdf.subsurface_weight = subsurface_weight;
                    bsdf.subsurface_albedo = mixed_ss_base_color;
                    bsdf.subsurface_radius = mat.sssParam;

                    vec3 add_diffuse_weight = vec3(0);

                    bssrdf_setup(
                            true, mat.sssFxiedRadius, true,
                            bsdf.subsurface_weight, bsdf.subsurface_albedo,
                            bsdf.subsurface_radius, add_diffuse_weight);

                    if(!is_black(add_diffuse_weight))
                    {
                        bsdf.enable_diffuse = true;
                        bsdf.diffuse_weight += add_diffuse_weight;
                    }
                }
            }
        }

        bsdf.enable_specular = false;
        if (_specular_weight > closure_weight_cutoff &&
            (mat.specular > closure_weight_cutoff || mat.metallic > closure_weight_cutoff))
        {
            bsdf.enable_specular = true;
            bsdf.specular_weight = weight * _specular_weight;
            bsdf.ior = (2.0f / (1.0f - sqrtf(0.08f * saturate(mat.specular)))) - 1.0f;
            float aspect = sqrtf(1.0f - mat.anisotropic * 0.9f);
            float roughness2 = mat.roughness * mat.roughness;
            bsdf.alpha_x = roughness2 / aspect;
            bsdf.alpha_y = roughness2 * aspect;

            vec3 rho_specular = RgbToY(mat.basecolor)>0.0f?mat.basecolor/RgbToY(mat.basecolor):vec3(0);
            bsdf.specular_color = mix(0.08f * mat.specular * rho_specular, mat.basecolor, mat.metallic);
        }

        bsdf.enable_clearcoat = false;
        if(mat.clearcoat > closure_weight_cutoff)
        {
            bsdf.enable_clearcoat = true;
            bsdf.clearcoat_weight = vec3(0.25f * mat.clearcoat);
            bsdf.clearcoat_alpha_x = mat.clearcoatRoughness * mat.clearcoatRoughness;
            bsdf.clearcoat_alpha_y = bsdf.clearcoat_alpha_x;

            bsdf.clearcoat_color = vec3(0.04);
            bsdf.clearcoat_ior = mat.clearcoatIOR;
        }
    }
}
