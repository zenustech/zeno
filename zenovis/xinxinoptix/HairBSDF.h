#pragma once
#include "zxxglslvec.h"
#include "DisneyBRDF.h"
namespace HairBSDF{
    /*
    * Define as fllow:
    * Hair as a curve r(s), r is a vector from origin to point on curve, s is the length of curve.
    * We have tanget vector t = (dr/ds) / |dr/ds|
    * Normal vector n = (dt/ds) / |dt/ds|
    * Plane p is perpendicular to t
    * Project w_i,w_o to p, get w_i' and w_o'
    * Theta_i is the angle from w_i' to wi [0,pi/2]
    * Theta_o is the angle from w_o' to wo [0,pi/2]
    * Phi_i is the angle from n to w_i' [0,2pi]
    * Phi_o is the angle from n to w_o' [0,2pi]
    * Phi = Phi_i - Phi_o
    * 
    * 
    * 
    * 
    * 
    */
    static __inline__ __device__ float pow20(float x)
    {
        float xx = x*x;
        float x5 = xx * xx * x;
        float x10 = x5 * x5;
        return x10 * x10;
    }
    static __inline__ __device__ float pow22(float x)
    {
        float xx = x*x;
        float x5 = xx * xx * x;
        float x11 = x5 * x5 * x;
        return x11 * x11;
    }
    static __inline__ __device__ float poly5(float x)
    {
        float x2 = x*x;
        float x4 = x2*x2;
        return 5.969
                - 0.215 * x
                + 2.532 * x2
                - 10.73 * x2 * x
                + 5.574 * x4
                + 0.245 * x4 * x;
    }
    static __inline__ __device__ float
    sinh(float x)
    {
        return (exp(x) - exp(-x)) * 0.5;
    }
    static __inline__ __device__ float
    csch(float x)
    {
        return 1.0f/sinh(x);
    }
    static __inline__ __device__ float
    I_0_v2(float x)
    {
      x = pbrt::Sqr(x);
      float val = 1.0f + 0.25f * x;
      float pow_x_2i = pbrt::Sqr(x);
      uint64_t i_fac_2 = 1;
      int pow_4_i = 16;
      for (int i = 2; i < 10; i++) {
        i_fac_2 *= i * i;
        const float newval = val + pow_x_2i / (pow_4_i * i_fac_2);
        if (val == newval) {
          return val;
        }
        val = newval;
        pow_x_2i *= x;
        pow_4_i *= 4;
      }
      return val;
    }
    static __inline__ __device__ float
    I_0(float x)
    {
        float sum = 1.0f;
        float P = 1.0f;
        for(int i=1;i<11;i++){
            P *= x * x / 4 / i / i;
            sum +=P;
        }
        return sum;
    }
    static __inline__ __device__ float
    logI_0(float x){
        return x>12.0f?
        (x + 0.5 * (-log(2 * M_PIf) + log(1 / x) + 1 / (8 * x)))
        :log(I_0_v2(x));
    }

    /*From weta, can be repalce */
    static __inline__ __device__ float 
    M_p_Weta(float sinTheta_i, float cosTheta_i, float sinTheta_o, float cosTheta_o, float beta)
    {
        float v = beta;
        v = max(v,0.0001f);
        float a = cosTheta_i * cosTheta_o / v;
        float b = sinTheta_i * sinTheta_o / v;
        if(v < 0.1f){
            return exp(logI_0(a) - b - 1 / v + 0.6931f + log(1 / (2 * v)));
        }
        float term1 = sinh(1/v) * (2*v);
        float term2 = exp(-b);
        float term3 = I_0_v2(a);
        return term2 * term3 / term1;
    }

    static __inline__ __device__ float
    Logistic(float x, float s) {
        x = abs(x);
        return exp(-x / s) / (s * pbrt::Sqr(1 + exp(-x / s)));
    }
    static __inline__ __device__ float 
    LogisticCDF(float x,float s)
    {
        float temp = exp(-x/s);
        if(temp>88.0f)
            return 0;
        return 1 / (1+temp);
    }
    static __inline__ __device__ float 
    TrimmedLogistic(float x, float s, float a, float b){
        float scaling_fac = 1.0f - 2.0f * LogisticCDF(-M_PIf, s);
        return Logistic(x, s) / scaling_fac;
    }
    static __inline__ __device__ float
    GetPhi(float p,float gammaO,float gammaT)
    {
        return 2 * p * gammaT - 2 * gammaO + p * M_PIf;
    }
    static __inline__ __device__ float
    my_fmod(float a, float b)
    {
        return a - floor(a/b)*b;
    }
    static __inline__ __device__ float wrap_angle(const float a)
    {
        auto M_2PIf = 2* M_PIf;
        return (a + M_PIf) - M_2PIf * floorf((a + M_PIf) / M_2PIf) - M_PIf;
    }
    static __inline__ __device__ float
    N_p(float Phi, int p, float s, float gammaO, float gammaT){

        //return 1;
        float dphi = Phi - GetPhi(p, gammaO, gammaT);
//        dphi = my_fmod(dphi, 2.0f*M_PIf);
//        dphi -= dphi>M_PIf? 2.0f*M_PIf : 0.0f;
    // Remap _dphi_ to $[-\pi,\pi]$
//        while (dphi > M_PIf) dphi -= 2 * M_PIf;
//        while (dphi < -M_PIf) dphi += 2 * M_PIf;
        return TrimmedLogistic(wrap_angle(dphi), s, -M_PIf, M_PIf);

    }
    static __inline__ __device__ void
    Ap(float cosTheta_o, float ior, float h, int p, vec3 color, vec3* ap)
    {
        float cosTheta = abs(cosTheta_o) * safesqrt(1-h*h);
        float f = BRDFBasics::DielectricFresnel(cosTheta,ior);
        float f2 = BRDFBasics::DielectricFresnel(abs(cosTheta_o),ior);
        vec3 tmp = pbrt::Sqr(1.0 - f) * color;
        ap[0] = vec3(f);
        ap[1] = tmp;
        ap[2] = ap[1] * color * f;
        ap[3] = ap[2] * color * f / (vec3(1.0f) - color * f);
//        if(p==0){
//            return vec3(f);
//        }else if(p==1){
//            return tmp;
//        }else if(p==2){
//            return tmp * color * f;
//        }else if(p==3)
//        {
//            return tmp * color * f * color * f / (vec3(1.0f) - color * f);
//        }
    }
    static __inline__ __device__ vec3
    Ap(float cosTheta_o, float ior, float h, int p, vec3 color)
    {
        float cosTheta = abs(cosTheta_o) * safesqrt(1-h*h);
        float f = BRDFBasics::DielectricFresnel(cosTheta,ior);
        vec3 tmp = pow(1.0f-f,2.0f) * color;
        if(p==0){
            return vec3(f);
        }else if(p==1){
            return tmp;
        }else if(p==2){
            return tmp * color * f;
        }else if(p==3)
        {
            return tmp * color * f * color * f / (vec3(1.0f) - color * f);
        }
    }
    static __inline__ __device__ float
    SampleTrimmedLogistic(float u, float s, float a, float b) {
        float k = LogisticCDF(b, s) - LogisticCDF(a, s);
        float x = -s * log(1 / (u * k + LogisticCDF(a, s)) - 1);
        return clamp(x, a, b);
    }
    static __inline__ __device__ vec3
    EvaluteHair(float sinTheta_i, float cosTheta_i,
                float sinTheta_o, float cosTheta_o,
                float Phi, float h,
                float ior, vec3 basecolor,
                float beta_m, float beta_n, float alpha)
    {

        return vec3(0);
    }

    static __inline__ __device__ vec3
    EvaluteHair2(vec3 wo,vec3 wi,
                 float h,float ior,
                 vec3 basecolor,
                float beta_m, float beta_n, float alpha, float &pdf)
    {
        float sinTheta_o = wo.x;
        float cosTheta_o = safesqrt(1 - pbrt::Sqr(sinTheta_o));
        float phiO = atan2(wo.z, wo.y);

        float sinTheta_i = wi.x;
        float cosTheta_i = safesqrt(1 - pbrt::Sqr(sinTheta_i));
        float phiI = atan2(wi.z, wi.y);
        float Phi = phiI - phiO;



        float s = 0.626657069f * (0.265f*beta_n + 1.194f*beta_n*beta_n + 5.372f*pow22(beta_n));

        float sinGammaT = h / safesqrt(ior*ior - (sinTheta_o * sinTheta_o)) *  cosTheta_o;
        float cosGammaT = safesqrt(1-sinGammaT * sinGammaT);
        float sinTheta_t = sinTheta_o/ior;
        float cosTheta_t = safesqrt(1-sinTheta_t * sinTheta_t);
        float gammaT = asin(sinGammaT);
        float gammaO = asin(h);

        vec3 sigma = pbrt::Sqr(log(pow(basecolor,1.0)) / poly5(beta_n));
        //sigma = sigma * sigma;
        vec3 T = exp(-sigma * (2 * cosGammaT / cosTheta_t));//pow(basecolor,  (2*cosGammaT/cosTheta_t));



        float v[4];
        v[0]= pbrt::Sqr(0.726f * beta_m + 0.812f * pbrt::Sqr(beta_m) + 3.7f * pow20(beta_m));
        v[1] = 0.25*v[0];
        v[2] = 4 * v[0];
        v[3] = v[2];

        float sin2kAlpha[3];
        float cos2kAlpha[3];
        sin2kAlpha[0] = sin(alpha/180.0f*M_PIf);
        cos2kAlpha[0] = safesqrt(1 - pbrt::Sqr(sin2kAlpha[0]));
        for (int i = 1; i < 3; ++i) {
            sin2kAlpha[i] = 2 * cos2kAlpha[i - 1] * sin2kAlpha[i - 1];
            cos2kAlpha[i] = pbrt::Sqr(cos2kAlpha[i - 1]) - pbrt::Sqr(sin2kAlpha[i - 1]);
        }
        vec3 ap[4];
        float apPdf[4];
        Ap(cosTheta_o,ior,h,0,T,ap);
        float sumY = RgbToY(ap[0]) + RgbToY(ap[1])  + RgbToY(ap[2])  + RgbToY(ap[3]) ;
        apPdf[0] = RgbToY(ap[0])/sumY; apPdf[1] = RgbToY(ap[1])/sumY; apPdf[2] = RgbToY(ap[2])/sumY; apPdf[3] = RgbToY(ap[3])/sumY;
        pdf = 0;
        vec3 sum = vec3(0);
        for(int i=0;i<3;i++)
        {
            float sinThetaOp, cosThetaOp;
            if (i == 0) {
                sinThetaOp = sinTheta_o * cos2kAlpha[1] - cosTheta_o * sin2kAlpha[1];
                cosThetaOp = cosTheta_o * cos2kAlpha[1] + sinTheta_o * sin2kAlpha[1];
            }
            // Handle remainder of $p$ values for hair scale tilt
            else if (i == 1) {
                sinThetaOp = sinTheta_o * cos2kAlpha[0] + cosTheta_o * sin2kAlpha[0];
                cosThetaOp = cosTheta_o * cos2kAlpha[0] - sinTheta_o * sin2kAlpha[0];
            } else if (i == 2) {
                sinThetaOp = sinTheta_o * cos2kAlpha[2] + cosTheta_o * sin2kAlpha[2];
                cosThetaOp = cosTheta_o * cos2kAlpha[2] - sinTheta_o * sin2kAlpha[2];
            } else {
                sinThetaOp = sinTheta_o;
                cosThetaOp = cosTheta_o;
            }
            cosThetaOp = abs(cosThetaOp);
            float Mp = M_p_Weta(sinTheta_i,cosTheta_i,sinThetaOp,cosThetaOp, v[i]);
            float Np = N_p(Phi,i,s,gammaO,gammaT);
            sum += Mp * ap[i] * Np;
            pdf += Mp * apPdf[i] * Np;
        }
        sum += M_p_Weta(sinTheta_i,cosTheta_i,sinTheta_o,cosTheta_o,v[3] ) * ap[3] / (2 * M_PIf);
        pdf += M_p_Weta(sinTheta_i,cosTheta_i,sinTheta_o,cosTheta_o,v[3] ) * apPdf[3] / (2 * M_PIf);
        return sum;
    }
    static __inline__ __device__ vec3
    SampleHair2(vec3 wo,vec3 &wi, unsigned int &seed,
                 float h,float ior,
                 vec3 basecolor,
                float beta_m, float beta_n, float alpha, float &pdf)
    {
        float sinTheta_o = wo.x;
        float cosTheta_o = safesqrt(1 - pbrt::Sqr(sinTheta_o));
        float phiO = atan2(wo.z, wo.y);


        float s = 0.626657069f * (0.265f*beta_n + 1.194f*pbrt::Sqr(beta_n) + 5.372f*pow22(beta_n));

        vec3 rand;
        rand.x = rnd(seed); rand.y = rnd(seed); rand.z = rnd(seed);

        float sinGammaT = h / safesqrt(ior*ior - (sinTheta_o * sinTheta_o)) *  cosTheta_o;
        float cosGammaT = safesqrt(1-sinGammaT * sinGammaT);
        float sinTheta_t = sinTheta_o/ior;
        float cosTheta_t = safesqrt(1-sinTheta_t * sinTheta_t);


        vec3 sigma = pbrt::Sqr(log(pow(basecolor,1.0)) / poly5(beta_n));
        //sigma = sigma * sigma;
        vec3 T = exp(-sigma * (2 * cosGammaT / cosTheta_t));
        vec3 ap[4];
        float apPdf[4];
        Ap(cosTheta_o,ior,h,0,T,ap);
        float sumY = RgbToY(ap[0]) + RgbToY(ap[1])  + RgbToY(ap[2])  + RgbToY(ap[3]) ;
        apPdf[0] = RgbToY(ap[0])/sumY; apPdf[1] = RgbToY(ap[1])/sumY; apPdf[2] = RgbToY(ap[2])/sumY; apPdf[3] = RgbToY(ap[3])/sumY;

        int p;
        for (p = 0; p < 3; ++p) {
            if (rand.z < apPdf[p]) break;
            rand.z -= apPdf[p];
        }
        rand.z /= apPdf[p];


        float v[4];
        v[0]= pbrt::Sqr(0.726f * beta_m + 0.812f * pbrt::Sqr(beta_m) + 3.7f * pow20(beta_m));
        v[1] = 0.25*v[0];
        v[2] = 4 * v[0];
        v[3] = v[2];
        rand.z = max(rand.z, 1e-5f);
        float cosTheta = 1 + v[p] * log(rand.z + (1 - rand.z) * exp(-2 / v[p]));
        float sinTheta = safesqrt(1 - pbrt::Sqr(cosTheta));
        float cosPhi = cos(2 * M_PIf * rand.y);
        float sinTheta_i = -cosTheta * sinTheta_o + sinTheta * cosPhi * cosTheta_o;
        float cosTheta_i = safesqrt(1 - pbrt::Sqr(sinTheta_i));
        float gammaT = asin(sinGammaT);
        float gammaO = asin(h);
        float dphi;
        if (p < 3)
            dphi =
                GetPhi(p, gammaO, gammaT) + SampleTrimmedLogistic(rand.x, s, -M_PIf, M_PIf);
        else
            dphi = 2 * M_PIf * rand.x;

        float phiI = phiO + dphi;
        wi = vec3(sinTheta_i, cosTheta_i * cos(phiI),
                   cosTheta_i * sin(phiI));

        float sin2kAlpha[3];
        float cos2kAlpha[3];
        sin2kAlpha[0] = sin(alpha/180.0f*M_PIf);
        cos2kAlpha[0] = safesqrt(1 - pbrt::Sqr(sin2kAlpha[0]));
        for (int i = 1; i < 3; ++i) {
            sin2kAlpha[i] = 2 * cos2kAlpha[i - 1] * sin2kAlpha[i - 1];
            cos2kAlpha[i] = pbrt::Sqr(cos2kAlpha[i - 1]) - pbrt::Sqr(sin2kAlpha[i - 1]);
        }
        float Phi = phiI - phiO;
        pdf = 0;
        vec3 sum = vec3(0);
        for(int i=0;i<3;i++)
        {
            float sinThetaOp, cosThetaOp;
            if (i == 0) {
                sinThetaOp = sinTheta_o * cos2kAlpha[1] - cosTheta_o * sin2kAlpha[1];
                cosThetaOp = cosTheta_o * cos2kAlpha[1] + sinTheta_o * sin2kAlpha[1];
            }
            // Handle remainder of $p$ values for hair scale tilt
            else if (i == 1) {
                sinThetaOp = sinTheta_o * cos2kAlpha[0] + cosTheta_o * sin2kAlpha[0];
                cosThetaOp = cosTheta_o * cos2kAlpha[0] - sinTheta_o * sin2kAlpha[0];
            } else if (i == 2) {
                sinThetaOp = sinTheta_o * cos2kAlpha[2] + cosTheta_o * sin2kAlpha[2];
                cosThetaOp = cosTheta_o * cos2kAlpha[2] - sinTheta_o * sin2kAlpha[2];
            } else {
                sinThetaOp = sinTheta_o;
                cosThetaOp = cosTheta_o;
            }
            cosThetaOp = abs(cosThetaOp);
            float Mp = M_p_Weta(sinTheta_i,cosTheta_i,sinThetaOp,cosThetaOp, v[i]);
            float Np = N_p(Phi,i,s,gammaO,gammaT);
            sum += Mp * ap[i] * Np;
            pdf += Mp * apPdf[i] * Np;
        }
        sum += M_p_Weta(sinTheta_i,cosTheta_i,sinTheta_o,cosTheta_o,v[3] ) * ap[3] / (2 * M_PIf);
        pdf += M_p_Weta(sinTheta_i,cosTheta_i,sinTheta_o,cosTheta_o,v[3] ) * apPdf[3] / (2 * M_PIf);
        return sum;
    }
}