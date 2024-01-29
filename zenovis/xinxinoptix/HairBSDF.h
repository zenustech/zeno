#pragma once
#include "zxxglslvec.h"
#include "TraceStuff.h"
#include "IOMat.h" 
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

    static __inline__ __device__ float
    sinh(float x)
    {
        float a = exp(2*x) - 1;
        float b = 2 * exp(x);
        return a / b;
    }
    static __inline__ __device__ float
    csch(float x)
    {
        float a = exp(2*x) - 1;
        float b = 2 * exp(x);
        return b / a;
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
        return x + 0.5 * (-log(2 * M_PIf) + log(1 / x) + 1 / (8 * x));
    }

    /*From weta, can be repalce */
    static __inline__ __device__ float 
    M_p_Weta(float sinTheta_i, float cosTheta_i, float sinTheta_o, float cosTheta_o, float beta)
    {
        float v = beta * beta;
        v = max(v,0.04);
        float a = cosTheta_i * cosTheta_o / v;
        float b = sinTheta_i * sinTheta_o / v;
        //if(beta < 0.3f){
        //    return exp(logI_0(a) - b - 1 / v + 0.6931f + log(1 / (2 * v)));
        //}
        float term1 = csch(1/v) / (2*v);
        float term2 = exp(-b);
        float term3 = I_0(a);
        return term1 * term2 * term3;

    }
    static __inline__ __device__ float
    M_p_UE(float sinTheta_i, float sinTheta_o, float beta, float alpha)
    {
        float term1 = 1 / beta / 2.5066283f; // sqrt(2*pi)
        float term2 = sinTheta_i + sinTheta_o - alpha ;
        float term3 = exp(-term2 / 2 / beta/beta);
        return term1 * term3;
    }
    static __inline__ __device__ vec3 
    N_r(float Phi, float h,float ior,vec3 basecolor)
    {
        float term1 = 0.25f * sqrtf((1.0f+cosf(Phi))/2.0f);
        vec3 term2 = mix(basecolor,vec3(1.0f),BRDFBasics::SchlickDielectic(h,ior));
        return vec3(term1 * term2);
    }
    static __inline__ __device__ float
    Logistic(float x, float s) {
        x = abs(x);
        return exp(-x / s) / (s * (1 + exp(-x / s) * (1 + exp(-x / s))));
    }
    static __inline__ __device__ float 
    LogisticCDF(float x,float s)
    {
        float temp = exp(-x/s);
        return 1 / (1+temp);
    }
    static __inline__ __device__ float 
    TrimmedLogistic(float x, float s, float a, float b){
        return Logistic(x, s) / (LogisticCDF(b, s) - LogisticCDF(a, s));
    }
    static __inline__ __device__ float
    GetPhi(int p,float gammaO,float gammaT)
    {
        return 2 * p * gammaT - 2 * gammaO + p * M_PIf;
    }
    static __inline__ __device__ float
    N_p(float Phi, int p, float s, float gammaO, float gammaT){
        float dphi = Phi - GetPhi(p, gammaO, gammaT);
    // Remap _dphi_ to $[-\pi,\pi]$
        while (dphi > M_PIf) dphi -= 2 * M_PIf;
        while (dphi < M_PIf) dphi += 2 * M_PIf;
        return TrimmedLogistic(dphi, s, -M_PIf, M_PIf);

    }
    static __inline__ __device__ vec3
    Ap(float cosTheta_o, float ior, float h, int p, vec3 color)
    {
        float cosTheta = abs(cosTheta_o) * sqrtf(1-h*h);
        float f = BRDFBasics::SchlickDielectic(cosTheta,ior);
        vec3 tmp = sqrtf(1-f) * color;
        if(p==0){
            return vec3(f);
        }else if(p==1){
            return tmp;
        }else if(p==2){
            return tmp * color * f;
        }
    }
    static __inline__ __device__ vec3 
    EvaluteHair(float sinTheta_i, float cosTheta_i,
                float sinTheta_o, float cosTheta_o, 
                float Phi, 
                float h, float ior, vec3 sigma_a,vec3 basecolor,
                float beta_m, float beta_n, float alpha)
    {
        float pow2BetaN = beta_n * beta_n;
        float pow4BetaN = pow2BetaN * pow2BetaN;
        float pow8BetaN = pow4BetaN * pow4BetaN;
        float pow16BetaN = pow8BetaN * pow8BetaN;
        float pow22BetaN = pow2BetaN * pow4BetaN * pow16BetaN;
        float s = 0.6255570569f * (0.265f*beta_n + 1.194f*sqrtf(beta_n) + 5.372f*pow22BetaN);
        float sinGammaT = h / sqrtf(ior*ior - (sinTheta_o * sinTheta_o)) *  cosTheta_o;
        float cosGammaT = sqrt(1-sinGammaT * sinGammaT);
        float sinTheta_t = sinTheta_o/ior;
        float cosTheta_t = sqrtf(1-sinTheta_t * sinTheta_t);
        float gammaT = acos(cosGammaT);
        float gammaO = asin(h);
        vec3 R = M_p_Weta(sinTheta_i,cosTheta_i,sinTheta_o,cosTheta_o,beta_m) * N_r(Phi,h,ior,basecolor) ;
        vec3 T = exp(-sigma_a * (2*cosGammaT/cosTheta_t));
        vec3 MP =  vec3(M_p_Weta(sinTheta_i,cosTheta_i,sinTheta_o,cosTheta_o,beta_m*2));
        vec3 NP = vec3(N_p(Phi,1,s,gammaO,gammaT));
        vec3 AP = Ap(cosTheta_o,ior,h,1,T);
        //vec3 TT =  M_p_Weta(sinTheta_i,cosTheta_i,sinTheta_o,cosTheta_o,beta_m*2)*N_p(Phi,1,s,gammaO,gammaT) * Ap(cosTheta_o,ior,h,1,T);
//M_p_UE(sinTheta_i,sinTheta_o,beta_m/2,sin(alpha)) *N_p(Phi,1,s,gammaO,gammaT) *
        //return R + TT ;
        return R + MP * NP * AP;
        //return vec3(M_p_Weta(sinTheta_i,cosTheta_i,sinTheta_o,cosTheta_o,beat_m));
    }
}