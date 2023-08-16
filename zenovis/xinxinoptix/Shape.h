#pragma once 
#include <Sampling.h>
#include <LightBounds.h>
#include <sutil/vec_math.h>

#ifdef __CUDACC_RTC__
#include "zxxglslvec.h"
#endif

struct LightSampleRecord {
    float3 p;
    float PDF;

    float3 n;
    float NoL;

    float3 dir;
    float dist;

    float intensity = 1.0f;
};

struct PointShape {
    float3 p;

    inline float PDF() {return 0.25f / M_PIf;}

    inline void sample(LightSampleRecord* lsr, const float2& uu, const float3& shadingP) {

        auto vector = p - shadingP;
        auto dist2 = dot(vector, vector);
        auto dist = sqrtf(dist2);

        lsr->dist = dist;
        lsr->dir = vector / dist;
        lsr->p = p;
        lsr->n = lsr->dir;

        lsr->PDF = PDF();
        lsr->NoL = 1.0f;
        lsr->intensity = 1.0 / dist2;
    }

    pbrt::LightBounds BoundAsLight() {

        float phi = 4 * M_PIf; 

        auto& tmp = reinterpret_cast<Vector3f&>(p);
        auto _bounds_ = pbrt::Bounds3f{tmp, tmp};
        
        return pbrt::LightBounds(_bounds_, Vector3f(0, 0, 1), 
            phi, cosf(M_PIf), cosf(M_PIf / 2), false);
    }
};

struct RectShape {
    float3 v0, v1, v2;
    float3 normal;
    float  area;

    inline float PDF() {
        return 1.0f / area;
    }

    inline void eval(LightSampleRecord* lsr, const float3& dir, const float& dist, const float3& shadingP) {

        float lightNoL = dot(-dir, normal);
        float lightPDF = dist * dist * PDF() / lightNoL;
        
        lsr->p = shadingP + dir * dist;
        lsr->n = normal;
        lsr->dir = dir;
        lsr->dist = dist;

        lsr->PDF = lightPDF;
        lsr->NoL = lightNoL;
    }  

    inline void sample(LightSampleRecord* lsr, const float2& uu, const float3& shadingP) {    

        lsr->p = v0 + v1 * uu.x + v2 * uu.y;
        lsr->n = normalize(normal);

        lsr->dir  = lsr->p - shadingP;
        lsr->dist = length(lsr->dir);
        lsr->dir  = normalize(lsr->dir);

        lsr->NoL = dot(-lsr->dir, lsr->n);
        lsr->PDF = 0.0f;
        
        if (fabsf(lsr->NoL) > __FLT_DENORM_MIN__) {
            lsr->PDF = lsr->dist * lsr->dist * PDF() / fabsf(lsr->NoL);
        }
    }

    inline bool hitAsLight(LightSampleRecord* lsr, const float3& ray_orig, const float3& ray_dir) {

        // assuming vectors are all normalized
        float denom = dot(normal, -ray_dir);
        if (denom <= __FLT_DENORM_MIN__) {return false;}
        
        float3 vector = ray_orig - v0;
        float t = dot(normal, vector) / denom;

        if (t <= 0) { return false; }

        auto P = ray_orig + ray_dir * t;
        auto delta = P - v0;

        auto v1v1 = dot(v1, v1);
        auto q1 = dot(delta, v1);
        if (q1<0.0f || q1>v1v1) {return false;}

        auto v2v2 = dot(v2, v2);        
        auto q2 = dot(delta, v2);
        if (q2<0.0f || q2>v2v2) {return false;}

        lsr->dir = ray_dir;
        lsr->dist = t;

        lsr->n = normal;
        lsr->NoL = denom;

        lsr->p = P;
        lsr->PDF = 1.0f;

        return true;
    }

    pbrt::Bounds3f bounds() {

        auto pmax = v0;
        auto pmin = v1;

        float3 tmp[3] = {v0+v1, v0+v2, v0+v1+v2};

        for (int i=0; i<3; i++) {
            pmax = fmaxf(pmax, tmp[i]);
            pmin = fminf(pmin, tmp[i]);
        }

        pbrt::Bounds3f result;
        result.pMax = reinterpret_cast<Vector3f&>(pmax);
        result.pMin = reinterpret_cast<Vector3f&>(pmin);

        return result;
    }

    pbrt::LightBounds BoundAsLight(float phi, bool doubleSided) {

        auto& nnn = reinterpret_cast<Vector3f&>(normal);
        auto dc = pbrt::DirectionCone(nnn);

        return pbrt::LightBounds(bounds(), nnn, phi * area, 
                dc.cosTheta, fmaxf(cosf(M_PIf / 2.0f), 0.0f), doubleSided);   
    }
};

struct ConeShape {
    float3 p;
    float3 dir;

    float bound;

    float cosFalloffStart;
    float cosFalloffEnd;

    inline void sample(LightSampleRecord* lsr, const float2& uu, const float3& shadingP) {
        auto vector = p - shadingP;
        auto dist2 = dot(vector, vector);
        auto dist = sqrtf(dist2);

        lsr->dir = vector / dist;
        lsr->dist = dist;

        lsr->n = dir;
        lsr->NoL = dot(lsr->dir, dir);

        lsr->p = p;
        lsr->PDF = 1.0f;

        #ifdef __CUDACC_RTC__
        lsr->intensity = smoothstep(cosFalloffEnd, cosFalloffStart, lsr->NoL);
        #endif

        lsr->intensity /= dist2;
    }

    inline float Phi() {
        return 2 * M_PIf * ((1.0f - cosFalloffStart) + (cosFalloffStart - cosFalloffEnd) / 2.0f);
    }

    pbrt::LightBounds BoundAsLight(float phi, bool doubleSided) {

        float cosTheta_e = cosf(acosf(cosFalloffEnd) - acosf(cosFalloffStart));
        // Allow a little slop here to deal with fp round-off error in the computation of
        // cosTheta_p in the importance function.
        if (cosTheta_e == 1 && cosFalloffEnd != cosFalloffStart)
            cosTheta_e = 0.999f;

        auto& w = reinterpret_cast<Vector3f&>(dir);
        auto& tmp = reinterpret_cast<Vector3f&>(p);
        auto bounds = pbrt::Bounds3f{tmp, tmp};

        return pbrt::LightBounds(bounds, w, 4 * M_PIf * phi, cosFalloffStart, cosTheta_e, false);
    }
};

struct SphereShape {
    float3 center;
    float  radius;
    float  area;

    inline float PDF() {
        return 1.0f / area;
    }

    inline float PDF(const float3& shadingP) {

        float3 vector = center - shadingP;

        auto dist2 = dot(vector, vector);
        auto dist = sqrtf(dist2);

        if (dist < radius) {
            return 1.0f / area;
        }

        float sinThetaMax2 = clamp( radius * radius / dist2, 0.0, 1.0);
        float cosThetaMax = sqrtf( 1.0 - sinThetaMax2 );
        return 1.0f / ( 2.0f * CUDART_PI_F * (1.0 - cosThetaMax) );
    }

    inline void eval(LightSampleRecord* lsr, const float3& dir, const float& _dist_, const float3& shadingP) {

        auto vector = center - shadingP;
        auto dist2 = dot(vector, vector);
        auto dist = sqrtf(dist2);

        lsr->PDF = PDF(shadingP);
        lsr->p = shadingP + dir * _dist_;
        lsr->n = normalize(lsr->p - center);

        lsr->dir = dir;
        lsr->dist = _dist_;

        lsr->NoL = dot(-lsr->dir, lsr->n); 
    }

    inline void sample(LightSampleRecord* lsr, const float2& uu, const float3& shadingP) {

        float3 vector = center - shadingP;
        float  dist2 = dot(vector, vector);
        float  dist = sqrtf(dist2);
        float3 dir = vector / dist;

        if (dist <= radius) { // inside sphere
            
            auto localP = pbrt::UniformSampleSphere(uu);
            auto worldP = center + localP * radius;

            auto localN = -localP; //facing center
            auto worldN =  localN; 

            lsr->p = worldP;
            lsr->n = worldN;
            lsr->PDF = 1.0f / (4.0f * CUDART_PI_F);

            lsr->dir  = worldP - shadingP;
            lsr->dist = length(lsr->dir);
            lsr->dir  = normalize(lsr->dir);

            lsr->NoL = dot(-dir, worldN);
            return;       
        }

        // Sample sphere uniformly inside subtended cone
        float invDc = 1 / dist;
        float3& wc = dir; float3 wcX, wcY;
        // Compute coordinate system for sphere sampling
        pbrt::CoordinateSystem(wc, wcX, wcY);

        // Compute $\theta$ and $\phi$ values for sample in cone
        float sinThetaMax = radius * invDc;
        float sinThetaMax2 = sinThetaMax * sinThetaMax;
        float invSinThetaMax = 1 / sinThetaMax;
        float cosThetaMax = sqrtf(fmaxf(0.0f, 1.0f - sinThetaMax2));

        float cosTheta  = (cosThetaMax - 1) * uu.x + 1;
        float sinTheta2 = 1 - cosTheta * cosTheta;

        if (sinThetaMax2 < 0.00068523f /* sin^2(1.5 deg) */) {
            /* Fall back to a Taylor series expansion for small angles, where
            the standard approach suffers from severe cancellation errors */
            sinTheta2 = sinThetaMax2 * uu.x;
            cosTheta = sqrtf(1 - sinTheta2);
        }

        // Compute angle $\alpha$ from center of sphere to sampled point on surface
        float cosAlpha = sinTheta2 * invSinThetaMax +
            cosTheta * sqrtf(fmaxf(0.f, 1.f - sinTheta2 * invSinThetaMax * invSinThetaMax));
        float sinAlpha = sqrtf(fmaxf(0.f, 1.f - cosAlpha*cosAlpha));
        float phi = uu.y * 2 * CUDART_PI_F;

        // Compute surface normal and sampled point on sphere
        float3 nWorld = pbrt::SphericalDirection(sinAlpha, cosAlpha, phi, -wcX, -wcY, -wc);
        float3 pWorld = center + radius * make_float3(nWorld.x, nWorld.y, nWorld.z);

        lsr->p = pWorld;
        lsr->n = nWorld;
        lsr->PDF = 1.0f / (2.0f * CUDART_PI_F * (1.0f - cosThetaMax)); // Uniform cone PDF.

        lsr->dir  = (pWorld - shadingP);
        lsr->dist = length(lsr->dir);
        lsr->dir  = normalize(lsr->dir);

        lsr->NoL = dot(-lsr->dir, lsr->n); 
    }

    pbrt::Bounds3f bounds() {

        auto pmax = center + make_float3(abs(radius));
        auto pmin = center - make_float3(abs(radius));

        pbrt::Bounds3f result;
        result.pMax = reinterpret_cast<Vector3f&>(pmax);
        result.pMin = reinterpret_cast<Vector3f&>(pmin);

        return result;
    }

    pbrt::LightBounds BoundAsLight(float phi, bool doubleSided) {

        auto dc = pbrt::DirectionCone::EntireSphere();

        return pbrt::LightBounds(bounds(), dc.w, phi * area, 
            dc.cosTheta, fmaxf(cos(M_PIf / 2.0f), 0.0f), doubleSided);   
    }
};