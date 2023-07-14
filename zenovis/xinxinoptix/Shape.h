#pragma once 
#include <Sampling.h>

struct LightSampleRecord {
    float3 p;
    float PDF;

    float3 n;
    float NoL;

    float3 dir;
    float dist;
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
};