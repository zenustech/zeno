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

    float2 uv;

    float intensity = 1.0f;
    bool isDelta = false;
};

struct PointShape {
    float3 p;

    inline float PDF() {return 0.25f / M_PIf;}

    inline void SampleAsLight(LightSampleRecord* lsr, const float2& uu, const float3& shadingP) {

        auto vector = p - shadingP;
        auto dist2 = dot(vector, vector);
        auto dist = sqrtf(dist2);

        lsr->dist = dist;
        lsr->dir = vector / dist;
        lsr->p = p;
        lsr->n = -lsr->dir;

        lsr->PDF = 1.0f; //PDF();
        lsr->NoL = 1.0f;
        lsr->intensity = M_PIf / dist2;
        lsr->isDelta = true;
    }

    pbrt::LightBounds BoundAsLight(float phi, bool doubleSided) {

        float Phi = 4 * M_PIf * phi; 

        auto& tmp = reinterpret_cast<Vector3f&>(p);
        auto bounds = pbrt::Bounds3f{tmp, tmp};
        
        return pbrt::LightBounds(bounds, Vector3f(0, 0, 1), 
            Phi, cosf(M_PIf), cosf(M_PIf / 2), false);
    }
};

struct RectShape {
    float3 v0, v1, v2;
    float3 normal;
    float  area;

    inline float PDF() {
        return 1.0f / area;
    }

    inline void EvalAfterHit(LightSampleRecord* lsr, const float3& dir, const float& dist, const float3& shadingP) {

        float lightNoL = dot(-dir, normal);
        float lightPDF = dist * dist * PDF() / lightNoL;
        
        lsr->p = shadingP + dir * dist;
        lsr->n = normal;
        lsr->dir = dir;
        lsr->dist = dist;

        auto delta = lsr->p - v0;
        delta -= dot(delta, normal) * normal;

        lsr->uv = { dot(delta, v1) / dot(v1, v1),
                    dot(delta, v2) / dot(v2, v2) };

        lsr->PDF = lightPDF;
        lsr->NoL = lightNoL;
    }  

    inline void SampleAsLight(LightSampleRecord* lsr, const float2& uu, const float3& shadingP) {    

        lsr->n = normalize(normal);
        lsr->p = v0 + v1 * uu.x + v2 * uu.y;

        lsr->uv = uu;

        lsr->dir = (lsr->p - shadingP);
        //lsr->dir = normalize(lsr->dir);
        auto sign = copysignf(1.0f, dot(lsr->n, -lsr->dir));
        
        lsr->p = rtgems::offset_ray(lsr->p, lsr->n * sign);
        lsr->dir = lsr->p - shadingP;
        lsr->dist = length(lsr->dir);
        lsr->dir = lsr->dir / lsr->dist;

        lsr->NoL = dot(-lsr->dir, lsr->n);
        lsr->PDF = 0.0f;
        
        if (fabsf(lsr->NoL) > __FLT_EPSILON__) {
            lsr->PDF = lsr->dist * lsr->dist * PDF() / fabsf(lsr->NoL);
        }
    }

    inline bool hitAsLight(LightSampleRecord* lsr, const float3& ray_orig, const float3& ray_dir) {

        // assuming normal and ray_dir are normalized
        float denom = dot(normal, -ray_dir);
        if (denom <= __FLT_DENORM_MIN__) {return false;}
        
        float3 vector = ray_orig - v0;
        float t = dot(normal, vector) / denom;

        if (t <= 0) { return false; }

        auto P = ray_orig + ray_dir * t;
        auto delta = P - v0;

        P -= normal * dot(normal, delta);
        delta = P - v0; 
        
        auto v1v1 = dot(v1, v1);
        auto q1 = dot(delta, v1);
        if (q1<0.0f || q1>v1v1) {return false;}

        auto v2v2 = dot(v2, v2);        
        auto q2 = dot(delta, v2);
        if (q2<0.0f || q2>v2v2) {return false;}

        lsr->uv = float2{q1, q2} / float2{v1v1, v2v2};

        lsr->p = P;
        lsr->PDF = 1.0f;
        lsr->n = normal;
        lsr->NoL = denom;

        lsr->p = rtgems::offset_ray(lsr->p, lsr->n);

        lsr->dir = ray_dir;
        lsr->dist = length(lsr->p - ray_orig);

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
    float range;

    float3 dir;
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

        auto Phi = M_PIf * 4 * phi;

        float cosTheta_e = cosf(acosf(cosFalloffEnd) - acosf(cosFalloffStart));
        // Allow a little slop here to deal with fp round-off error in the computation of
        // cosTheta_p in the importance function.
        if (cosTheta_e == 1 && cosFalloffEnd != cosFalloffStart)
            cosTheta_e = 0.999f;

        auto& w = reinterpret_cast<Vector3f&>(dir);
        auto& tmp = reinterpret_cast<Vector3f&>(p);
        auto bounds = pbrt::Bounds3f{tmp, tmp};

        return pbrt::LightBounds(bounds, w, Phi, cosFalloffStart, cosTheta_e, false);
    }
};

struct SphereShape {
    float3 center;
    float  radius;
    float  area;

    inline float PDF() {
        return 1.0f / area;
    }

    inline float PDF(const float3& shadingP, float dist2, float NoL) {

        if (dist2 < radius * radius) {
            return dist2 / fabsf(NoL);
        }

        float sinThetaMax2 = clamp( radius * radius / dist2, 0.0, 1.0);

        if (sinThetaMax2 <= __FLT_EPSILON__) {
            return 1.0f; // point light
        }

        float cosThetaMax = sqrtf( 1.0 - sinThetaMax2 );
        return 1.0f / ( 2.0f * M_PIf * (1.0 - cosThetaMax) );
    }

    inline bool hitAsLight(LightSampleRecord* lsr, const float3& ray_origin, const float3& ray_dir) {

        float3 f = ray_origin - center;
        float b2 = dot(f, ray_dir);
        if (b2 >= 0) { return false; }

		float r2 = radius * radius;

		float3 fd = f - b2 * ray_dir;
		float discriminant = r2 - dot(fd, fd);

		if (discriminant >= 0.0f)
		{
			float c = dot(f, f) - r2;
			float sqrtVal = sqrt(discriminant);

			// include Press, William H., Saul A. Teukolsky, William T. Vetterling, and Brian P. Flannery, 
			// "Numerical Recipes in C," Cambridge University Press, 1992.
			float q = (b2 >= 0) ? -sqrtVal - b2 : sqrtVal - b2;

            lsr->dir = ray_dir;
            lsr->dist = fminf(c/q, q);
            lsr->p = ray_origin + ray_dir * lsr->dist;
            // lsr->n = normalize(lsr->p - center);
            // lsr->p = rtgems::offset_ray(lsr->p, lsr->n);
            // lsr->dist = length(lsr->p - ray_origin);
            return true;

			// we don't bother testing for division by zero
			//ReportHit(c / q, 0, sphrAttr);
			// more distant hit - not needed if we know we will intersect with the outside of the sphere
			//ReportHit(q / a, 0, sphrAttr);
		}
        return false;
    }

    inline void EvalAfterHit(LightSampleRecord* lsr, const float3& dir, const float& distance, const float3& shadingP) {

        auto vector = center - shadingP;
        auto dist2 = dot(vector, vector);
        auto dist = sqrtf(dist2);

        lsr->p = shadingP + dir * distance;
        lsr->n = normalize(lsr->p - center);
        if (dist2 < radius * radius) {
            lsr->n *= -1;
        }

        lsr->NoL = dot(lsr->n, -dir);
        lsr->PDF = PDF(shadingP, dist2, lsr->NoL);

        lsr->dir = dir;
        lsr->dist = distance;
    }

    inline void SampleAsLight(LightSampleRecord* lsr, const float2& uu, const float3& shadingP) {

        float3 vector = center - shadingP;
        float  dist2 = dot(vector, vector);
        float  dist = sqrtf(dist2);
        float3 dir = vector / dist;

        float radius2 = radius * radius;

        if (dist2 <= radius2) { // inside sphere
            
            auto localP = pbrt::UniformSampleSphere(uu);
            auto worldP = center + localP * radius;

            auto localN = -localP; //facing center
            auto worldN =  localN; 

            lsr->p = rtgems::offset_ray(worldP, worldN);
            lsr->n = worldN;

            vector = lsr->p - shadingP;
            dist2 = dot(vector, vector);

            if (dist2 == 0) {
                lsr->PDF = 0.0f; return;
            }

            dist = sqrtf(dist2);
            dir = vector / dist;

            lsr->dist = dist;
            lsr->dir  = dir;

            lsr->NoL = dot(-dir, worldN);
            lsr->PDF = lsr->dist * lsr->dist / lsr->NoL;
            return;       
        }

        assert(dist > radius);

        // Sample sphere uniformly inside subtended cone
        float invDc = 1.0f / dist;
        float3& wc = dir; float3 wcX, wcY;
        pbrt::CoordinateSystem(wc, wcX, wcY);

        // Compute $\theta$ and $\phi$ values for sample in cone
        float sinThetaMax = radius * invDc;
        const float sinThetaMax2 = sinThetaMax * sinThetaMax;
        float invSinThetaMax = 1.0f / sinThetaMax;

        assert(sinThetaMax2 > 0);
        const float cosThetaMax = sqrtf(1.0f - clamp(sinThetaMax2, 0.0f, 1.0f));

        auto epsilon = 2e-3f;

        if (sinThetaMax < epsilon) {
            
            lsr->p = center - dir * radius;
            lsr->p = rtgems::offset_ray(lsr->p, -dir);

            lsr->n = -dir;
            lsr->dir = dir;
            lsr->dist = length(lsr->p - shadingP);

            lsr->PDF = 1.0f;
            lsr->NoL = 1.0f;
            lsr->intensity = M_PIf * radius2 / (lsr->dist * lsr->dist);
            lsr->isDelta = true;
            return;
        } // point light

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
        float sinAlpha = sqrtf(fmaxf(0.f, 1.f - cosAlpha * cosAlpha));
        float phi = uu.y * 2 * M_PIf;

        // Compute surface normal and sampled point on sphere
        float3 nWorld = pbrt::SphericalDirection(sinAlpha, cosAlpha, phi, -wcX, -wcY, -wc);
        float3 pWorld = center + radius * nWorld;

        lsr->p = rtgems::offset_ray(pWorld, nWorld);
        lsr->n = nWorld;

        vector = lsr->p - shadingP;
        dist2 = dot(vector, vector);
        dist = sqrtf(dist2);
        dir = vector / dist;

        lsr->dist = dist;
        lsr->dir  = dir;
        
        lsr->PDF = 1.0f / (2.0f * M_PIf * (1.0f - cosThetaMax)); // Uniform cone PDF.
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
