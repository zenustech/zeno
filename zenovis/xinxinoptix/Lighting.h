#pragma once

#include "TraceStuff.h"
#include "zxxglslvec.h"

#include "DisneyBRDF.h"
#include "DisneyBSDF.h"

#include <cuda/random.h>
#include <cuda/helpers.h>
#include <sutil/vec_math.h>

static __inline__ __device__
int GetLightIndex(float p, ParallelogramLight* lightP, int n)
{
    int s = 0, e = n-1;
    while( s < e )
    {
        int j = (s+e)/2;
        float pc = lightP[j].cdf/lightP[n-1].cdf;
        if(pc<p)
        {
            s = j+1;
        }
        else
        {
            e = j;
        }
    }
    return e;
}

static __inline__ __device__
vec3 ImportanceSampleEnv(float* env_cdf, int* env_start, int nx, int ny, float p, float &pdf)
{
    if(nx*ny == 0)
    {
        pdf = 1.0f;
        return vec3(0);
    }
    int start = 0; int end = nx*ny-1;
    while(start<end-1)
    {
        int mid = (start + end)/2;
        if(env_cdf[mid]<p)
        {
            start = mid;
        }
        else
        {
            end = mid;
        }
    }
    start = env_start[start];
    int i = start%nx;
    int j = start/nx;
    float theta = ((float)i + 0.5f)/(float) nx * 2.0f * 3.1415926f - 3.1415926f;
    float phi = ((float)j + 0.5f)/(float) ny * 3.1415926f;
    float twoPi2sinTheta = 2.0f * M_PIf * M_PIf * sin(phi);
    pdf = env_cdf[start + nx*ny] / twoPi2sinTheta;
    vec3 dir = normalize(vec3(cos(theta), sin(phi - 0.5f * 3.1415926f), sin(theta)));
    dir = dir.rotY(to_radians(-params.sky_rot))
             .rotZ(to_radians(-params.sky_rot_z))
             .rotX(to_radians(-params.sky_rot_x))
             .rotY(to_radians(-params.sky_rot_y));
    return dir;
}


template<bool _MIS_, typename TypeEvalBxDF, typename TypeAux>
static __inline__ __device__
void DirectLighting(RadiancePRD *prd, RadiancePRD& shadow_prd, const float3& P, const float3& ray_dir, TypeEvalBxDF& evalBxDF, TypeAux& taskAux) {

    const float3 wo = normalize(-ray_dir); 

    float3 light_attenuation = vec3(1.0f);
    float pl = prd->rndf();
    int lidx = GetLightIndex(pl, params.lights, params.num_lights);
    float sum = 0.0f;
    for(int lidx=0;lidx<params.num_lights;lidx++)
    {
        ParallelogramLight light = params.lights[lidx];
        float3 light_pos = light.corner + light.v1 * 0.5 + light.v2 * 0.5;

        // Calculate properties of light sample (for area based pdf)
        float Ldist = length(light_pos - P);
        float3 L = normalize(light_pos - P);
        float nDl = 1.0f;//clamp(dot(N, L), 0.0f, 1.0f);
        float LnDl = clamp(-dot(light.normal, L), 0.000001f, 1.0f);
        float A = length(cross(params.lights[lidx].v1, params.lights[lidx].v2));
        sum += length(light.emission)  * nDl * LnDl * A / (M_PIf * Ldist * Ldist );
    }

    const float ProbSky = 0.5f;
    float directionPDF = 1.0f;

    if(prd->rndf() > ProbSky) {
        bool computed = false;
        float ppl = 0;
        for (int lidx = 0; lidx < params.num_lights && computed == false; lidx++) {
            ParallelogramLight light = params.lights[lidx];
            float2 z = {prd->rndf(), prd->rndf()};
            const float z1 = z.x;
            const float z2 = z.y;
            float3 light_tpos = light.corner + light.v1 * 0.5f + light.v2 * 0.5f;
            float3 light_pos = light.corner + light.v1 * z1 + light.v2 * z2;

            // Calculate properties of light sample (for area based pdf)
            float tLdist = length(light_tpos - P);
            float3 tL = normalize(light_tpos - P);
            float tnDl = 1.0f; //clamp(dot(N, tL), 0.0f, 1.0f);
            float tLnDl = clamp(-dot(light.normal, tL), 0.000001f, 1.0f);
            float tA = length(cross(params.lights[lidx].v1, params.lights[lidx].v2));
            ppl += length(light.emission) * tnDl * tLnDl * tA / (M_PIf * tLdist * tLdist) / sum;
            if (ppl > pl) {
                float Ldist = length(light_pos - P) + 1e-6f;
                float3 L = normalize(light_pos - P);
                float nDl = 1.0f; //clamp(dot(N, L), 0.0f, 1.0f);
                float LnDl = clamp(-dot(light.normal, L), 0.0f, 1.0f);
                float A = length(cross(params.lights[lidx].v1, params.lights[lidx].v2));
                float weight = 0.0f;
                if (nDl > 0.0f && LnDl > 0.0f) {

                    traceOcclusion(params.handle, P, L,
                                   1e-5f,         // tmin
                                   Ldist - 1e-5f, // tmax,
                                   &shadow_prd);

                    light_attenuation = shadow_prd.shadowAttanuation;
                    if (fmaxf(light_attenuation) > 0.0f) {

                        weight = sum * nDl / tnDl * LnDl / tLnDl * (tLdist * tLdist) / (Ldist  * Ldist) /
                                 (length(light.emission)+1e-6f) ;
                    }
                }

                auto inverseProb = 1.0f/(1.0f-ProbSky);
                auto bxdf_value = evalBxDF(L, wo, directionPDF);

                prd->radiance = light_attenuation * weight * inverseProb * light.emission * bxdf_value;
                computed = true;
            }
        }
    } else {

        float env_weight_sum = 1e-8f;
        int NSamples = prd->depth<=2?1:1;//16 / pow(4.0f, (float)prd->depth-1);
        for(int samples=0;samples<NSamples;samples++) {

            bool hasenv = params.skynx | params.skyny;
            hasenv = params.usingHdrSky && hasenv;
            float envpdf = 1.0f;

            vec3 sunLightDir = hasenv? ImportanceSampleEnv(params.skycdf, params.sky_start,
                                                            params.skynx, params.skyny, rnd(prd->seed), envpdf)
                                    : vec3(params.sunLightDirX, params.sunLightDirY, params.sunLightDirZ);
            auto sun_dir = BRDFBasics::halfPlaneSample(prd->seed, sunLightDir,
                                                    params.sunSoftness * 0.0f); //perturb the sun to have some softness
            sun_dir = hasenv ? normalize(sunLightDir):normalize(sun_dir);

            float tmpPdf;
            auto illum = float3(envSky(sun_dir, sunLightDir, make_float3(0., 0., 1.),
                                        40, // be careful
                                        .45, 15., 1.030725f * 0.3f, params.elapsedTime, tmpPdf));

            auto LP = P;
            auto Ldir = sun_dir;

            //LP = rtgems::offset_ray(LP, sun_dir);
            traceOcclusion(params.handle, LP, sun_dir,
                        1e-5f, // tmin
                        1e16f, // tmax,
                        &shadow_prd);

            light_attenuation = shadow_prd.shadowAttanuation;

            auto inverseProb = 1.0f/ProbSky;
            auto bxdf_value = evalBxDF(sun_dir, wo, directionPDF, illum);

            vec3 tmp(1.0f);

            if constexpr(_MIS_) {
                float misWeight = BRDFBasics::PowerHeuristic(envpdf, directionPDF);
                misWeight = misWeight>0.0f?misWeight:1.0f;
                misWeight = directionPDF>1e-5f?misWeight:0.0f;
                misWeight = envpdf>1e-5?misWeight:0.0f;

                tmp = vec3(misWeight * 1.0f / (float)NSamples * light_attenuation  / envpdf * inverseProb);
            } else {
                tmp = (1.0f / NSamples) * light_attenuation * inverseProb;
            }

            prd->radiance += (float3)(tmp) * bxdf_value;
            taskAux(tmp);
        }
    }
};