#pragma once
#include "TraceStuff.h"
#include "zxxglslvec.h"

#include "DisneyBRDF.h"
#include "DisneyBSDF.h"

#include "Shape.h"

#include <cuda/random.h>
#include <cuda/helpers.h>
#include <sutil/vec_math.h>

static __inline__ __device__
int GetLightIndex(float p, GenericLight* lightP, int n)
{
    int s = 0, e = n-1;
    while( s < e )
    {
        int j = (s+e)/2;
        float pc = lightP[j].CDF/lightP[n-1].CDF;
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
namespace detail {
    template <typename T> struct is_void {
        static constexpr bool value = false;
    };
    template <> struct is_void<void> {
        static constexpr bool value = true;
    };
}

template<bool _MIS_, typename TypeEvalBxDF, typename TypeAux = void>
static __inline__ __device__
void DirectLighting(RadiancePRD *prd, RadiancePRD& shadow_prd, const float3& shadingP, const float3& ray_dir, TypeEvalBxDF& evalBxDF, TypeAux* taskAux=nullptr) {

    const float3 wo = normalize(-ray_dir); 
    float3 light_attenuation = vec3(1.0f);

    const float ProbSky = params.num_lights>0? 0.5f : 1.0f;
    float scatterPDF = 1.0f;

    if(prd->rndf() > ProbSky) {

        float lightPickPDF = 1.0f - ProbSky;

        uint lighIdx = GetLightIndex(prd->rndf(), params.lights, params.num_lights);
        auto& light = params.lights[lighIdx];

        float prevCDF = lighIdx>0? params.lights[lighIdx-1].CDF : 0.0f;
        lightPickPDF *= (light.CDF - prevCDF) / params.lights[params.num_lights-1].CDF;

        LightSampleRecord lsr{};
        float2 uu = {prd->rndf(), prd->rndf()};

        if (light.shape == 0) {
            light.rect.sample(&lsr, uu, shadingP);
        } else {
            light.sphere.sample(&lsr, uu, shadingP);
        }

        lsr.PDF *= lightPickPDF;

        lsr.p = rtgems::offset_ray(lsr.p, lsr.n);

            if (light.config & LightConfigDoubleside) {
                lsr.NoL = abs(lsr.NoL);
            }

            if (lsr.NoL > _FLT_EPL_ && lsr.PDF > __FLT_DENORM_MIN__) {

                traceOcclusion(params.handle, shadingP, lsr.dir,
                                1e-5,         // tmin
                                lsr.dist - 2e-5f, // tmax,
                                &shadow_prd);
                light_attenuation = shadow_prd.shadowAttanuation;

                if (length(light_attenuation) > 0.0f) {
                    
                    auto bxdf_value = evalBxDF(lsr.dir, wo, scatterPDF);

                    auto misWeight = BRDFBasics::PowerHeuristic(lsr.PDF, scatterPDF);

                    prd->radiance = light_attenuation * light.emission * bxdf_value;
                    prd->radiance *= misWeight / lsr.PDF;
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

            auto LP = shadingP;
            auto Ldir = sun_dir;

            //LP = rtgems::offset_ray(LP, sun_dir);
            traceOcclusion(params.handle, LP, sun_dir,
                        1e-5f, // tmin
                        1e16f, // tmax,
                        &shadow_prd);

            light_attenuation = shadow_prd.shadowAttanuation;

            auto inverseProb = 1.0f/ProbSky;
            auto bxdf_value = evalBxDF(sun_dir, wo, scatterPDF, illum);

            vec3 tmp(1.0f);

            if constexpr(_MIS_) {
                float misWeight = BRDFBasics::PowerHeuristic(envpdf, scatterPDF);
                misWeight = misWeight>0.0f?misWeight:1.0f;
                misWeight = scatterPDF>1e-5f?misWeight:0.0f;
                misWeight = envpdf>1e-5?misWeight:0.0f;

                tmp = vec3(misWeight * 1.0f / (float)NSamples * light_attenuation  / envpdf * inverseProb);
            } else {
                tmp = (1.0f / NSamples) * light_attenuation * inverseProb;
            }

            prd->radiance += (float3)(tmp) * bxdf_value;

            if constexpr (!detail::is_void<TypeAux>::value) {
                if (taskAux != nullptr) {
                    (*taskAux)(tmp);
                }
            }// TypeAux
        }
    }
};