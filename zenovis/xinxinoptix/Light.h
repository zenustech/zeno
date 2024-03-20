#pragma once
#include "Sampling.h"
#include "LightTree.h"

#include "TraceStuff.h"
#include "DisneyBRDF.h"
// #include "DisneyBSDF.h"
#include "proceduralSky.h"

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
    pdf = 1.0f;
    start = env_start[start];
    int i = start%nx;
    int j = start/nx;
    float theta = ((float)i + 0.5f)/(float) nx * 2.0f * 3.1415926f - 3.1415926f;
    float phi = ((float)j + 0.5f)/(float) ny * 3.1415926f;
    float twoPi2sinTheta = 2.0f * M_PIf * M_PIf * sin(phi);
    //pdf = env_cdf[start + nx*ny] / twoPi2sinTheta;
    vec3 dir = normalize(vec3(cos(theta), sin(phi - 0.5f * 3.1415926f), sin(theta)));
    dir = dir.rotY(to_radians(-params.sky_rot))
             .rotZ(to_radians(-params.sky_rot_z))
             .rotX(to_radians(-params.sky_rot_x))
             .rotY(to_radians(-params.sky_rot_y));
    return dir;
}

static __inline__ __device__ void cihouSphereLightUV(LightSampleRecord &lsr, GenericLight &light) {

    if (zeno::LightShape::Sphere == light.shape) {
        mat3 localAxis = {
            -reinterpret_cast<vec3&>(light.T), 
            -reinterpret_cast<vec3&>(light.N), 
            +reinterpret_cast<vec3&>(light.B) };

        auto sampleDir = localAxis * (lsr.n);
        lsr.uv = vec2(sphereUV(sampleDir, false));
    }
}

static __inline__ __device__ bool cihouMaxDistanceContinue(LightSampleRecord &lsr, GenericLight &light) {
    if (lsr.dist >= light.maxDistance) {
        return false;
    }
    
    if (light.maxDistance < FLT_MAX) {
        auto delta = 1.0f - lsr.dist / light.maxDistance;
        lsr.intensity *= smoothstep(0.0f, 1.0f, delta);
    }

    return true;
}

static __inline__ __device__ vec3 cihouLightEmission(LightSampleRecord &lsr, GenericLight &light, uint32_t depth) {

    auto intensity = (depth == 0 && light.vIntensity >= 0.0f) ? light.vIntensity : light.intensity;

    if (light.tex != 0u) {
        float3 color = (vec3)texture2D(light.tex, lsr.uv);
        if (light.texGamma != 1.0f) {
            color = pow(color, light.texGamma);
        }
        color = color * light.color;
        color = color * intensity;
        return color;
    }
    
    return light.color * intensity;
}

static __inline__ __device__ float sampleIES(const float* iesProfile, float h_angle, float v_angle) {

    if (iesProfile == nullptr) { return 0.0f; }

    int h_num = *(int*)iesProfile; ++iesProfile;
    int v_num = *(int*)iesProfile; ++iesProfile;

    const float* h_angles = iesProfile; iesProfile+=h_num;
    const float* v_angles = iesProfile; iesProfile+=v_num;
    const float* intensity = iesProfile;

    auto h_angle_min = h_angles[0];
    auto h_angle_max = h_angles[h_num-1];

    if (h_angle > h_angle_max || h_angle < h_angle_min) { return 0.0f; }

    auto v_angle_min = v_angles[0];
    auto v_angle_max = v_angles[v_num-1];

    if (v_angle > v_angle_max || v_angle < v_angle_min) { return 0.0f; }

    auto lambda = [](float angle, const float* angles, uint num) -> uint { 

        auto start = 0u, end = num-1u;
        auto _idx_ = start;

        while (start<end) {
            _idx_ = (start + end) / 2;

            if(angles[_idx_] > angle) {
                end = _idx_; continue;
            }

            if(angles[_idx_+1] < angle) {
                start = _idx_+1; continue;
            }

            break;
        }
        return _idx_;
    };

    auto v_idx = lambda(v_angle, v_angles, v_num);
    auto h_idx = lambda(h_angle, h_angles, h_num);

    auto _a_ = intensity[h_idx * v_num + v_idx];
    auto _b_ = intensity[h_idx * v_num + v_idx+1];

    auto _c_ = intensity[(h_idx+1) * v_num + v_idx];
    auto _d_ = intensity[(h_idx+1) * v_num + v_idx+1];

    auto v_ratio = (v_angle-v_angles[v_idx]) / (v_angles[v_idx+1]-v_angles[v_idx]);
    auto h_ratio = (h_angle-h_angles[h_idx]) / (h_angles[h_idx+1]-h_angles[h_idx]);

    v_ratio = clamp(v_ratio, 0.0f, 1.0f);
    h_ratio = clamp(h_ratio, 0.0f, 1.0f);

    auto _ab_ = mix(_a_, _b_, v_ratio);
    auto _cd_ = mix(_c_, _d_, v_ratio);

    return mix(_ab_, _cd_, h_ratio);
}

static __inline__ __device__ void sampleSphereIES(LightSampleRecord& lsr, const float2& uu, const float3& shadingP, const float3& center, float radius) {

    float3 vector = center - shadingP;
    float dist2 = dot(vector, vector);

    if (dist2 < radius * radius) {
        lsr.PDF = 0.0f;
        return; 
    }

    lsr.dist = sqrtf(dist2);
    lsr.dir = vector/lsr.dist;
    lsr.n = -lsr.dir;

    lsr.p = center;
    if (radius > 0) {
        lsr.p += radius * lsr.n;
        lsr.p = rtgems::offset_ray(lsr.p, lsr.n);
        lsr.dist = length(lsr.p - shadingP);
    }

    lsr.NoL = 1.0f;
    lsr.PDF = 1.0f;

    lsr.intensity = 1.0f / dist2;
}

static __inline__ __device__ float light_spread_attenuation(
                                            const float3& ray_dir,
                                            const float3& normal,
                                            const float spread,
                                            const float tan_void,
                                            const float spreadNormalize)
{
    const float cos_a = -dot(ray_dir, normal);
    auto angle_a = acosf(fabsf(cos_a));
    auto angle_b = spread * 0.5f * M_PIf;

    if (angle_a > angle_b) {
        return 0.0f;
    }
    angle_a = clamp(angle_a, 0.0f, 1.52367f);
    const float tan_a = tanf(angle_a);
    return fmaxf((1.0f - tan_void * tan_a) * spreadNormalize, 0.0f);
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
static __forceinline__ __device__
void DirectLighting(RadiancePRD *prd, ShadowPRD& shadowPRD, const float3& shadingP, const float3& ray_dir, 
                    TypeEvalBxDF& evalBxDF, TypeAux* taskAux=nullptr, float3* RadianceWithoutShadow=nullptr) {

    const float3 wo = normalize(-ray_dir); 
    float3 light_attenuation = vec3(1.0f);

    const float _SKY_PROB_ = params.skyLightProbablity();

    float scatterPDF = 1.f;
    float UF = prd->rndf();

    if(UF >= _SKY_PROB_) {

        if (params.num_lights == 0u || params.lightTreeSampler == 0u) return;

        auto lightTree = reinterpret_cast<pbrt::LightTreeSampler*>(params.lightTreeSampler);
        if (lightTree == nullptr) return;

        float lightPickProb = 1.0f - _SKY_PROB_;
        UF = (UF - _SKY_PROB_) / lightPickProb;

        const Vector3f& SP = reinterpret_cast<const Vector3f&>(shadingP);
        const Vector3f& SN = reinterpret_cast<const Vector3f&>(prd->geometryNormal);

        auto pick = lightTree->sample(UF, SP, SN);
        if (pick.prob <= 0.0f) { return; }

        uint lighIdx = min(pick.lightIdx, params.num_lights-1);
        auto& light = params.lights[lighIdx];

        bool enabled = light.mask & prd->lightmask;
        if (!enabled) { return; }

        lightPickProb *= pick.prob;
        LightSampleRecord lsr;

        const float* iesProfile = reinterpret_cast<const float*>(light.ies);

        if (light.type == zeno::LightType::IES && nullptr != iesProfile) {

            auto radius = (light.shape == zeno::LightShape::Sphere)? light.sphere.radius : 0.0f;
        
            sampleSphereIES(lsr, {}, shadingP, light.cone.p, radius);
            if (lsr.PDF <= 0.0f) return; 

            auto v_angle = acosf(dot(-lsr.dir, light.N));
            auto h_angle = acosf(dot(-lsr.dir, light.T));

            auto intensity = sampleIES(iesProfile, h_angle, v_angle);
            if (intensity <= 0.0f) return;
            lsr.intensity *= intensity;
        } 
        else if (light.type == zeno::LightType::Spot) {

            light.cone.sample(&lsr, {0,0}, shadingP);
            lsr.isDelta = true;
            
            if (lsr.intensity <= 0) { return; }

            auto n_len = dot(-lsr.dir, light.N);
            auto t_len = dot(-lsr.dir, light.T);
            auto b_len = dot(-lsr.dir, light.B);

            auto tanU = t_len / n_len;
            auto tanV = b_len / n_len;

            auto hfov = tanf(light.spreadMajor * 0.5f * M_PIf);
            if (fabsf(tanU) > hfov || fabsf(tanV) > hfov) {return;}
            lsr.uv = 0.5f + 0.5f * float2 { tanU/hfov, tanV/hfov };
        }
        else if (light.type == zeno::LightType::Projector) {

            light.point.SampleAsLight(&lsr, {}, shadingP);
            lsr.isDelta = true;

            auto n_len = dot(-lsr.dir, light.N);
            auto t_len = dot(-lsr.dir, light.T);
            auto b_len = dot(-lsr.dir, light.B);

            if (n_len <= 0) {return;}

            auto tanU = t_len / n_len;
            auto tanV = b_len / n_len;

            auto spreadU = clamp(light.spreadMajor, 0.001f, 0.999f);
            auto spreadV = clamp(light.spreadMinor, 0.000f, 0.999f);
            if (spreadV <= 0) { spreadV = spreadU; }

            auto Ufov = tanf(spreadU * 0.5f * M_PIf);
            auto Vfov = tanf(spreadV * 0.5f * M_PIf);
            
            if (fabsf(tanU) > Ufov || fabsf(tanV) > Vfov) {return;}
            lsr.uv = 0.5f + 0.5f * float2 { tanU/Ufov, tanV/Vfov };
        }
        else if(light.shape == zeno::LightShape::Point) {
            light.point.SampleAsLight(&lsr, {}, shadingP);
        }
        else if (light.type == zeno::LightType::Direction) {

            bool valid = false;
            switch (light.shape) {
                case zeno::LightShape::Plane:
                case zeno::LightShape::Ellipse: {
                    valid = light.rect.hitAsLight(&lsr, shadingP, -light.N); break;
                }
                case zeno::LightShape::Sphere: {
                    auto dir = normalize(light.sphere.center - shadingP);
                    valid = light.sphere.hitAsLight(&lsr, shadingP, dir); 
                    if (valid) {
                        cihouSphereLightUV(lsr, light);
                        lsr.intensity *= 1.0f / (lsr.dist * lsr.dist); 
                    }
                    break;
                }
                default: return;
            }
            if (!valid) { return; }

            lsr.intensity *= 2.0f * M_PIf;
            lsr.PDF = 1.0f;
            lsr.NoL = 1.0f;
            lsr.isDelta = true;

        } else { // Diffuse

            float2 uu = {prd->rndf(), prd->rndf()};

            switch (light.shape) {
                case zeno::LightShape::Plane: {

                    auto rect = light.rect; 
                    float2 uvScale, uvOffset;
                    bool valid = SpreadClampRect(rect.v, rect.axisX, rect.lenX, rect.axisY, rect.lenY, 
                                                rect.normal, shadingP, 
                                                light.spreadMajor, uvScale, uvOffset);
                    if (!valid) return;

                    rect.SampleAsLight(&lsr, uu, shadingP);
                    lsr.uv = uvOffset + lsr.uv * uvScale;
                    break;
                }
                case zeno::LightShape::Ellipse: { 

                    auto rect = light.rect;
                    float2 uvScale, uvOffset;
                    bool valid = SpreadClampRect(rect.v, rect.axisX, rect.lenX, rect.axisY, rect.lenY, 
                                                rect.normal, shadingP, 
                                                light.spreadMajor, uvScale, uvOffset, light.rect.isEllipse);
                    if (!valid) return;

                    rect.isEllipse = false; // disable ellipse test for sub rect
                    rect.SampleAsLight(&lsr, uu, shadingP);
                    lsr.uv = uvOffset + lsr.uv * uvScale;
                    if (length(lsr.uv-0.5f) > 0.5f) { 
                        return; // not inside ellipse
                    }
                    break;
                }   
                case zeno::LightShape::Sphere: {
                    light.sphere.SampleAsLight(&lsr, uu, shadingP); 
                    cihouSphereLightUV(lsr, light);
                    break; 
                }   
                case zeno::LightShape::TriangleMesh: {
                    float3* normalBuffer = reinterpret_cast<float3*>(params.triangleLightNormalBuffer);
                    float2* coordsBuffer = reinterpret_cast<float2*>(params.triangleLightCoordsBuffer);
                    light.triangle.SampleAsLight(&lsr, uu, shadingP, prd->geometryNormal, normalBuffer, coordsBuffer); break;
                }
                default: break;
            }

            if (light.spreadMajor < 1.0f) {
                
                auto void_angle = 0.5f * (1.0f - light.spreadMajor) * M_PIf;
                auto atten = light_spread_attenuation(
                                        lsr.dir,
                                        lsr.n,
                                        light.spreadMajor,
                                        tanf(void_angle),
                                        light.spreadNormalize);
                lsr.intensity *= atten;
            }
        }

        lsr.p -= params.cam.eye;
        //lsr.p = rtgems::offset_ray(lsr.p, lsr.n);
        lsr.dist = length(lsr.p - shadowPRD.origin);

        if (!cihouMaxDistanceContinue(lsr, light)) { return; }
        
        float3 emission = cihouLightEmission(lsr, light, prd->depth);

        lsr.PDF *= lightPickProb;

        if (light.config & zeno::LightConfigDoubleside) {
            lsr.NoL = abs(lsr.NoL);
        }

        if (light.falloffExponent != 2.0f) {
            lsr.intensity *= powf(lsr.dist, 2.0f-light.falloffExponent);
        }

        if (lsr.NoL > _FLT_EPL_ && lsr.PDF > 1e-2) {

            shadowPRD.lightIdx = lighIdx;
            shadowPRD.maxDistance = lsr.dist;
            
            traceOcclusion(params.handle, shadowPRD.origin, lsr.dir, 0, lsr.dist, &shadowPRD);
            light_attenuation = shadowPRD.attanuation;

            if (nullptr==RadianceWithoutShadow && lengthSquared(light_attenuation) == 0.0f) return;

            emission *= lsr.intensity;
            auto bxdf_value = evalBxDF(lsr.dir, wo, scatterPDF);
            auto misWeight = 1.0f;

            if constexpr(_MIS_) {
                if (!light.isDeltaLight() && !lsr.isDelta) {
                    misWeight = BRDFBasics::PowerHeuristic(lsr.PDF, scatterPDF);
                }
            }

                float3 radianceNoShadow = emission * bxdf_value;
                radianceNoShadow *= misWeight / (lsr.PDF + 1e-4);

                if (nullptr != RadianceWithoutShadow) {
                    *RadianceWithoutShadow = radianceNoShadow;
                }

                if constexpr (!detail::is_void<TypeAux>::value) {
                    auto tmp = light_attenuation * misWeight / (lsr.PDF + 1e-4);
                    (*taskAux)(emission * tmp);
                }// TypeAux

                prd->radiance = radianceNoShadow * light_attenuation; // with shadow
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
            if(tmpPdf <= 0.0f) { return; }

            auto Ldir = sun_dir;

            if (envpdf < __FLT_DENORM_MIN__) {
                return;
            }

            shadowPRD.maxDistance = 1e16f;
            traceOcclusion(params.handle, shadowPRD.origin, sun_dir,
                        1e-5f, // tmin
                        1e16f, // tmax,
                        &shadowPRD);
            light_attenuation = shadowPRD.attanuation;

            if (nullptr==RadianceWithoutShadow && lengthSquared(light_attenuation) == 0.0f) return;

            auto inverseProb = 1.0f/_SKY_PROB_;
            auto bxdf_value = evalBxDF(sun_dir, wo, scatterPDF);

            float tmp = 1.0f;

            if constexpr(_MIS_) {
                float misWeight = BRDFBasics::PowerHeuristic(tmpPdf, scatterPDF);
                misWeight = misWeight>0.0f?misWeight:1.0f;
                misWeight = scatterPDF>1e-5f?misWeight:0.0f;
                misWeight = tmpPdf>1e-5f?misWeight:0.0f;

                tmp = (1.0f / NSamples) * misWeight * inverseProb  / tmpPdf;
            } else {
                tmp = (1.0f / NSamples) * inverseProb / tmpPdf;
            }

            float3 radianceNoShadow = illum * tmp * bxdf_value; 

            if (nullptr != RadianceWithoutShadow) {
                *RadianceWithoutShadow = radianceNoShadow;
            }

            if constexpr (!detail::is_void<TypeAux>::value) {
                (*taskAux)(illum * tmp * light_attenuation);
            }// TypeAux

            prd->radiance += radianceNoShadow * light_attenuation; // with shadow
        }
    }
};