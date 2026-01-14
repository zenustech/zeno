#pragma once
#include "Sampling.h"
#include "LightTree.h"

#include "TraceStuff.h"
#include "DisneyBRDF.h"
// #include "DisneyBSDF.h"
#include "proceduralSky.h"

#include "Portal.h"

static __inline__ __device__
vec3 ImportanceSampleEnv(float* env_cdf, int* env_start, int nx, int ny, float p, float &pdf, float2& uv, float r0, float r1)
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
        if(__ldg(env_cdf+mid) < p)
            start = mid;
        else
            end = mid;
    }
    pdf = 1.0f;
    start = __ldg(env_start+start);
    int i = start%nx;
    int j = start/nx;

    uv = { (i+r0)/nx, (j+r1)/ny };
    uv.x = clamp(uv.x, 0.5f/nx, 1.0f - 0.5f/nx);
    uv.y = clamp(uv.y, 0.5f/ny, 1.0f - 0.5f/ny);

    float theta = uv.x * 2.0f * M_PIf - 0.5 * M_PIf;
    float phi = uv.y * M_PIf;
    //float twoPi2sinTheta = 2.0f * M_PIf * M_PIf * sinf(phi);
    //pdf = env_cdf[start + nx*ny] / twoPi2sinTheta;
    vec3 dir = vec3(cosf(theta), sinf(phi - 0.5f * M_PIf), sinf(theta));

    const auto& rotation = params.sky_onitator;
    dir = optix_impl::optixTransformVector(rotation[0], rotation[1], rotation[2], dir);
    return normalize(dir);
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

static __inline__ __device__ vec4 cihouLightEmission(LightSampleRecord &lsr, GenericLight &light, uint32_t depth) {

    auto intensity = (depth == 0 && light.vIntensity >= 0.0f) ? light.vIntensity : light.intensity;
    vec3 emission = light.color * intensity;

    if (light.tex != 0u) {
        vec4 rgba = texture2D(light.tex, lsr.uv);
        vec3 color = *(vec3*)&rgba;
        if (light.texGamma != 1.0f) {
            color = pow(color, light.texGamma);
        }
        color *= rgba.w;
        return vec4(color * emission, rgba.w);
    }
    return vec4(emission, 1.0f);
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
        lsr.dist = length(lsr.p - shadingP);
    }

    lsr.NoL = 1.0f;
    lsr.PDF = 1.0f;

    lsr.intensity = PointIntensity(dist2, lsr.dist, 1.0f);
}

static __inline__ __device__ float light_spread_attenuation(
                                            const float NdotL,
                                            const float spread,
                                            const float spreadNormalize)
{
    const float cos_a = NdotL;
    auto angle_a = acosf(fabsf(NdotL));
    auto angle_b = spread * 0.5f * M_PIf;
    
    auto angle_v = fmaxf(0.5f * M_PIf - angle_b, 0.0f);

    const float tan_a = tanf(angle_a);
    const float tan_void = tanf(angle_v);
    return fmaxf((1.0f - tan_void * tan_a) * spreadNormalize, 0.0f);
}

//tacken from sjb
static void sampleEquiAngular( vec3 ray_ori, vec3 ray_dir, float tmin, float tmax, float xi,
                               vec3 lightPos, float& dt, float& pdf)
{
    // get coord of closest point to light along (infinite) ray
    float delta = dot(lightPos - ray_ori, ray_dir);
    //if (delta<=0) return;
    
    // get distance this point is from light
    float D = length(ray_ori + delta * ray_dir - lightPos);

    // get angle of endpoints
    float thetaA = atan2f(tmin - delta, D);
    float thetaB = atan2f(tmax - delta, D);
    if (thetaB < thetaA) swap(thetaA, thetaB);

    float dTheta = thetaB - thetaA;
    if (fabs(dTheta) < 1e-3f) {
        // fallback: uniform in [tmin, tmax]
        dt  = mix(tmin, tmax, xi);
        pdf = 1.0f / fmaxf(tmax - tmin, 1e-5f);
        return;
    }
    
    // take sample
    float theta = mix(thetaA, thetaB, xi);
    float t = D * tanf(theta);
    dt = delta + t;
    pdf = D/(dTheta * (D*D + t*t));
};

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
void DirectLighting(ShadowPRD& shadowPRD, float3 shadingP, const float3& ray_dir, 
                    TypeEvalBxDF& evalBxDF, TypeAux* taskAux=nullptr, float3* RadianceWithoutShadow=nullptr) {

    const float3 wo = normalize(-ray_dir);
    const float _SKY_PROB_ = params.num_lights>0?0.5:1.0f;//no need to do importance...just half chance for the distant lights and half chance for the dynamic lights

    float scatterPDF = 1.f;

    auto prd = &shadowPRD;
    float UF = prd->rndf();

    if(UF >= _SKY_PROB_) {

        if (params.num_lights == 0u || params.lightTreeSampler == 0u) return;

        auto lightTree = reinterpret_cast<pbrt::LightTreeSampler*>(params.lightTreeSampler);
        if (lightTree == nullptr) return;

        float lightPickProb = 1.0f - _SKY_PROB_;
        UF = (UF - _SKY_PROB_) / lightPickProb;

        const Vector3f& SP = reinterpret_cast<const Vector3f&>(shadingP);
        const Vector3f& SN = reinterpret_cast<const Vector3f&>(shadowPRD.ShadowNormal);

        auto pick = lightTree->sample(UF, SP, SN, shadowPRD.fog_tmax);
        if (pick.prob <= 0.0f) { return; }

        uint lighIdx = min(pick.lightIdx, params.num_lights-1);
        auto& light = params.lights[lighIdx];

        // bool enabled = light.mask & prd->lightmask;
        // if (!enabled) { return; }

        lightPickProb *= pick.prob;
        LightSampleRecord lsr;

        float line_dt = 0.0f;
        float line_pdf = 1.0f;

        if (shadowPRD.fog_tmax > 0) {

            float t0 = 0;
            float t1 = shadowPRD.fog_tmax;

            float3 lightPos;

            if (zeno::LightShape::Point == light.shape) 
            {
                lightPos = light.point.p; 
                lightPos -= params.cam.eye;

                if (zeno::LightType::Spot == light.type) {

                    Interval I = intersectCone(shadowPRD.origin, ray_dir, lightPos, light.cone.dir, light.cone.cosFalloffEnd);
                    if (!I.hit) return;
                    
                    t0 = max(I.t_in, 0.0f);
                    t1 = min(I.t_out, shadowPRD.fog_tmax);

                    if (t0 >= t1) return;

                } else if (zeno::LightType::Projector == light.type) {

                    auto spreadU = clamp(light.spreadMajor, 0.001f, 0.999f);
                    auto spreadV = clamp(light.spreadMinor, 0.000f, 0.999f);
                    if (spreadV <= 0) { spreadV = spreadU; }

                    auto angleU = (spreadU * 0.5f * M_PIf);
                    auto angleV = (spreadV * 0.5f * M_PIf);

                    Interval I = intersectPyramid(shadowPRD.origin, ray_dir, lightPos, light.N, light.T, light.B, angleU, angleV);
                    if (!I.hit) return;

                    t0 = max(I.t_in, 0.0f);
                    t1 = min(I.t_out, shadowPRD.fog_tmax);
                }
            }
            else if (zeno::LightShape::Sphere == light.shape) 
            {
                lightPos = light.sphere.center - params.cam.eye;
            }
            else if (zeno::LightShape::Plane == light.shape || zeno::LightShape::Ellipse == light.shape) 
            {
                const auto& rect = light.rect;
                
                if (light.spreadMajor < 1.0f) {
                    auto angle = light.spreadMajor * 0.5f * M_PIf;
                    Interval I = intersectFrustum(shadowPRD.origin, ray_dir, rect.v-params.cam.eye, rect.axisX, rect.lenX, rect.axisY, rect.lenY, angle, angle);
                    if (!I.hit) return;

                    t0 = fmaxf(I.t_in, 0.0f);
                    t1 = fminf(I.t_out, shadowPRD.fog_tmax);
                } else {
                    auto diff = shadowPRD.origin - (rect.v-params.cam.eye);
                    auto vlen = dot(diff, rect.normal);
                    auto step = dot(ray_dir, rect.normal);
                    auto tt = fabs(vlen / step);

                    if (vlen > 0) { // under light
                        if (isfinite(tt) && step<0) // look up
                            t1 = fminf(t1, tt);
                    } else {        // above light
                        if (isfinite(tt) && step>0) // look down
                            t0 = fmaxf(t0, tt);
                        else
                            return;
                    }
                }

                let center = rect.center() - params.cam.eye;
                let& axisX = rect.axisX;
                let& axisY = rect.axisY;
                let& lenX = rect.lenX;
                let& lenY = rect.lenY;

                float3 pp[2]; 
                auto& p0 = pp[0];
                auto& p1 = pp[1];
                p0 = shadowPRD.origin + ray_dir * t0;
                p1 = shadowPRD.origin + ray_dir * t1;

                #pragma unroll
                for (char i=0; i<2; ++i)
                {
                    auto& p = pp[i];
                    auto delta = (p - center);
                    auto deltaX = dot(delta, axisX); 
                    auto deltaY = dot(delta, axisY);

                    auto tx = (rect.lenX * 0.5f) / deltaX;
                    auto ty = (rect.lenY * 0.5f) / deltaY;

                    if (!isfinite(tx)) tx = 1e20f;
                    if (!isfinite(ty)) ty = 1e20f;
                    
                    auto tt = fminf(abs(tx), abs(ty));
                    p = center + tt * deltaX * axisX + tt * deltaY * axisY;
                }

                auto scale = fmaxf(lenX, lenY) * 0.5f;
                lightPos = mix(p0, p1, prd->rndf());
                lightPos -= light.rect.normal * scale;
            } 
            else if (zeno::LightShape::TriangleMesh == light.shape) {
                lightPos = light.triangle.center() - params.cam.eye;
            }

            sampleEquiAngular(shadowPRD.origin, ray_dir, t0, t1, shadowPRD.rndf(), lightPos, line_dt, line_pdf);

            shadingP         += ray_dir * line_dt; // wolrd space
            shadowPRD.origin += ray_dir * line_dt; // camera space

            if (!isfinite(line_pdf)) { return; }

            lightPickProb *= line_pdf;
            shadowPRD.fog_dt = line_dt;
        }

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

            auto d = n_len * lsr.dist;
            lsr.intensity = PointIntensity(d * d, d, 1.0f);

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
                    light.sphere.SampleAsLight(&lsr, uu, shadingP, light.spreadMajor); 
                    cihouSphereLightUV(lsr, light);
                    break; 
                }   
                case zeno::LightShape::TriangleMesh: {
                    float3* normalBuffer = reinterpret_cast<float3*>(params.triangleLightNormalBuffer);
                    float2* coordsBuffer = reinterpret_cast<float2*>(params.triangleLightCoordsBuffer);
                    light.triangle.SampleAsLight(&lsr, uu, shadingP, *(float3*)&SN, normalBuffer, coordsBuffer); break;
                }
                default: break;
            }

            if (light.spreadMajor < 1.0f && light.spreadNormalize != 0.0f) {

                auto atten = light_spread_attenuation(
                                        lsr.NoL,
                                        light.spreadMajor,
                                        light.spreadNormalize);
                lsr.intensity *= atten;
            }
        }

        lsr.p -= params.cam.eye;
        lsr.dist = length(lsr.p - shadowPRD.origin);

        if (!cihouMaxDistanceContinue(lsr, light)) { return; }
        
        const auto rgba = cihouLightEmission(lsr, light, max(prd->depth, 1));
        float3 emission = *(float3*)&rgba;
        lsr.PDF *= lightPickProb;

        if (light.config & zeno::LightConfigDoubleside) {
            lsr.NoL = abs(lsr.NoL);
        }
        if (light.falloffExponent != 2.0f) {
            lsr.intensity *= powf(lsr.dist, 2.0f-light.falloffExponent);
        }
        emission *= lsr.intensity;
        if (sum(emission)==0) return;
        if (!isfinite(lsr.PDF)) return;

        if (lsr.NoL > _FLT_EPL_ && lsr.PDF > 1e-6f) {

            shadowPRD.lightIdx = lighIdx;
            shadowPRD.maxDistance = lsr.dist;
            
            traceOcclusion(params.handle, shadowPRD.origin, lsr.dir, 0, lsr.dist, &shadowPRD, ~LightMatMask & EverythingMask);
            auto light_attenuation = shadowPRD.attanuation;

            if (nullptr==RadianceWithoutShadow && lengthSquared(light_attenuation) == 0.0f) return;

            auto bxdf_value = evalBxDF(lsr.dir, wo, scatterPDF);
            auto misWeight = 1.0f;

            if constexpr(_MIS_) {
                if (!light.isDeltaLight() && !lsr.isDelta) {
                    misWeight = BRDFBasics::PowerHeuristic(lsr.PDF, scatterPDF);
                }
            }
                misWeight = misWeight / lsr.PDF;

                float3 radianceNoShadow = emission * bxdf_value;
                radianceNoShadow *= misWeight;

                if (nullptr != RadianceWithoutShadow) {
                    *RadianceWithoutShadow = radianceNoShadow;
                }

                if constexpr (!detail::is_void<TypeAux>::value) {
                    auto tmp = light_attenuation * misWeight;
                    (*taskAux)(emission * tmp);
                }// TypeAux

                prd->radiance += radianceNoShadow * light_attenuation; // with shadow
        } 
    
    }
    else{
        auto dlights = reinterpret_cast<DistantLightList*>(params.dlights_ptr);
        auto plights = reinterpret_cast<PortalLightList*>(params.plights_ptr);
        float dlight_wt = nullptr != dlights && dlights->COUNT()>0?1.0f:0.0f;
        float plight_wt = plights != nullptr && plights->COUNT()>0?1.0f:0.0f;
        float elight_wt = params.sky_strength>0?1.0f:0.0f;
        float total_wt = dlight_wt + plight_wt + elight_wt;
        if(total_wt==0) return;
        float dlight_pr = dlight_wt/total_wt;
        float plight_pr = plight_wt/total_wt;
        float elight_pr = elight_wt/total_wt;
        float selectp = prd->rndf();

        if (shadowPRD.fog_tmax > 0) {
            shadingP         += ray_dir * shadowPRD.fog_dt; // wolrd space
            shadowPRD.origin += ray_dir * shadowPRD.fog_dt; // camera space
        }

        auto shadeTask = [&](float3 sampleDir, float samplePDF, float3 illum, const bool mis) {

            shadowPRD.attanuation = vec3(1.0);
            shadowPRD.maxDistance = FLT_MAX;
            traceOcclusion(params.handle, shadowPRD.origin, sampleDir,
                        0, // tmin
                        FLT_MAX, // tmax,
                        &shadowPRD, ~LightMatMask & EverythingMask);


            if (nullptr==RadianceWithoutShadow && lengthSquared(shadowPRD.attanuation) == 0.0f) return;

            auto bxdf_value = evalBxDF(sampleDir, wo, scatterPDF);

            float tmp = 1.0f / samplePDF;

            if (mis) {
                float misWeight = BRDFBasics::PowerHeuristic(samplePDF, scatterPDF, 1.0);
                misWeight = misWeight>0.0f?misWeight:1.0f;
                misWeight = scatterPDF>1e-5f?misWeight:0.0f;
                misWeight = samplePDF>1e-5f?misWeight:0.0f;

                tmp *= misWeight;
            }

            float3 radianceNoShadow = illum * tmp * bxdf_value; 

            if (nullptr != RadianceWithoutShadow) {
                *RadianceWithoutShadow += radianceNoShadow;
            }

            if constexpr (!detail::is_void<TypeAux>::value) {
                (*taskAux)(illum * tmp * shadowPRD.attanuation);
            }// TypeAux
            prd->radiance += radianceNoShadow * shadowPRD.attanuation; // with shadow
        }; // shadeTask

        UF = UF / _SKY_PROB_;
        UF = clamp(UF, 0.0f, 1.0f);

        auto binsearch = [&](float* cdf, uint min, uint max) {
            //auto idx = min;
            while(min < max) {
                auto _idx_ = (min + max) / 2;
                auto _cdf_ = cdf[_idx_];

                if (_cdf_ > UF) {
                    max = _idx_; continue; //include
                }
                if (_cdf_ < UF) {
                    min = _idx_+1; continue;
                }
                min = _idx_; break;
            }
            return min;
        };


        
        if (nullptr != dlights && dlights->COUNT() && selectp<dlight_pr) {

            auto idx = binsearch(dlights->cdf, 0, dlights->COUNT());
            auto& dlight = dlights->list[idx];
            auto dlight_dir = reinterpret_cast<vec3&>(dlight.direction);

            auto sample_dir = BRDFBasics::halfPlaneSample(prd->seed, dlight_dir, dlight.angle/180.0f);
            auto sample_prob = 1.0f / dlights->COUNT();

            if (dlight.intensity > 0) {
                auto ccc = dlight.color * dlight.intensity;
                auto illum = reinterpret_cast<float3&>(ccc);
                shadeTask(sample_dir, sample_prob, illum / ( _SKY_PROB_ * dlight_pr), false);
            }
            return;
        }
        


        if (plights != nullptr && plights->COUNT() && selectp>dlight_pr && selectp<(dlight_pr + plight_pr)) {

            uint idx = binsearch(plights->cdf, 0, plights->COUNT());
            auto plight = &plights->list[idx];

            LightSampleRecord lsr; lsr.PDF = 0.0f;
            float2 uu = { prd->rndf(), prd->rndf() };
            float3 color {};
            
            plight->sample(lsr, reinterpret_cast<const Vector3f&>(shadingP), uu, color);
            
            lsr.PDF *= plights->pdf[idx];
            if (lsr.PDF > 0) {
                //auto suv = sphereUV(lsr.dir, true);
                //color = (vec3)texture2D(params.sky_texture, vec2(suv.x, suv.y));
                shadeTask(lsr.dir, lsr.PDF, color * params.sky_strength/ ( _SKY_PROB_ * plight_pr), false);
            }
            return;
        }

        if(selectp>(dlight_pr + plight_pr)&&selectp<1)
        { // SKY
            bool hasenv = params.skynx | params.skyny;
            hasenv = params.usingHdrSky && hasenv;
            float envpdf = 1.0f;

            vec3 sunLightDir = vec3(params.sunLightDirX, params.sunLightDirY, params.sunLightDirZ);

            float2 skyuv = {};
            vec3 sample_dir = hasenv? ImportanceSampleEnv(params.skycdf, params.sky_start,
                                                            params.skynx, params.skyny, rnd(prd->seed), envpdf, skyuv, prd->rndf(), prd->rndf())
                                    : BRDFBasics::halfPlaneSample(prd->seed, sunLightDir,
                                                    params.sunSoftness * 0.0f);
            float samplePDF;
            float3 illum = sampleSkyTexture(skyuv, 100, 0, samplePDF);
            samplePDF *= 1.0f;
            if(samplePDF <= 0.0f) { return; }

            shadeTask(sample_dir, samplePDF, illum/( _SKY_PROB_ * elight_pr), true);
            return;
        }
    }
};