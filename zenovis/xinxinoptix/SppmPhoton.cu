#include <cuda/helpers.h>
#include "SppmDevice.h"
extern "C" { __constant__ Params params; }

static __forceinline__ __device__ void sppmAddGuideScore(double* address, double value) {
    auto* bits = reinterpret_cast<unsigned long long*>(address);
    unsigned long long old = *bits, assumed;
    do {
        assumed = old;
        old = atomicCAS(bits, assumed, __double_as_longlong(value + __longlong_as_double(assumed)));
    } while (old != assumed);
}

// The mapping acts on uniform emission coordinates, so the existing geometric
// PDFs/Jacobians remain valid. A joint emitter/bin PDF supplies the extra weight.
static __forceinline__ __device__ void sppmGuideUV(RadiancePRD& prd, float& u, float& v) {
    const auto& launch = params.sppm;
    if (!launch.guideCells) return;
    if (launch.guideActive) {
        const uint32_t bin = prd.sppmGuideBin % SppmGuideBins;
        u = (float(bin % SppmGuideSide) + u) / float(SppmGuideSide);
        v = (float(bin / SppmGuideSide) + v) / float(SppmGuideSide);
    } else {
        const uint32_t x = min(uint32_t(u*SppmGuideSide), SppmGuideSide-1u);
        const uint32_t y = min(uint32_t(v*SppmGuideSide), SppmGuideSide-1u);
        prd.sppmGuideBin += y*SppmGuideSide+x;
    }
}

static __forceinline__ __device__ float3 sppmGuidedSphere(RadiancePRD& prd) {
    float u = rnd(prd.seed), v = rnd(prd.seed);
    sppmGuideUV(prd, u, v);
    const float z = 1.0f-2.0f*u, phi = 2.0f*M_PIf*v;
    const float r = sqrtf(fmaxf(0.0f, 1.0f-z*z));
    return make_float3(r*cosf(phi), r*sinf(phi), z);
}

static __forceinline__ __device__ float3 sppmSphere(uint32_t& seed) {
    const float z = 1.0f - 2.0f * rnd(seed);
    const float phi = 2.0f * M_PIf * rnd(seed);
    const float r = sqrtf(fmaxf(0.0f, 1.0f - z*z));
    return make_float3(r*cosf(phi), r*sinf(phi), z);
}

static __forceinline__ __device__ float3 sppmOrient(float3 v, float3 n) {
    Onb frame(n);
    return float3(frame.m_tangent) * v.x + float3(frame.m_binormal) * v.y + n * v.z;
}

static __forceinline__ __device__ void sppmInfiniteOrigin(
    RadiancePRD& prd, float3 direction, float3& origin, bool guide)
{
    const float radius = params.sppm.sceneRadius;
    float u = rnd(prd.seed), v = rnd(prd.seed);
    if (guide) sppmGuideUV(prd, u, v);
    const float r = radius * sqrtf(u), phi = 2.0f*M_PIf*v;
    origin = params.sppm.sceneCenter - direction * (radius * 1.001f) +
        sppmOrient(make_float3(r*cosf(phi), r*sinf(phi), 0.0f), direction);
}

static __forceinline__ __device__ bool sppmEmit(
    RadiancePRD& prd, float3& origin, float3& direction, float3& power)
{
    const auto& launch = params.sppm;
    if (launch.emitterCount == 0u) return false;
    const float select = rnd(prd.seed);
    uint32_t emitter = min(uint32_t(select * launch.emitterCount), launch.emitterCount - 1u);
    float selectionWeight = float(launch.emitterCount);
    if (launch.guideActive) {
        prd.sppmGuideBin = sppmGuideSelect(launch.guideCdf, launch.guideCells, select);
        emitter = prd.sppmGuideBin / SppmGuideBins;
        selectionWeight = sppmGuideWeight(launch.guideCdf, prd.sppmGuideBin);
    } else if (launch.guideCells) prd.sppmGuideBin = emitter * SppmGuideBins;
    prd.sppmGuideWeight = selectionWeight;
    if (emitter >= params.num_lights) {
        const uint32_t index = emitter - params.num_lights;
        if (index < launch.distantCount) {
            const auto* distant = reinterpret_cast<const DistantLightList*>(params.dlights_ptr);
            const auto& light = distant->list[index];
            const auto n = reinterpret_cast<const vec3&>(light.direction);
            // DistantLightData intensity is irradiance, matching its NEE path.
            direction = -float3(BRDFBasics::halfPlaneSample(prd.seed, n, light.angle / 180.0f));
            power = float3(light.color) * light.intensity;
        } else {
            const float3 outgoing = sppmGuidedSphere(prd);
            float pdf = 0.0f;
            power = float3(sampleSkyTexture(vec3(outgoing), 100.0f, 0.0f, pdf)) * (4.0f*M_PIf);
            direction = -outgoing;
        }
        sppmInfiniteOrigin(prd, direction, origin, index < launch.distantCount);
        power *= selectionWeight * M_PIf * launch.sceneRadius * launch.sceneRadius;
        return sppmFinite(power) && sppmMax(power) > 0.0f;
    }

    prd.sppmLight = emitter;
    auto& light = params.lights[emitter];
    LightSampleRecord sample = {};
    float area = 1.0f, directionPdf = 1.0f, cosine = 1.0f, profile = 1.0f;
    float3 normal = normalize(light.N);
    float u = rnd(prd.seed), v = rnd(prd.seed);
    const bool ies = light.type == zeno::LightType::IES && light.ies != 0u;
    const bool point = light.shape == zeno::LightShape::Point ||
        light.type == zeno::LightType::Spot || light.type == zeno::LightType::Projector ||
        ies;
    if (!point && light.type == zeno::LightType::Direction && light.shape != zeno::LightShape::Sphere)
        sppmGuideUV(prd, u, v);
    if (point) {
        sample.p = light.point.p;
    } else if (light.shape == zeno::LightShape::Plane || light.shape == zeno::LightShape::Ellipse) {
        const auto& rect = light.rect;
        normal = rect.normal;
        float2 uv = make_float2(u, v);
        area = rect.area;
        if (light.shape == zeno::LightShape::Ellipse) {
            const float r = sqrtf(u), phi = 2.0f*M_PIf*v;
            uv = make_float2(0.5f + 0.5f*r*cosf(phi), 0.5f + 0.5f*r*sinf(phi));
            area *= 0.25f*M_PIf;
        }
        sample.p = rect.v + rect.axisX * (uv.x * rect.lenX) + rect.axisY * (uv.y * rect.lenY);
        sample.uv = uv;
    } else if (light.shape == zeno::LightShape::Sphere) {
        normal = light.type == zeno::LightType::Direction && !point ? sppmGuidedSphere(prd) : sppmSphere(prd.seed);
        sample.p = light.sphere.center + light.sphere.radius * normal;
        area = light.sphere.area;
        sample.n = normal;
        cihouSphereLightUV(sample, light);
    } else if (light.shape == zeno::LightShape::TriangleMesh) {
        const auto& triangle = light.triangle;
        const float root = sqrtf(u);
        const float3 bary = make_float3(1.0f-root, root*(1.0f-v), root*v);
        sample.p = triangle.p0*bary.x + triangle.p1*bary.y + triangle.p2*bary.z;
        normal = triangle.faceNormal;
        area = triangle.area;
        if (triangle.coordsBufferOffset != UINT_MAX && params.triangleLightCoordsBuffer) {
            const auto* uv = reinterpret_cast<const float2*>(params.triangleLightCoordsBuffer) + triangle.coordsBufferOffset;
            sample.uv = uv[0]*bary.x + uv[1]*bary.y + uv[2]*bary.z;
        }
    } else return false;
    if (!(area > 0.0f)) return false;

    if (light.type == zeno::LightType::Spot) {
        sppmGuideUV(prd, u, v);
        const float z = 1.0f-u*(1.0f-light.cone.cosFalloffEnd);
        const float phi = 2.0f*M_PIf*v, r = sqrtf(fmaxf(0.0f, 1.0f-z*z));
        direction = sppmOrient(make_float3(r*cosf(phi), r*sinf(phi), z), normalize(light.N));
        directionPdf = 1.0f/(2.0f*M_PIf*(1.0f-light.cone.cosFalloffEnd));
        profile = smoothstep(light.cone.cosFalloffEnd, light.cone.cosFalloffStart, z);
        const float fov = tanf(light.spreadMajor*0.5f*M_PIf);
        const float tu = dot(direction, light.T)/z, tv = dot(direction, light.B)/z;
        if (fabsf(tu)>fov || fabsf(tv)>fov) return false;
        sample.uv = make_float2(0.5f+0.5f*tu/fov, 0.5f+0.5f*tv/fov);
    } else if (light.type == zeno::LightType::Projector) {
        sppmGuideUV(prd, u, v);
        const float su = clamp(light.spreadMajor, 0.001f, 0.999f);
        const float sv = light.spreadMinor > 0.0f ? clamp(light.spreadMinor, 0.001f, 0.999f) : su;
        const float tu = tanf(su*0.5f*M_PIf), tv = tanf(sv*0.5f*M_PIf);
        direction = normalize(light.N + light.T*((2*u-1)*tu) + light.B*((2*v-1)*tv));
        const float z = dot(direction, light.N);
        directionPdf = 1.0f/(4.0f*tu*tv*z*z*z);
        profile = 1.0f/(z*z);
        sample.uv = make_float2(u, v);
    } else if (ies) {
        direction = sppmGuidedSphere(prd);
        directionPdf = 1.0f/(4.0f*M_PIf);
        profile = sampleIES(reinterpret_cast<const float*>(light.ies),
            acosf(clamp(dot(direction, light.T), -1.0f, 1.0f)),
            acosf(clamp(dot(direction, light.N), -1.0f, 1.0f)));
        if (light.shape == zeno::LightShape::Sphere)
            sample.p += light.sphere.radius * direction;
    } else if (point) {
        direction = sppmGuidedSphere(prd);
        directionPdf = 1.0f/(4.0f*M_PIf);
    } else if (light.type == zeno::LightType::Direction) {
        if (light.shape == zeno::LightShape::TriangleMesh) return false;
        direction = normal;
        profile = 2.0f*M_PIf;
        // Spherical direction lights use radial intensity measured from
        // the surface. They do not use Lambertian area-light emission.
        if (light.shape == zeno::LightShape::Sphere) area = 4.0f*M_PIf;
    } else {
        const bool twoSided = (light.config & zeno::LightConfigDoubleside) != 0;
        if (twoSided && rnd(prd.seed)<0.5f) normal = -normal;
        vec3 local;
        float du = rnd(prd.seed), dv = rnd(prd.seed);
        sppmGuideUV(prd, du, dv);
        cosine_sample_hemisphere(du, dv, local);
        direction = sppmOrient(float3(local), normal);
        cosine = fmaxf(0.0f, dot(direction, normal));
        directionPdf = cosine / (M_PIf * (twoSided ? 2.0f : 1.0f));
        if (light.spreadMajor < 1.0f && light.spreadNormalize != 0.0f)
            profile = light_spread_attenuation(cosine, light.spreadMajor, light.spreadNormalize);
    }
    if (!(directionPdf > 0.0f)) return false;
    const auto rgba = cihouLightEmission(sample, light, 1u);
    power = make_float3(rgba.x, rgba.y, rgba.z) *
        (profile * cosine * area * selectionWeight / directionPdf);
    origin = sample.p - params.cam.eye;
    // Keep emitter geometry in traversal: inward photons from a two-sided
    // sphere must be absorbed by its opposite surface rather than leak out.
    if (ies && light.shape == zeno::LightShape::Sphere)
        origin = rtgems::offset_ray(origin, direction);
    else if (light.shape != zeno::LightShape::Point &&
             (light.shape != zeno::LightShape::Sphere || !point))
        origin = rtgems::offset_ray(origin, dot(direction, normal) >= 0.0f ? normal : -normal);
    return sppmFinite(power) && sppmMax(power)>0.0f && sppmFinite(direction);
}

// Clip paths inside their spherical emitter to its exit. Light proxies
// serve exterior visibility; inward emission must not escape the closed shell.
// The miss program handles this boundary and its texture cutout explicitly.
static __forceinline__ __device__ float sppmSphereExit(
    const RadiancePRD& prd, float3 origin, float3 direction)
{
    if (prd.sppmLight >= params.num_lights) return FLT_MAX;
    const auto& light = params.lights[prd.sppmLight];
    if (light.type != zeno::LightType::Diffuse || light.shape != zeno::LightShape::Sphere)
        return FLT_MAX;
    const float3 relative = origin - (light.sphere.center - params.cam.eye);
    const float c = dot(relative, relative) - light.sphere.radius * light.sphere.radius;
    if (!(c < 0.0f)) return FLT_MAX;
    const float b = dot(relative, direction);
    return nextafterf(-b + sqrtf(fmaxf(0.0f, b*b-c)), 0.0f);
}

extern "C" __global__ void __raygen__sppm_photon() {
    const auto index = optixGetLaunchIndex();
    const uint32_t i = index.x + index.y * optixGetLaunchDimensions().x;
    if (i >= params.sppm.photonsPerIteration) return;
    RadiancePRD prd = {};
    prd.sppmMode = 2u;
    prd.sppmLight = UINT_MAX;
    prd.seed = sppmPhotonSeed(i, params.sppm.iteration, params.outside_random_number);
    prd.eventseed = pcg_hash(prd.seed);
    prd.vdcseed = pcg_hash(prd.eventseed);
    prd.offset = params.sppm.iteration;
    prd.offset2 = prd.offset3 = prd.offset;
    prd.max_depth = uint8_t(params.max_bounce);
    prd.minSpecRough = 0.0f;
    prd.ss_alpha_queue[0] = half3(-1.0f);
    float3 origin, direction, power;
    atomicAdd(&params.sppm.statistics->emitted, 1ull);
    if (!sppmEmit(prd, origin, direction, power)) return;
    prd.attenuation = power;
    prd.maxDistance = FLT_MAX;
    const uint8_t mask = EverythingMask ^ VolumeMatMask;
    unsigned long long traced = 0;
    for (uint32_t depth = 0; depth < params.max_bounce && !prd.done; ++depth) {
        prd.origin = origin;
        prd.direction = direction;
        prd.sppmIncoming = prd.attenuation;
        prd.sppmGatherPending = 0u;
        float tmin = 0.0f;
        prd.maxDistance = sppmSphereExit(prd, origin, direction);
        do {
            prd.alphaHit = false;
            prd._tmin_ = 0.0f;
            traceRadiance(params.handle, origin, direction, tmin, prd.maxDistance, &prd, mask);
            tmin = prd._tmin_;
        } while (prd.alphaHit && !prd.done);
        if (prd.sppmGatherPending) {
            sppmGather(prd.sppmGatherPosition, prd.sppmGatherNormal,
                prd.sppmGatherWi, prd.sppmGatherBeta, prd);
        }
        ++traced;
        if (prd.done || !sppmFinite(prd.attenuation)) break;
        const float before = sppmMax(prd.sppmIncoming);
        const float survival = fminf(1.0f, sppmMax(prd.attenuation) / fmaxf(before, 1.0e-30f));
        if (!(survival>0.0f) || rnd(prd.seed)>=survival) break;
        prd.attenuation /= survival;
        origin = prd.origin;
        direction = prd.direction;
    }
    atomicAdd(&params.sppm.statistics->traced, traced);
    if (prd.sppmGathered) atomicAdd(&params.sppm.statistics->gathered, prd.sppmGathered);
    if (prd.sppmCaustics) atomicAdd(&params.sppm.statistics->caustic, prd.sppmCaustics);
    if (prd.sppmGuideUseful) {
        atomicAdd(&params.sppm.statistics->usefulPhotonPaths, 1ull);
    }
    if (params.sppm.guideCells && prd.sppmGuideBin < params.sppm.guideCells &&
        prd.sppmGuideWeight > 0.0f && prd.sppmGuideScore > 0.0)
        sppmAddGuideScore(&params.sppm.guideScores[prd.sppmGuideBin], prd.sppmGuideScore / double(prd.sppmGuideWeight));
}


extern "C" __global__ void __miss__sppm_photon() {
    auto* prd = getPRD();
    if (prd->maxDistance < FLT_MAX) {
        auto& light = params.lights[prd->sppmLight];
        LightSampleRecord sample = {};
        sample.n = normalize(prd->origin + prd->direction * prd->maxDistance -
                             (light.sphere.center - params.cam.eye));
        cihouSphereLightUV(sample, light);
        const auto emission = cihouLightEmission(sample, light, 1u);
        if (emission.w < 1.0f && rnd(prd->seed) >= emission.w) {
            // Continue beyond the boundary in the same alpha traversal loop.
            prd->_tmin_ = nextafterf(nextafterf(prd->maxDistance, FLT_MAX), FLT_MAX);
            prd->maxDistance = FLT_MAX;
            prd->alphaHit = true;
            return;
        }
        atomicAdd(&params.sppm.statistics->absorbed, 1ull);
    }
    prd->done = true;
    prd->radiance = make_float3(0.0f);
}

// Hardware broad phase over receiver spheres (the transpose of photon gathering).
extern "C" __global__ void __intersection__sppm_search() {
    const auto& pixel = params.sppm.pixels[optixGetPrimitiveIndex()];
    if (pixel.visible && lengthSquared(pixel.position-optixGetWorldRayOrigin()) <= pixel.radius*pixel.radius)
        optixReportIntersection(0.0f, 0u);
}
extern "C" __global__ void __anyhit__sppm_search() {
    auto* query = reinterpret_cast<SppmGatherQuery*>(unpackPointer(optixGetPayload_0(), optixGetPayload_1()));
    sppmAccumulate(optixGetPrimitiveIndex(), query->position, query->normal, query->wi, query->beta, *query->photon);
    optixIgnoreIntersection();
}
extern "C" __global__ void __miss__sppm_search() {}
