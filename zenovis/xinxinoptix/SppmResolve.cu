#include <cuda/helpers.h>
#include "TraceStuff.h"
#include "CameraPostProcess.h"
#include "OSD.h"
extern "C" { __constant__ Params params; }
#include "RestirFg.h"

static __forceinline__ __device__ void fgMarkQuery(const FgSurface& s) {
    if (!s.valid) return;
    const float r=params.sppm.initialRadius, cellSize=2*r;
    // Conservative sphere AABB, including boundary roundoff. Hash collisions
    // keep extra photons; they can never discard a contributing one.
    const int3 lo=sppmCell(make_float3(nextafterf(s.position.x-r,-INFINITY),
        nextafterf(s.position.y-r,-INFINITY),nextafterf(s.position.z-r,-INFINITY)),cellSize);
    const int3 hi=sppmCell(make_float3(nextafterf(s.position.x+r,INFINITY),
        nextafterf(s.position.y+r,INFINITY),nextafterf(s.position.z+r,INFINITY)),cellSize);
    for(int z=lo.z;z<=hi.z;++z) for(int y=lo.y;y<=hi.y;++y) for(int x=lo.x;x<=hi.x;++x)
        atomicExch(&params.sppm.heads[sppmHash(make_int3(x,y,z),params.sppm.hashSize)],0u);
}

extern "C" __global__ void __raygen__sppm_grid() {
    const uint3 index = optixGetLaunchIndex();
    const uint32_t i = index.y*params.width + index.x;
    auto& pixel = params.sppm.pixels[i];
    if (params.sppm.fgEnabled) {
        if (!params.sppm.fgCull) return;
        fgMarkQuery(params.sppm.fgInitial[i].source);
        fgMarkQuery(params.sppm.fgCandidates[i].point);
        return;
    }
    // Pixel-radius calibration completes after camera tracing, before this pass.
    if (params.sppm.iteration == 0u) pixel.radius = params.sppm.initialRadius;
    if (params.sppm.searchBackend) {
        auto& box = params.sppm.searchBounds[i];
        box.record = params.sppm.searchRecord;
        if (pixel.visible) {
            // Round outward so broad-phase bounds cannot reject a boundary
            // accepted by the shared floating-point spherical distance test.
            box.minX = nextafterf(pixel.position.x-pixel.radius, -INFINITY);
            box.minY = nextafterf(pixel.position.y-pixel.radius, -INFINITY);
            box.minZ = nextafterf(pixel.position.z-pixel.radius, -INFINITY);
            box.maxX = nextafterf(pixel.position.x+pixel.radius, INFINITY);
            box.maxY = nextafterf(pixel.position.y+pixel.radius, INFINITY);
            box.maxZ = nextafterf(pixel.position.z+pixel.radius, INFINITY);
        } else {
            // NaN bounds mark inactive custom primitives in an OptiX build.
            box.minX = box.minY = box.minZ = box.maxX = box.maxY = box.maxZ = __int_as_float(0x7fffffff);
        }
        return;
    }
    if (!pixel.visible) return;
    pixel.cell = sppmCell(pixel.position, params.sppm.initialRadius);
    const uint32_t bucket = sppmHash(pixel.cell, params.sppm.hashSize);
    pixel.next = atomicExch(&params.sppm.heads[bucket], i);
}

extern "C" __global__ void __raygen__sppm_resolve() {
    const auto index = optixGetLaunchIndex();
    const uint32_t i = index.y*params.width + index.x;
    auto& pixel = params.sppm.pixels[i];
    if (params.sppm.fgEnabled) pixel.radius=params.sppm.initialRadius;
    if (!params.sppm.fgEnabled && pixel.m > 0u) {
        const float n = pixel.n + params.sppm.alpha*float(pixel.m);
        const float ratio = n / (pixel.n + float(pixel.m));
        pixel.radius *= sqrtf(ratio);
        pixel.tau = (pixel.tau + pixel.phi) * ratio;
        pixel.n = n;
    }
    const float iterations = float(params.sppm.iteration)+1.0f;
    const double emitted = double(params.sppm.photonsPerIteration) * double(params.sppm.iteration+1ull);
    float3 color = pixel.direct / iterations;
    if (params.sppm.fgEnabled) color += pixel.tau / iterations;
    else if (emitted > 0.0)
        color += pixel.tau * float(1.0 / (emitted * double(M_PIf) * pixel.radius * pixel.radius));
    const auto& cam = params.cam;
    if (params.physical_camera_exposure)
        color = PhysicalCamera(color, cam.aperture < 0.0001f ? params.physical_camera_aperture : cam.aperture,
            params.physical_camera_shutter_speed, params.physical_camera_iso, 0.18f, true, false);
    params.accum_buffer[i] = color;
    if (params.physical_camera_aces) color = ACESFitted(color);
    const float2 uv = make_float2(index.x+0.5f, index.y+0.5f);
    auto& display = params.frame_buffer[i];
    display = makeSRGB(color, 2.2f, InterleavedGradientNoise(uv)-0.5f);
    if (params.frame_time > 0)
        drawOSD(reinterpret_cast<uchar3*>(&display), params.frame_time, uv, 20);
}
