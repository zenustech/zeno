#include <optix.h>
#include "TraceStuff.h"

extern "C" __global__ void __anyhit__shadow_cutout()
{
    RadiancePRD* prd = getPRD();
    prd->shadowAttanuation = {};
    prd->attenuation = {};
    optixTerminateRay();
    return;
}

extern "C" __global__ void __closesthit__radiance()
{
    RadiancePRD* prd = getPRD();
    prd->done = true;
    prd->depth += 1;
    return;
}

extern "C" __global__ void __closesthit__occlusion()
{
    setPayloadOcclusion( true );
}
