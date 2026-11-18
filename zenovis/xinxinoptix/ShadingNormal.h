#pragma once

// Requires the CUDA/sutil float3 vector operations used by the material shader.
namespace shadingNormal {

// Lift the ideal reflection above the actual triangle, keeping the normal's
// azimuth and changing its tilt only as much as necessary. In the plane of
// N and Ng, R.Ng = Iz*cos(2*theta) + Ix*sin(2*theta). Solve its boundary with
// half-angle identities. This also prevents an exterior view from ending up
// behind a strongly perturbed normal. Inputs are unit vectors, Ng faces V.
__host__ __device__ inline float3 validReflection(const float3& ng, const float3& v, const float3& n)
{
    const float iz=dot(ng,v);
    const float nz=dot(ng,n);
    if(!(iz>0.f))return n;
    if(!(nz>0.f))return ng;
    const float threshold=fminf(.9f*iz,.01f);
    if(2.f*dot(n,v)*nz-iz>=threshold)return n;

    const float3 tangent=n-nz*ng;
    const float length2=dot(tangent,tangent);
    if(!(length2>1.e-12f))return ng;
    const float3 x=tangent/sqrtf(length2);
    const float ix=dot(v,x);
    const float a=ix*ix+iz*iz;
    if(!(a>1.e-20f))return ng;
    const float q=sqrtf(fmaxf(0.f,a-threshold*threshold));
    // Rationalized roots avoid subtracting nearly equal values when the
    // incident direction is almost parallel to the triangle.
    float nx,nzCorrected;
    if(ix>=0.f) {
        const float z2=fminf(1.f,fmaxf(0.f,(iz+threshold)*(iz+threshold)/
            (2.f*(a+iz*threshold+ix*q))));
        nzCorrected=sqrtf(z2);nx=sqrtf(fmaxf(0.f,1.f-z2));
    } else {
        const float x2=fminf(1.f,fmaxf(0.f,(iz-threshold)*(iz-threshold)/
            (2.f*(a-iz*threshold-ix*q))));
        nx=sqrtf(x2);nzCorrected=sqrtf(fmaxf(0.f,1.f-x2));
    }
    return nx*x+nzCorrected*ng;
}

} // namespace shadingNormal
