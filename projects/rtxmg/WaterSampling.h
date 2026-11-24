#pragma once
#include "WaterDevice.h"

namespace zeno_water_device {
__device__ __forceinline__ float3 transformPoint(const float* m,float3 p) {
    return make_float3(m[0]*p.x+m[1]*p.y+m[2]*p.z+m[3],m[4]*p.x+m[5]*p.y+m[6]*p.z+m[7],m[8]*p.x+m[9]*p.y+m[10]*p.z+m[11]);
}
__device__ __forceinline__ float3 transformVector(const float* m,float3 p) {
    return make_float3(m[0]*p.x+m[1]*p.y+m[2]*p.z,m[4]*p.x+m[5]*p.y+m[6]*p.z,m[8]*p.x+m[9]*p.y+m[10]*p.z);
}
__device__ __forceinline__ float3 transformNormal(const float* inverse,float3 n) {
    return make_float3(inverse[0]*n.x+inverse[4]*n.y+inverse[8]*n.z,
        inverse[1]*n.x+inverse[5]*n.y+inverse[9]*n.z,inverse[2]*n.x+inverse[6]*n.y+inverse[10]*n.z);
}
struct Sample { float height, dx, dz; };
__device__ __forceinline__ void cubicWeights(float t,float* w) {
    const float t2=t*t,t3=t2*t;
    w[0]=-.5f*t+t2-.5f*t3;w[1]=1.f-2.5f*t2+1.5f*t3;
    w[2]=.5f*t+2.f*t2-1.5f*t3;w[3]=-.5f*t2+.5f*t3;
}
__device__ __forceinline__ Sample sample(const Surface& s,float x,float z) {
    if(!s.field || s.amplitude==0.f)return {0,0,0};
    const float u=(x-s.center[0])/s.period*s.resolution,v=(z-s.center[2])/s.period*s.resolution;
    const int ix=int(floorf(u)),iy=int(floorf(v)),mask=int(s.resolution)-1;
    float wx[4],wy[4];cubicWeights(u-floorf(u),wx);cubicWeights(v-floorf(v),wy);
    Sample out{};
    for(int j=0;j<4;++j)for(int i=0;i<4;++i) {
        const auto f=s.field[((iy+j-1)&mask)*s.resolution+((ix+i-1)&mask)];
        const float w=wx[i]*wy[j];out.height+=w*f.x;out.dx+=w*f.y;out.dz+=w*f.z;
    }
    if(s.boundaryWidth>0.f) {
        const float px=(x-s.center[0])/(s.size[0]*.5f),pz=(z-s.center[2])/(s.size[1]*.5f);
        float distance,dx=0,dz=0;
        if(s.ellipse) {
            const float r=sqrtf(px*px+pz*pz),scale=fminf(s.size[0],s.size[1])*.5f;
            distance=(1.f-r)*scale;
            if(r>1.e-8f){dx=-scale*px/(r*s.size[0]*.5f);dz=-scale*pz/(r*s.size[1]*.5f);}
        } else {
            const float a=(1.f-fabsf(px))*s.size[0]*.5f,b=(1.f-fabsf(pz))*s.size[1]*.5f;
            distance=fminf(a,b);
            if(a<b)dx=px>=0?-1.f:1.f;else dz=pz>=0?-1.f:1.f;
        }
        const float t=fminf(1.f,fmaxf(0.f,distance/s.boundaryWidth));
        const float fade=t*t*(3.f-2.f*t),derivative=(t>0.f&&t<1.f)?6.f*t*(1.f-t)/s.boundaryWidth:0.f;
        out.dx=out.dx*fade+out.height*derivative*dx;out.dz=out.dz*fade+out.height*derivative*dz;out.height*=fade;
    }
    return out;
}
__device__ __forceinline__ float2 domain(const Patch& p,unsigned int i,unsigned int j) {
    // A balanced neighbor is at most twice as coarse. Collapsing alternate
    // boundary samples gives both sides the exact same positions and heights.
    if((j==0 && (p.coarseEdges&1u)) || (j==GridSegments && (p.coarseEdges&4u)))i=((i+1)/2)*2;
    if((i==GridSegments && (p.coarseEdges&2u)) || (i==0 && (p.coarseEdges&8u)))j=((j+1)/2)*2;
    const float scale=ldexpf(1.f,-int(p.depth))/GridSegments;
    return make_float2(float(p.x*GridSegments+i)*scale,float(p.y*GridSegments+j)*scale);
}
__device__ __forceinline__ float3 position(const Surface& s,const Patch& p,unsigned int i,unsigned int j) {
    if(s.flipWinding){const auto swap=i;i=j;j=swap;}
    const auto uv=domain(p,i,j);
    float x=2.f*uv.x-1.f,z=1.f-2.f*uv.y;
    if(s.ellipse){const float a=x;x*=sqrtf(fmaxf(0.f,1.f-z*z*.5f));z*=sqrtf(fmaxf(0.f,1.f-a*a*.5f));}
    x=s.center[0]+x*s.size[0]*.5f;z=s.center[2]+z*s.size[1]*.5f;
    const auto point=make_float3(x,s.center[1]+sample(s,x,z).height,z);
    return s.transformed?transformPoint(s.objectToGeometry,point):point;
}
}
