#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>

namespace zeno::cluster {
// Match NVIDIA's 22-bit octahedral-normal / 10-bit tangent-angle convention.
inline void tangentBasis(const float* n,float* t,float* b) {
    if(n[2]<-.99998796f) {t[0]=0;t[1]=-1;t[2]=0;b[0]=-1;b[1]=0;b[2]=0;return;}
    const float a=1.f/(1.f+n[2]),v=-n[0]*n[1]*a;
    t[0]=1-n[0]*n[0]*a;t[1]=v;t[2]=-n[0];
    b[0]=v;b[1]=1-n[1]*n[1]*a;b[2]=-n[1];
}
inline uint32_t packTangent(const float* n,const float* tangent) {
    float t[3],b[3];tangentBasis(n,t,b);
    float dt=0,db=0;for(int i=0;i<3;++i){dt+=t[i]*tangent[i];db+=b[i]*tangent[i];}
    const float angle=std::atan2(dt,db)/3.14159265358979323846f;
    const uint32_t bits=uint32_t(std::clamp((angle+1)*.5f,0.f,1.f)*511.f+.5f);
    return 0x80000000u|(bits<<1)|(tangent[3]>0?1u:0u);
}
}
