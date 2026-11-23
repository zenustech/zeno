#pragma once
#include "WaterDevice.h"

namespace zeno::water {
struct TessellationInput {
    float center[3],size[2],eye[3],projection,amplitude,edgePixels;
    float right[3],up[3],front[3],tanX,tanY;
    unsigned int ellipse,capacity,offset,hashOffset,hashSize;
};
struct TessellationStats {
    unsigned int clusters,minDepth,maxDepth,limited;
    float effectivePixels;
};
namespace tessellation {
using zeno_water_device::Patch;
using zeno_water_device::MaxDepth;
using Key=unsigned long long;
__device__ Key key(unsigned int x,unsigned int y,unsigned int depth) {
    return (Key(depth)<<56 | Key(x)<<28 | y)+1;
}
__device__ unsigned int hash(Key k,unsigned int mask) {
    k^=k>>33;k*=0xff51afd7ed558ccdULL;k^=k>>33;
    return unsigned(k)&mask;
}
__device__ unsigned int neighbor(Patch p,int direction,const Key* keys,const unsigned int* values,unsigned int mask) {
    const int lattice=1<<MaxDepth,span=lattice>>p.depth,x=int(p.x)*span,y=int(p.y)*span;
    int qx=x+span/2,qy=y+span/2;
    if(direction==0)qy=y-1;else if(direction==1)qx=x+span;
    else if(direction==2)qy=y+span;else qx=x-1;
    if(qx<0 || qy<0 || qx>=lattice || qy>=lattice)return ~0u;
    for(int depth=int(p.depth);depth>=0;--depth) {
        const auto k=key(unsigned(qx)>>(MaxDepth-depth),unsigned(qy)>>(MaxDepth-depth),unsigned(depth));
        auto slot=hash(k,mask);
        while(keys[slot] && keys[slot]!=k)slot=(slot+1)&mask;
        if(keys[slot]==k)return values[slot];
    }
    return ~0u;
}
__device__ float2 ellipseBounds(float lo,float hi,float a,float b) {
    const float closest=a<=0 && b>=0?0.f:fminf(fabsf(a),fabsf(b)),farthest=fmaxf(fabsf(a),fabsf(b));
    const float small=sqrtf(fmaxf(0.f,1.f-farthest*farthest*.5f)),large=sqrtf(fmaxf(0.f,1.f-closest*closest*.5f));
    return make_float2(lo*(lo<0?large:small),hi*(hi>0?large:small));
}
__device__ float projected(const TessellationInput& s,Patch p) {
    const float span=ldexpf(1.f,-int(p.depth));
    float2 x=make_float2(2.f*p.x*span-1.f,2.f*(p.x+1)*span-1.f);
    float2 z=make_float2(1.f-2.f*(p.y+1)*span,1.f-2.f*p.y*span);
    if(s.ellipse){const auto a=ellipseBounds(x.x,x.y,z.x,z.y);z=ellipseBounds(z.x,z.y,x.x,x.y);x=a;}
    x=make_float2(s.center[0]+x.x*s.size[0]*.5f,s.center[0]+x.y*s.size[0]*.5f);
    z=make_float2(s.center[2]+z.x*s.size[1]*.5f,s.center[2]+z.y*s.size[1]*.5f);
    const float dx=fmaxf(0.f,fmaxf(x.x-s.eye[0],s.eye[0]-x.y));
    const float dz=fmaxf(0.f,fmaxf(z.x-s.eye[2],s.eye[2]-z.y));
    const float dy=fmaxf(0.f,fabsf(s.eye[1]-s.center[1])-8.f*s.amplitude);
    const float distance=fmaxf(.01f,sqrtf(dx*dx+dy*dy+dz*dz));
    bool visible=true;
    if(s.tanX>0 && s.tanY>0) {
        const float3 center=make_float3((x.x+x.y)*.5f-s.eye[0],s.center[1]-s.eye[1],(z.x+z.y)*.5f-s.eye[2]);
        const float3 extent=make_float3((x.y-x.x)*.5f,8.f*s.amplitude,(z.y-z.x)*.5f);
        for(unsigned int plane=0;plane<5;++plane) {
            float n[3];
            for(unsigned int axis=0;axis<3;++axis) {
                n[axis]=s.front[axis];
                if(plane<2)n[axis]=n[axis]*s.tanX+s.right[axis]*(plane==0?1.f:-1.f);
                else if(plane<4)n[axis]=n[axis]*s.tanY+s.up[axis]*(plane==2?1.f:-1.f);
            }
            const float upper=n[0]*center.x+n[1]*center.y+n[2]*center.z+
                fabsf(n[0])*extent.x+fabsf(n[1])*extent.y+fabsf(n[2])*extent.z;
            if(upper<0){visible=false;break;}
        }
    }
    if(visible && s.tanX>0 && s.tanY>0) {
        const float stepX=(x.y-x.x)/zeno_water_device::GridSegments,stepZ=(z.y-z.x)/zeno_water_device::GridSegments;
        float rate=0;
        for(unsigned int corner=0;corner<8;++corner) {
            const float p[3]{(corner&1?x.y:x.x)-s.eye[0],s.center[1]+(corner&4?8.f:-8.f)*s.amplitude-s.eye[1],(corner&2?z.y:z.x)-s.eye[2]};
            float depth=0,right=0,up=0;
            for(unsigned int axis=0;axis<3;++axis){depth+=p[axis]*s.front[axis];right+=p[axis]*s.right[axis];up+=p[axis]*s.up[axis];}
            depth=fmaxf(depth,.01f);
            for(unsigned int axis=0;axis<3;axis+=2) {
                const float dr=s.right[axis]*depth-right*s.front[axis],du=s.up[axis]*depth-up*s.front[axis];
                rate=fmaxf(rate,s.projection*(axis==0?stepX:stepZ)*sqrtf(dr*dr+du*du)/(depth*depth));
            }
        }
        return 1.25f*rate;
    }
    // Complete coverage also retains water for reflection and refraction rays.
    return 1.5f*fmaxf(s.size[0],s.size[1])*span*s.projection/(distance*zeno_water_device::GridSegments*(visible?1.f:16.f));
}

// One cooperative block owns each bounded surface queue. Selection and 2:1
// balancing stay on the GPU; indirect counts feed the subsequent OptiX build.
__global__ void select(const TessellationInput* inputs,Patch* workA,Patch* workB,Patch* cut,
                       Key* hashKeys,unsigned int* hashValues,unsigned int* split,TessellationStats* statistics) {
    const unsigned int surface=blockIdx.x;const auto s=inputs[surface];
    auto* a=workA+s.offset;auto* b=workB+s.offset;cut+=s.offset;split+=s.offset;
    auto* keys=hashKeys+s.hashOffset;auto* values=hashValues+s.hashOffset;const auto mask=s.hashSize-1;
    __shared__ unsigned int count,nextCount,overflow,changes,minDepth,maxDepth;
    float pixels=s.edgePixels;
    for(unsigned int retry=0;retry<64;++retry,pixels*=1.5f) {
        a=workA+s.offset;b=workB+s.offset;
        if(!threadIdx.x){a[0]={0,0,0,0,surface};count=1;overflow=0;}
        __syncthreads();
        for(unsigned int level=0;level<MaxDepth;++level) {
            if(!threadIdx.x){nextCount=0;changes=0;}
            __syncthreads();
            for(unsigned int i=threadIdx.x;i<count;i+=blockDim.x) {
                const auto p=a[i];const bool refine=p.depth<MaxDepth && projected(s,p)>pixels;
                const auto n=refine?4u:1u,j=atomicAdd(&nextCount,n);
                if(j+n>s.capacity){atomicExch(&overflow,1u);continue;}
                if(refine) {
                    atomicAdd(&changes,1u);
                    for(unsigned int k=0;k<4;++k)b[j+k]={p.x*2+(k&1),p.y*2+(k>>1),p.depth+1,0,surface};
                } else b[j]=p;
            }
            __syncthreads();
            if(__syncthreads_or(overflow!=0))break;
            auto* swap=a;a=b;b=swap;
            if(!threadIdx.x)count=nextCount;
            __syncthreads();
            if(!__syncthreads_or(changes!=0))break;
        }
        if(__syncthreads_or(overflow!=0))continue;
        for(unsigned int pass=0;pass<=MaxDepth;++pass) {
            for(unsigned int i=threadIdx.x;i<s.hashSize;i+=blockDim.x)keys[i]=0;
            for(unsigned int i=threadIdx.x;i<count;i+=blockDim.x)split[i]=0;
            if(!threadIdx.x){changes=0;nextCount=0;}
            __syncthreads();
            for(unsigned int i=threadIdx.x;i<count;i+=blockDim.x) {
                const auto p=a[i];const auto k=key(p.x,p.y,p.depth);auto slot=hash(k,mask);
                while(atomicCAS(keys+slot,Key(0),k)!=0)slot=(slot+1)&mask;
                values[slot]=i;
            }
            __syncthreads();
            for(unsigned int i=threadIdx.x;i<count;i+=blockDim.x)for(int edge=0;edge<4;++edge) {
                const auto n=neighbor(a[i],edge,keys,values,mask);
                if(n!=~0u && a[n].depth+1<a[i].depth)atomicExch(split+n,1u);
            }
            __syncthreads();
            for(unsigned int i=threadIdx.x;i<count;i+=blockDim.x)if(split[i])atomicAdd(&changes,1u);
            __syncthreads();
            if(!__syncthreads_or(changes!=0))break;
            if(pass==MaxDepth){if(!threadIdx.x)overflow=1;__syncthreads();break;}
            for(unsigned int i=threadIdx.x;i<count;i+=blockDim.x) {
                const auto p=a[i];const auto n=split[i]?4u:1u,j=atomicAdd(&nextCount,n);
                if(j+n>s.capacity){atomicExch(&overflow,1u);continue;}
                if(split[i])for(unsigned int k=0;k<4;++k)b[j+k]={p.x*2+(k&1),p.y*2+(k>>1),p.depth+1,0,surface};
                else b[j]=p;
            }
            __syncthreads();
            if(__syncthreads_or(overflow!=0))break;
            auto* swap=a;a=b;b=swap;
            if(!threadIdx.x)count=nextCount;
            __syncthreads();
        }
        if(__syncthreads_or(overflow!=0))continue;
        if(!threadIdx.x){minDepth=MaxDepth;maxDepth=0;}
        __syncthreads();
        for(unsigned int i=threadIdx.x;i<count;i+=blockDim.x) {
            auto p=a[i];p.coarseEdges=0;
            for(int edge=0;edge<4;++edge) {
                const auto n=neighbor(p,edge,keys,values,mask);
                if(n!=~0u && a[n].depth<p.depth)p.coarseEdges|=1u<<edge;
            }
            atomicMin(&minDepth,p.depth);atomicMax(&maxDepth,p.depth);cut[i]=p;
        }
        __syncthreads();
        if(!threadIdx.x)statistics[surface]={count,minDepth,maxDepth,retry!=0,pixels};
        return;
    }
    // Even an extreme metric must leave a complete, bounded surface.
    if(!threadIdx.x){cut[0]={0,0,0,0,surface};statistics[surface]={1,0,0,1,pixels};}
}
__global__ void compact(const TessellationInput* inputs,unsigned int surfaces,const TessellationStats* stats,
                        const Patch* cut,Patch* output,TessellationStats* total) {
    const auto surface=blockIdx.x;unsigned int prefix=0;
    for(unsigned int i=0;i<surface;++i)prefix+=stats[i].clusters;
    for(unsigned int i=threadIdx.x;i<stats[surface].clusters;i+=blockDim.x)output[prefix+i]=cut[inputs[surface].offset+i];
    if(!surface && !threadIdx.x) {
        TessellationStats sum{0,MaxDepth,0,0,0};
        for(unsigned int i=0;i<surfaces;++i){const auto s=stats[i];sum.clusters+=s.clusters;sum.minDepth=min(sum.minDepth,s.minDepth);
            sum.maxDepth=max(sum.maxDepth,s.maxDepth);sum.limited|=s.limited;sum.effectivePixels=fmaxf(sum.effectivePixels,s.effectivePixels);}
        *total=sum;
    }
}
} // namespace tessellation
} // namespace zeno::water
