#include "ClusterCoverageGpu.h"
#include <cuda_runtime.h>

namespace zeno::cluster::coverage {
namespace {
void check(cudaError_t e) {if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
struct Range {uint32_t group,coarse,nodes,begin,end;};
__device__ double box(Point p,const Surface::Node& n) {
    double d=0;
    for(int j=0;j<3;++j){const double v=fmax(fmax(n.lo.x[j]-p.x[j],0.),p.x[j]-n.hi.x[j]);d+=v*v;}
    return d;
}
__device__ double closest(Point p,const Triangle* triangles,const Surface::Node* nodes,uint32_t& found) {
    uint32_t stack[64],count=1;stack[0]=0;double best=INFINITY;found=0;
    // The host BVH halves every non-leaf range, so uint32 triangle counts
    // require at most 33 stack entries; 64 leaves ample space.
    while(count) {
        const auto& n=nodes[stack[--count]];
        if(box(p,n)>=best)continue;
        if(n.left) {
            const double a=box(p,nodes[n.left]),b=box(p,nodes[n.right]);
            stack[count++]=a<b?n.right:n.left;
            stack[count++]=a<b?n.left:n.right;
        } else for(uint32_t i=n.begin;i<n.end;++i) {
            const double d=triangleDistance(p,triangles[i]);
            if(d<best){best=d;found=i;}
        }
    }
    return sqrt(best);
}
__device__ double bound(const Triangle& t,const Triangle* triangles,const Surface::Node* nodes) {
    double d=0,edge=0;uint32_t candidates[7];
    for(int i=0;i<3;++i) {
        d=fmax(d,closest(t.p[i],triangles,nodes,candidates[i*2]));
        d=fmax(d,closest((t.p[i]+t.p[(i+1)%3])*.5,triangles,nodes,candidates[i*2+1]));
        const auto e=t.p[i]-t.p[(i+1)%3];edge=fmax(edge,dot(e,e));
    }
    d=fmax(d,closest((t.p[0]+t.p[1]+t.p[2])*(1./3),triangles,nodes,candidates[6]));
    double result=d+sqrt(edge)/3;
    for(int i=0;i<7;++i) {
        bool repeated=false;for(int j=0;j<i;++j)repeated|=candidates[i]==candidates[j];if(repeated)continue;
        double b=0;for(int j=0;j<3;++j)b=fmax(b,triangleDistance(t.p[j],triangles[candidates[i]]));
        result=fmin(result,sqrt(b));
    }
    return result;
}
__global__ void evaluate(const Triangle* coarse,const Surface::Node* nodes,const Triangle* fine,
                         const Range* ranges,double* bounds,bool convex) {
    const auto r=ranges[blockIdx.x];double b=0;
    for(uint32_t i=r.begin+threadIdx.x;i<r.end;i+=blockDim.x) {
        if(convex) {
            const auto& t=fine[i];uint32_t id;
            closest((t.p[0]+t.p[1]+t.p[2])*(1./3),coarse+r.coarse,nodes+r.nodes,id);
            double d=0;for(int j=0;j<3;++j)d=fmax(d,triangleDistance(t.p[j],coarse[r.coarse+id]));
            b=fmax(b,sqrt(d));
        } else b=fmax(b,bound(fine[i],coarse+r.coarse,nodes+r.nodes));
    }
    __shared__ double maximum[128];maximum[threadIdx.x]=b;__syncthreads();
    for(int step=64;step;step/=2){if(threadIdx.x<step)maximum[threadIdx.x]=fmax(maximum[threadIdx.x],maximum[threadIdx.x+step]);__syncthreads();}
    if(threadIdx.x==0)bounds[blockIdx.x]=maximum[0];
}
struct DeviceBuffer {
    void* p=nullptr;size_t capacity=0;
    ~DeviceBuffer(){if(p)cudaFree(p);}
    void reserve(size_t bytes){if(bytes>capacity){if(p)check(cudaFree(p));p=nullptr;capacity=0;check(cudaMalloc(&p,bytes));capacity=bytes;}}
    template<class T>T* ptr(){return static_cast<T*>(p);}
    template<class T>void upload(const std::vector<T>& data,cudaStream_t stream){reserve(data.size()*sizeof(T));check(cudaMemcpyAsync(p,data.data(),data.size()*sizeof(T),cudaMemcpyHostToDevice,stream));}
};
}
struct GpuBatch::Impl {
    std::vector<Triangle> coarse,fine;
    std::vector<Surface::Node> nodes;
    std::vector<Range> ranges;
    DeviceBuffer dc,df,dn,dr,db;
    cudaStream_t stream=nullptr;
    bool convex;
    Impl(bool c):convex(c){check(cudaSetDevice(0));check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));}
    ~Impl(){if(stream){cudaStreamSynchronize(stream);cudaStreamDestroy(stream);}}
};
GpuBatch::GpuBatch(bool convex):impl(std::make_unique<Impl>(convex)){}
GpuBatch::~GpuBatch()=default;
void GpuBatch::add(uint32_t group,const Surface& s,const std::vector<Triangle>& f) {
    auto& b=*impl;
    if(b.coarse.size()+s.data().size()>UINT32_MAX||b.nodes.size()+s.hierarchy().size()>UINT32_MAX||b.fine.size()+f.size()>UINT32_MAX)
        throw std::runtime_error("Coverage batch index overflow");
    b.ranges.push_back({group,uint32_t(b.coarse.size()),uint32_t(b.nodes.size()),uint32_t(b.fine.size()),uint32_t(b.fine.size()+f.size())});
    b.coarse.insert(b.coarse.end(),s.data().begin(),s.data().end());
    b.nodes.insert(b.nodes.end(),s.hierarchy().begin(),s.hierarchy().end());
    b.fine.insert(b.fine.end(),f.begin(),f.end());
}
size_t GpuBatch::triangles()const{return impl->fine.size();}
void GpuBatch::flush(std::vector<double>& bounds) {
    auto& b=*impl;if(b.ranges.empty())return;
    b.dc.upload(b.coarse,b.stream);b.dn.upload(b.nodes,b.stream);b.df.upload(b.fine,b.stream);b.dr.upload(b.ranges,b.stream);
    b.db.reserve(b.ranges.size()*sizeof(double));
    evaluate<<<uint32_t(b.ranges.size()),128,0,b.stream>>>(b.dc.ptr<Triangle>(),b.dn.ptr<Surface::Node>(),b.df.ptr<Triangle>(),b.dr.ptr<Range>(),b.db.ptr<double>(),b.convex);
    check(cudaGetLastError());std::vector<double> result(b.ranges.size());
    check(cudaMemcpyAsync(result.data(),b.db.p,result.size()*sizeof(double),cudaMemcpyDeviceToHost,b.stream));
    check(cudaStreamSynchronize(b.stream));
    for(size_t i=0;i<result.size();++i) {
        if(!std::isfinite(result[i]))throw std::runtime_error("Non-finite GPU coverage bound");
        bounds.at(b.ranges[i].group)=result[i];
    }
    b.coarse.clear();b.nodes.clear();b.fine.clear();b.ranges.clear();
}
}
