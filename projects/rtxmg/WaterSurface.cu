#include "WaterSurface.h"
#include "WaterTessellation.h"
#include "WaterSampling.h"
#include "ClusterRuntime.h"
#include <optix_stubs.h>
#include <cufft.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <tuple>

namespace zeno::water {
bool Settings::operator==(const Settings& b) const {
    return std::tie(center,size,wind,absorption,amplitude,period,windSpeed,depth,roughness,ior,time,loopDuration,edgePixels,boundaryWidth,resolution,seed,maxClusters,ellipse)==
        std::tie(b.center,b.size,b.wind,b.absorption,b.amplitude,b.period,b.windSpeed,b.depth,b.roughness,b.ior,b.time,b.loopDuration,b.edgePixels,b.boundaryWidth,b.resolution,b.seed,b.maxClusters,b.ellipse);
}
void validate(const Settings& s) {
    for(float v:s.center)if(!std::isfinite(v))throw std::runtime_error("Water center must be finite");
    for(float v:s.size)if(!std::isfinite(v)||v<=0)throw std::runtime_error("Water dimensions must be positive");
    for(float v:s.wind)if(!std::isfinite(v))throw std::runtime_error("Water wind direction must be finite");
    for(float v:s.absorption)if(!std::isfinite(v)||v<0)throw std::runtime_error("Water absorption must be nonnegative");
    if(!std::isfinite(s.time)||!std::isfinite(s.loopDuration)||s.loopDuration<0||!std::isfinite(s.amplitude)||s.amplitude<0||s.amplitude>100||
       !std::isfinite(s.period)||s.period<=0||!std::isfinite(s.depth)||s.depth<=0||
       !std::isfinite(s.windSpeed)||s.windSpeed<0||s.windSpeed>100||
       !std::isfinite(s.roughness)||s.roughness<0||s.roughness>1||!std::isfinite(s.ior)||s.ior<=1||
       !std::isfinite(s.edgePixels)||s.edgePixels<=0||!std::isfinite(s.boundaryWidth)||s.boundaryWidth<0)
        throw std::runtime_error("Invalid water spectrum, material or tessellation parameter");
    if(s.resolution<32||s.resolution>2048||(s.resolution&(s.resolution-1))||!s.maxClusters||s.maxClusters>65536||s.ellipse>1)
        throw std::runtime_error("Water requires a power-of-two FFT from 32 to 2048, 1..65536 clusters, and a rectangle/ellipse footprint");
}
namespace {
using Clock=std::chrono::steady_clock;
using namespace zeno_water_device;
double elapsed(Clock::time_point t){return std::chrono::duration<double,std::milli>(Clock::now()-t).count();}
void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(std::string("Water CUDA: ")+cudaGetErrorString(e));}
void check(OptixResult e){if(e!=OPTIX_SUCCESS)throw std::runtime_error("Water OptiX: "+std::to_string(int(e)));}
void fftCheck(cufftResult e){if(e!=CUFFT_SUCCESS)throw std::runtime_error("Water cuFFT: "+std::to_string(int(e)));}
template<class T>T* ptr(CUdeviceptr p){return reinterpret_cast<T*>(p);}
struct Buffer {
    CUdeviceptr p=0;size_t bytes=0;
    Buffer()=default;Buffer(const Buffer&)=delete;Buffer& operator=(const Buffer&)=delete;
    ~Buffer(){if(p)cudaFree(ptr<void>(p));}
    void reserve(size_t n,cudaStream_t stream) {
        if(n<=bytes)return;
        void* next=nullptr;check(cudaMallocAsync(&next,std::max<size_t>(256,n),stream));
        if(p)check(cudaFreeAsync(ptr<void>(p),stream));p=reinterpret_cast<CUdeviceptr>(next);bytes=std::max<size_t>(256,n);
    }
};
struct Event {
    cudaEvent_t event{};
    explicit Event(bool timing=false){check(cudaEventCreateWithFlags(&event,timing?cudaEventDefault:cudaEventDisableTiming));}
    ~Event(){if(event)cudaEventDestroy(event);}
    void record(cudaStream_t stream){check(cudaEventRecord(event,stream));}
};
template<class T>struct Pinned {
    T* data=nullptr;size_t capacity=0;
    ~Pinned(){if(data)cudaFreeHost(data);}
    void reserve(size_t n){if(n<=capacity)return;if(data)check(cudaFreeHost(data));check(cudaHostAlloc(reinterpret_cast<void**>(&data),n*sizeof(T),cudaHostAllocDefault));capacity=n;}
};
uint32_t hash(uint32_t v){v^=v>>16;v*=0x7feb352du;v^=v>>15;v*=0x846ca68bu;return v^(v>>16);}
double uniform(uint32_t v){return (double(hash(v))+.5)/4294967296.;}
bool sameSpectrum(const Settings& a,const Settings& b) {
    return a.amplitude==b.amplitude && a.period==b.period && a.windSpeed==b.windSpeed && a.wind==b.wind &&
        a.depth==b.depth && a.resolution==b.resolution && a.seed==b.seed;
}
bool sameTopology(const Settings& a,const Settings& b) {
    // Wave time, spectrum and shading do not change the projected patch bound.
    // Amplitude does: selection uses a conservative height envelope.
    return std::tie(a.center,a.size,a.amplitude,a.edgePixels,a.maxClusters,a.ellipse)==
        std::tie(b.center,b.size,b.amplitude,b.edgePixels,b.maxClusters,b.ellipse);
}
struct Spectrum {
    Settings settings;
    Buffer initial,transform,workspace;
    Pinned<cufftComplex> source;
    cufftHandle plan=0;
    ~Spectrum(){if(plan)cufftDestroy(plan);}
    void initialize(const Settings& s,cudaStream_t stream) {
        settings=s;const uint32_t n=s.resolution;const size_t count=size_t(n)*n;
        source.reserve(count);std::vector<double> power(count);
        const double windLength=std::hypot(s.wind[0],s.wind[1]);
        const double wx=windLength>0?s.wind[0]/windLength:1.,wz=windLength>0?s.wind[1]/windLength:0.;
        const double L=double(s.windSpeed)*s.windSpeed/9.81;double energy=0;
        for(uint32_t y=0;y<n;++y)for(uint32_t x=0;x<n;++x) {
            const double kx=2.*3.14159265358979323846*(x<=n/2?int(x):int(x)-int(n))/s.period;
            const double kz=2.*3.14159265358979323846*(y<=n/2?int(y):int(y)-int(n))/s.period;
            const double k2=kx*kx+kz*kz;
            const double directional=k2>0?std::pow((kx*wx+kz*wz)/std::sqrt(k2),2):0;
            const double p=k2>0 && L>0?std::exp(-1./(k2*L*L))*directional*std::exp(-k2*.0004)/(k2*k2):0.;
            power[size_t(y)*n+x]=p;energy+=p;
        }
        const double normalization=energy>0?double(s.amplitude)*count/std::sqrt(2.*energy):0;
        for(uint32_t i=0;i<count;++i) {
            const double radius=std::sqrt(-2.*std::log(uniform(i*2+s.seed)));
            const double angle=6.28318530717958647692*uniform(i*2+s.seed+1);
            const double scale=normalization*std::sqrt(power[i]*.5);
            source.data[i]={float(radius*std::cos(angle)*scale),float(radius*std::sin(angle)*scale)};
        }
        initial.reserve(count*sizeof(cufftComplex),stream);transform.reserve(3*count*sizeof(cufftComplex),stream);
        check(cudaMemcpyAsync(ptr<void>(initial.p),source.data,count*sizeof(cufftComplex),cudaMemcpyHostToDevice,stream));
        fftCheck(cufftCreate(&plan));fftCheck(cufftSetAutoAllocation(plan,0));
        int dimensions[2]{int(n),int(n)};size_t work=0;
        fftCheck(cufftMakePlanMany(plan,2,dimensions,nullptr,1,int(count),nullptr,1,int(count),CUFFT_C2C,3,&work));
        workspace.reserve(std::max<size_t>(work,1),stream);fftCheck(cufftSetWorkArea(plan,ptr<void>(workspace.p)));fftCheck(cufftSetStream(plan,stream));
    }
};
__device__ float2 multiply(float2 a,float2 b){return make_float2(a.x*b.x-a.y*b.y,a.x*b.y+a.y*b.x);}
__global__ void evolve(const float2* initial,float2* output,uint32_t n,float period,float depth,float time,float loopDuration) {
    const uint32_t i=blockIdx.x*blockDim.x+threadIdx.x,count=n*n;if(i>=count)return;
    const uint32_t x=i%n,y=i/n;
    const float kx=6.28318530717958647692f*float(x<=n/2?int(x):int(x)-int(n))/period;
    const float kz=6.28318530717958647692f*float(y<=n/2?int(y):int(y)-int(n))/period;
    const float k=hypotf(kx,kz);float omega=sqrtf(9.81f*k*tanhf(k*depth));
    // Integer multiples of the loop frequency preserve the dispersion relation
    // approximately while making both heights and velocities loop continuously.
    if(loopDuration>0.f) {
        const float fundamental=6.28318530717958647692f/loopDuration;
        omega=roundf(omega/fundamental)*fundamental;
    }
    float sine,cosine;sincosf(omega*time,&sine,&cosine);
    const auto a=multiply(initial[i],make_float2(cosine,sine));
    const auto mirror=initial[((n-y)&(n-1))*n+((n-x)&(n-1))];
    const auto b=multiply(make_float2(mirror.x,-mirror.y),make_float2(cosine,-sine));
    const float2 h=make_float2(a.x+b.x,a.y+b.y);output[i]=h;
    const float dx=x==n/2?0.f:kx,dz=y==n/2?0.f:kz;
    output[count+i]=make_float2(-h.y*dx,h.x*dx);output[2*count+i]=make_float2(-h.y*dz,h.x*dz);
}
__global__ void packField(const float2* transformed,float4* field,uint32_t count) {
    const auto i=blockIdx.x*blockDim.x+threadIdx.x;if(i<count)
        field[i]=make_float4(transformed[i].x/count,transformed[count+i].x/count,transformed[2*count+i].x/count,0.f);
}
__global__ void fill(const Patch* patches,const Surface* surfaces,const TessellationStats* stats,float3* vertices,
                     const CUdeviceptr* templates,OptixClusterAccelBuildInputTemplatesArgs* arguments,uint32_t sbt) {
    const auto id=blockIdx.x;if(id>=stats->clusters)return;const auto p=patches[id];const auto s=surfaces[p.surface];
    for(unsigned v=threadIdx.x;v<GridVertices;v+=blockDim.x)vertices[id*GridVertices+v]=position(s,p,v%(GridSegments+1),v/(GridSegments+1));
    if(!threadIdx.x) {
        OptixClusterAccelBuildInputTemplatesArgs a{};a.clusterIdOffset=id;a.sbtIndexOffset=sbt;
        a.clusterTemplate=templates[0];a.vertexBuffer=reinterpret_cast<CUdeviceptr>(vertices+id*GridVertices);a.vertexStrideInBytes=sizeof(float3);arguments[id]=a;
    }
}
__global__ void gasArguments(const TessellationStats* stats,CUdeviceptr handles,OptixClusterAccelBuildInputClustersArgs* args) {
    if(!threadIdx.x)*args={stats->clusters,0,handles};
}

}

struct Runtime::Impl {
    struct Bank {
        Buffer positions,patches,surfaces,arguments,clas,clasHandles,gas,gasHandle,gasArgs,tessellationInputs,tessellationStats,tessellationTotal;
        std::vector<std::unique_ptr<Buffer>> fields;
        std::vector<float> fieldTimes,fieldLoopDurations;
        std::vector<uint64_t> fieldGenerations;
        std::vector<Settings> settings;
        Pinned<zeno_water_device::Surface> uploadSurfaces;
        Pinned<TessellationInput> uploadTessellation;
        Pinned<TessellationStats> tessellationResult;
        Pinned<uint64_t> result;
        Event done,read,auxiliary,begin{true},fftEnd{true},tessellationEnd{true},clasEnd{true},gasEnd{true};
        bool wasRead=false,wasAuxiliary=false;
        uint32_t sbt=0;
        uint64_t topologyRevision=0;
        size_t reservedClusters=0;
        OptixAccelBufferSizes clasSizes{},gasSizes{};
        Statistics stats;
    };
    cudaStream_t stream{};
    Bank banks[2];
    Buffer scratch,templateData,templateHandle,templateArgs;
    Buffer workA,workB,workCut,hashKeys,hashValues,split;
    uint32_t templateSbtLimit=0;
    Pinned<OptixClusterAccelBuildInputGridsArgs> templateUpload;
    std::vector<std::unique_ptr<Spectrum>> spectra;
    std::vector<uint64_t> generations;
    std::vector<Settings> plannedSettings;
    View plannedView{};
    bool havePlan=false;
    int active=-1,pending=-1;
    uint64_t revision=0,topologyRevision=0;
    Impl(){check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));}
    ~Impl(){
        if(stream)cudaStreamSynchronize(stream);
        for(auto& b:banks){if(b.wasRead)cudaEventSynchronize(b.read.event);if(b.wasAuxiliary)cudaEventSynchronize(b.auxiliary.event);}
        spectra.clear();if(stream)cudaStreamDestroy(stream);
    }
    void build(OptixDeviceContext ctx,OptixClusterAccelBuildInput input,CUdeviceptr args,Buffer& output,Buffer& handles,const OptixAccelBufferSizes* reserved=nullptr,CUdeviceptr count=0) {
        OptixAccelBufferSizes sizes{};
        if(reserved)sizes=*reserved;
        else check(optixClusterAccelComputeMemoryUsage(ctx,OPTIX_CLUSTER_ACCEL_BUILD_MODE_IMPLICIT_DESTINATIONS,&input,&sizes));
        output.reserve(sizes.outputSizeInBytes,stream);scratch.reserve(sizes.tempSizeInBytes,stream);
        OptixClusterAccelBuildModeDesc desc{};desc.mode=OPTIX_CLUSTER_ACCEL_BUILD_MODE_IMPLICIT_DESTINATIONS;
        desc.implicitDest.outputBuffer=output.p;desc.implicitDest.outputBufferSizeInBytes=output.bytes;
        desc.implicitDest.tempBuffer=scratch.p;desc.implicitDest.tempBufferSizeInBytes=scratch.bytes;
        desc.implicitDest.outputHandlesBuffer=handles.p;
        check(optixClusterAccelBuild(ctx,stream,&desc,&input,args,count,0));
    }
    void initializeTemplate(OptixDeviceContext ctx,uint32_t sbt) {
        if(templateData.p && sbt<=templateSbtLimit)return;
        unsigned int limit=0;check(optixDeviceContextGetProperty(ctx,OPTIX_DEVICE_PROPERTY_LIMIT_MAX_STRUCTURED_GRID_RESOLUTION,&limit,sizeof(limit)));
        if(limit<GridSegments)throw std::runtime_error("The GPU does not support the required 8x8 structured cluster grid");
        templateUpload.reserve(1);auto& a=templateUpload.data[0];a={};a.dimensions[0]=GridSegments;a.dimensions[1]=GridSegments;
        a.basePrimitiveInfo.primitiveFlags=OPTIX_CLUSTER_ACCEL_PRIMITIVE_FLAG_DISABLE_TRIANGLE_FACE_CULLING;
        templateArgs.reserve(sizeof(a),stream);templateHandle.reserve(sizeof(uint64_t),stream);
        check(cudaMemcpyAsync(ptr<void>(templateArgs.p),&a,sizeof(a),cudaMemcpyHostToDevice,stream));
        OptixClusterAccelBuildInput input{};input.type=OPTIX_CLUSTER_ACCEL_BUILD_TYPE_TEMPLATES_FROM_GRIDS;
        input.grids.flags=OPTIX_CLUSTER_ACCEL_BUILD_FLAG_PREFER_FAST_TRACE;input.grids.maxArgCount=1;
        input.grids.vertexFormat=OPTIX_VERTEX_FORMAT_FLOAT3;input.grids.maxWidth=GridSegments;input.grids.maxHeight=GridSegments;
        // Template limits include the material offset applied at instantiation.
        input.grids.maxSbtIndexValue=sbt;
        build(ctx,input,templateArgs.p,templateData,templateHandle);
        templateSbtLimit=sbt;
    }
    void publish() {
        auto& b=banks[pending];active=pending;pending=-1;b.stats.pending=false;b.stats.revision=++revision;
        check(cudaEventElapsedTime(&b.stats.fftMs,b.begin.event,b.fftEnd.event));
        check(cudaEventElapsedTime(&b.stats.tessellationMs,b.fftEnd.event,b.tessellationEnd.event));
        check(cudaEventElapsedTime(&b.stats.clasMs,b.tessellationEnd.event,b.clasEnd.event));
        const auto t=b.tessellationResult.data[0];
        if(!t.clusters || t.clusters>b.reservedClusters)throw std::runtime_error("Invalid GPU water tessellation count");
        b.stats.clusters=t.clusters;b.stats.triangles=t.clusters*GridSegments*GridSegments*2;
        b.stats.minDepth=t.minDepth;b.stats.maxDepth=t.maxDepth;b.stats.qualityLimited=t.limited!=0;
        b.stats.effectiveEdgePixels=t.effectivePixels;
        check(cudaEventElapsedTime(&b.stats.gasMs,b.clasEnd.event,b.gasEnd.event));
    }
};
Runtime::Runtime():impl(std::make_unique<Impl>()){}
Runtime::~Runtime()=default;
bool Runtime::update(OptixDeviceContext ctx,uint32_t sbt,const std::vector<Settings>& inputSettings,View view,bool interactive) {
    auto& p=*impl;bool changed=false;
    auto settings=inputSettings;
    for(auto& s:settings) {
        validate(s);
        if(s.loopDuration>0.f) {
            s.time=std::fmod(s.time,s.loopDuration);
            if(s.time<0.f)s.time+=s.loopDuration;
        }
    }
    for(auto v:view.eye)if(!std::isfinite(v))throw std::runtime_error("Nonfinite water camera");
    for(const auto& axis:{view.right,view.up,view.front})for(float v:axis)if(!std::isfinite(v))throw std::runtime_error("Nonfinite water camera basis");
    if(!std::isfinite(view.tanX)||!std::isfinite(view.tanY)||view.tanX<0||view.tanY<0)throw std::runtime_error("Invalid water view bounds");
    if(!std::isfinite(view.projection)||view.projection<=0)throw std::runtime_error("Invalid water projection");
    if(settings.empty())return false;
    const size_t capacity=std::accumulate(settings.begin(),settings.end(),size_t(0),[](size_t n,const Settings& s){return n+s.maxClusters;});
    if(capacity>262144 || settings.size()>64)throw std::runtime_error("Water exceeds the combined surface/cluster limit");
    const bool blocking=!interactive || p.active<0 || p.banks[p.active].sbt!=sbt;
    if(p.pending>=0) {
        auto status=cudaEventQuery(p.banks[p.pending].done.event);
        if(status==cudaErrorNotReady && !blocking)return false;
        if(status==cudaErrorNotReady)check(cudaEventSynchronize(p.banks[p.pending].done.event));else check(status);
        p.publish();changed=true;
    }
    const bool sameView=p.havePlan && p.plannedView.eye==view.eye && p.plannedView.projection==view.projection &&
        p.plannedView.right==view.right && p.plannedView.up==view.up && p.plannedView.front==view.front &&
        p.plannedView.tanX==view.tanX && p.plannedView.tanY==view.tanY;
    if(sameView && p.plannedSettings==settings && p.banks[p.active].sbt==sbt)return changed;
    const bool reuseTopology=sameView && p.plannedSettings.size()==settings.size() &&
        std::equal(settings.begin(),settings.end(),p.plannedSettings.begin(),sameTopology);
    if(!reuseTopology)++p.topologyRevision;
    p.plannedView=view;p.plannedSettings=settings;p.havePlan=true;
    const auto submitStart=Clock::now();
    const std::lock_guard<std::recursive_mutex> submission(zeno::cluster::renderSubmissionMutex());
    const int next=p.active<0?0:1-p.active;auto& b=p.banks[next];
    if(b.wasRead)check(cudaStreamWaitEvent(p.stream,b.read.event,0));
    if(b.wasAuxiliary)check(cudaStreamWaitEvent(p.stream,b.auxiliary.event,0));
    p.initializeTemplate(ctx,sbt);
    while(p.spectra.size()<settings.size()){p.spectra.emplace_back();p.generations.push_back(0);}
    while(b.fields.size()<settings.size()) {
        b.fields.push_back(std::make_unique<Buffer>());
        b.fieldTimes.push_back(std::numeric_limits<float>::quiet_NaN());
        b.fieldLoopDurations.push_back(0);b.fieldGenerations.push_back(0);
    }
    for(size_t i=0;i<settings.size();++i) {
        if(!p.spectra[i] || !sameSpectrum(p.spectra[i]->settings,settings[i])) {
            auto spectrum=std::make_unique<Spectrum>();spectrum->initialize(settings[i],p.stream);
            p.spectra[i]=std::move(spectrum);++p.generations[i];
        }
    }
    const bool sbtChanged=b.sbt!=sbt;b.settings=settings;b.sbt=sbt;
    b.positions.reserve(capacity*GridVertices*sizeof(float3),p.stream);b.patches.reserve(capacity*sizeof(zeno_water_device::Patch),p.stream);
    b.surfaces.reserve(settings.size()*sizeof(zeno_water_device::Surface),p.stream);b.arguments.reserve(capacity*sizeof(OptixClusterAccelBuildInputTemplatesArgs),p.stream);
    b.clasHandles.reserve(capacity*sizeof(uint64_t),p.stream);b.gasHandle.reserve(sizeof(uint64_t),p.stream);b.gasArgs.reserve(sizeof(OptixClusterAccelBuildInputClustersArgs),p.stream);
    b.uploadSurfaces.reserve(settings.size());b.uploadTessellation.reserve(settings.size());b.tessellationResult.reserve(1);b.result.reserve(1);
    const bool selectTopology=!reuseTopology;
    unsigned int offset=0,hashCapacity=0;
    for(size_t i=0;selectTopology && i<settings.size();++i) {
        const auto& s=settings[i];auto& t=b.uploadTessellation.data[i];t={};
        std::copy(s.center.begin(),s.center.end(),t.center);std::copy(s.size.begin(),s.size.end(),t.size);std::copy(view.eye.begin(),view.eye.end(),t.eye);
        std::copy(view.right.begin(),view.right.end(),t.right);std::copy(view.up.begin(),view.up.end(),t.up);std::copy(view.front.begin(),view.front.end(),t.front);t.tanX=view.tanX;t.tanY=view.tanY;
        t.projection=view.projection;t.amplitude=s.amplitude;t.edgePixels=s.edgePixels;t.ellipse=s.ellipse;t.capacity=s.maxClusters;t.offset=offset;t.hashOffset=hashCapacity;
        t.hashSize=1;while(t.hashSize<2*s.maxClusters)t.hashSize*=2;
        offset+=s.maxClusters;hashCapacity+=t.hashSize;
    }
    b.tessellationTotal.reserve(sizeof(TessellationStats),p.stream);
    if(selectTopology) {
        p.workA.reserve(capacity*sizeof(zeno_water_device::Patch),p.stream);p.workB.reserve(capacity*sizeof(zeno_water_device::Patch),p.stream);
        p.workCut.reserve(capacity*sizeof(zeno_water_device::Patch),p.stream);p.split.reserve(capacity*sizeof(unsigned int),p.stream);
        p.hashKeys.reserve(hashCapacity*sizeof(tessellation::Key),p.stream);p.hashValues.reserve(hashCapacity*sizeof(unsigned int),p.stream);
        b.tessellationInputs.reserve(settings.size()*sizeof(TessellationInput),p.stream);
        b.tessellationStats.reserve(settings.size()*sizeof(TessellationStats),p.stream);
        check(cudaMemcpyAsync(ptr<void>(b.tessellationInputs.p),b.uploadTessellation.data,settings.size()*sizeof(TessellationInput),cudaMemcpyHostToDevice,p.stream));
    }
    b.begin.record(p.stream);
    for(size_t i=0;i<settings.size();++i) {
        const auto& s=settings[i];auto& spectrum=*p.spectra[i];const uint32_t size=s.resolution*s.resolution;
        b.fields[i]->reserve(size*sizeof(float4),p.stream);
        if(b.fieldTimes[i]!=s.time || b.fieldLoopDurations[i]!=s.loopDuration || b.fieldGenerations[i]!=p.generations[i]) {
            evolve<<<(size+255)/256,256,0,p.stream>>>(ptr<const float2>(spectrum.initial.p),ptr<float2>(spectrum.transform.p),s.resolution,s.period,s.depth,s.time,s.loopDuration);
            fftCheck(cufftExecC2C(spectrum.plan,ptr<cufftComplex>(spectrum.transform.p),ptr<cufftComplex>(spectrum.transform.p),CUFFT_INVERSE));
            packField<<<(size+255)/256,256,0,p.stream>>>(ptr<const float2>(spectrum.transform.p),ptr<float4>(b.fields[i]->p),size);
            b.fieldTimes[i]=s.time;b.fieldLoopDurations[i]=s.loopDuration;b.fieldGenerations[i]=p.generations[i];
        }
        auto& d=b.uploadSurfaces.data[i];d={};d.field=ptr<const float4>(b.fields[i]->p);
        std::copy(s.center.begin(),s.center.end(),d.center);std::copy(s.size.begin(),s.size.end(),d.size);std::copy(s.absorption.begin(),s.absorption.end(),d.absorption);
        d.period=s.period;d.amplitude=s.amplitude;d.roughness=s.roughness;d.ior=s.ior;d.boundaryWidth=s.boundaryWidth;d.resolution=s.resolution;d.ellipse=s.ellipse;
    }
    b.fftEnd.record(p.stream);
    unsigned int fillClusters=unsigned(capacity);
    if(selectTopology) {
        tessellation::select<<<unsigned(settings.size()),256,0,p.stream>>>(ptr<const TessellationInput>(b.tessellationInputs.p),ptr<zeno_water_device::Patch>(p.workA.p),
            ptr<zeno_water_device::Patch>(p.workB.p),ptr<zeno_water_device::Patch>(p.workCut.p),ptr<tessellation::Key>(p.hashKeys.p),ptr<unsigned int>(p.hashValues.p),ptr<unsigned int>(p.split.p),ptr<TessellationStats>(b.tessellationStats.p));
        tessellation::compact<<<unsigned(settings.size()),256,0,p.stream>>>(ptr<const TessellationInput>(b.tessellationInputs.p),unsigned(settings.size()),ptr<const TessellationStats>(b.tessellationStats.p),
            ptr<const zeno_water_device::Patch>(p.workCut.p),ptr<zeno_water_device::Patch>(b.patches.p),ptr<TessellationStats>(b.tessellationTotal.p));
    } else {
        const auto& source=p.banks[p.active];fillClusters=source.stats.clusters;
        if(b.topologyRevision!=p.topologyRevision) {
            // Only the other bank's first use needs a copy. Thereafter a fixed
            // camera animates without selection, balancing or patch transfers.
            check(cudaMemcpyAsync(ptr<void>(b.patches.p),ptr<const void>(source.patches.p),fillClusters*sizeof(zeno_water_device::Patch),cudaMemcpyDeviceToDevice,p.stream));
            check(cudaMemcpyAsync(ptr<void>(b.tessellationTotal.p),ptr<const void>(source.tessellationTotal.p),sizeof(TessellationStats),cudaMemcpyDeviceToDevice,p.stream));
        }
    }
    b.topologyRevision=p.topologyRevision;
    b.tessellationEnd.record(p.stream);
    check(cudaMemcpyAsync(ptr<void>(b.surfaces.p),b.uploadSurfaces.data,settings.size()*sizeof(zeno_water_device::Surface),cudaMemcpyHostToDevice,p.stream));
    fill<<<fillClusters,128,0,p.stream>>>(ptr<const zeno_water_device::Patch>(b.patches.p),ptr<const zeno_water_device::Surface>(b.surfaces.p),ptr<const TessellationStats>(b.tessellationTotal.p),
        ptr<float3>(b.positions.p),ptr<const CUdeviceptr>(p.templateHandle.p),ptr<OptixClusterAccelBuildInputTemplatesArgs>(b.arguments.p),sbt);
    check(cudaGetLastError());
    OptixClusterAccelBuildInput input{};input.type=OPTIX_CLUSTER_ACCEL_BUILD_TYPE_CLUSTERS_FROM_TEMPLATES;
    input.triangles.flags=OPTIX_CLUSTER_ACCEL_BUILD_FLAG_PREFER_FAST_TRACE;input.triangles.maxArgCount=uint32_t(capacity);input.triangles.vertexFormat=OPTIX_VERTEX_FORMAT_FLOAT3;
    input.triangles.maxSbtIndexValue=sbt;input.triangles.maxUniqueSbtIndexCountPerArg=1;
    input.triangles.maxTriangleCountPerArg=GridSegments*GridSegments*2;input.triangles.maxVertexCountPerArg=GridVertices;
    if(capacity>b.reservedClusters || sbtChanged) {
        check(optixClusterAccelComputeMemoryUsage(ctx,OPTIX_CLUSTER_ACCEL_BUILD_MODE_IMPLICIT_DESTINATIONS,&input,&b.clasSizes));
        OptixClusterAccelBuildInput allocationInput{};allocationInput.type=OPTIX_CLUSTER_ACCEL_BUILD_TYPE_GASES_FROM_CLUSTERS;
        allocationInput.clusters.flags=OPTIX_CLUSTER_ACCEL_BUILD_FLAG_PREFER_FAST_TRACE;allocationInput.clusters.maxArgCount=1;
        allocationInput.clusters.maxTotalClusterCount=uint32_t(capacity);allocationInput.clusters.maxClusterCountPerArg=uint32_t(capacity);
        check(optixClusterAccelComputeMemoryUsage(ctx,OPTIX_CLUSTER_ACCEL_BUILD_MODE_IMPLICIT_DESTINATIONS,&allocationInput,&b.gasSizes));
        b.reservedClusters=capacity;
    }
    p.build(ctx,input,b.arguments.p,b.clas,b.clasHandles,&b.clasSizes,b.tessellationTotal.p);b.clasEnd.record(p.stream);
    gasArguments<<<1,1,0,p.stream>>>(ptr<const TessellationStats>(b.tessellationTotal.p),b.clasHandles.p,ptr<OptixClusterAccelBuildInputClustersArgs>(b.gasArgs.p));
    input={};input.type=OPTIX_CLUSTER_ACCEL_BUILD_TYPE_GASES_FROM_CLUSTERS;input.clusters.flags=OPTIX_CLUSTER_ACCEL_BUILD_FLAG_PREFER_FAST_TRACE;
    input.clusters.maxArgCount=1;input.clusters.maxTotalClusterCount=uint32_t(capacity);input.clusters.maxClusterCountPerArg=uint32_t(capacity);
    p.build(ctx,input,b.gasArgs.p,b.gas,b.gasHandle,&b.gasSizes);b.gasEnd.record(p.stream);
    check(cudaMemcpyAsync(b.result.data,ptr<void>(b.gasHandle.p),sizeof(uint64_t),cudaMemcpyDeviceToHost,p.stream));
    check(cudaMemcpyAsync(b.tessellationResult.data,ptr<void>(b.tessellationTotal.p),sizeof(TessellationStats),cudaMemcpyDeviceToHost,p.stream));b.done.record(p.stream);
    b.stats={};b.stats.pending=true;b.stats.submitMs=elapsed(submitStart);
    b.stats.reusedTopology=reuseTopology;b.stats.topologyRevision=p.topologyRevision;p.pending=next;
    if(blocking){check(cudaEventSynchronize(b.done.event));p.publish();changed=true;}
    return changed;
}
void Runtime::rendered(cudaStream_t stream,bool auxiliary) {
    auto& p=*impl;if(p.active<0)return;auto& b=p.banks[p.active];
    if(auxiliary){b.auxiliary.record(stream);b.wasAuxiliary=true;}else{b.read.record(stream);b.wasRead=true;}
}
OptixTraversableHandle Runtime::handle() const {return impl->active<0?0:impl->banks[impl->active].result.data[0];}
zeno_water_device::Launch Runtime::launch() const {
    if(impl->active<0)return {};const auto& b=impl->banks[impl->active];
    return {ptr<const zeno_water_device::Surface>(b.surfaces.p),ptr<const zeno_water_device::Patch>(b.patches.p),uint32_t(b.settings.size()),b.stats.clusters};
}
Statistics Runtime::statistics() const {
    const auto& p=*impl;Statistics s=p.active>=0?p.banks[p.active].stats:Statistics{};s.pending=p.pending>=0;
    s.bytes=p.scratch.bytes+p.templateData.bytes+p.templateHandle.bytes+p.templateArgs.bytes+
        p.workA.bytes+p.workB.bytes+p.workCut.bytes+p.hashKeys.bytes+p.hashValues.bytes+p.split.bytes;
    for(const auto& b:p.banks) {
        s.bytes+=b.positions.bytes+b.patches.bytes+b.surfaces.bytes+b.arguments.bytes+b.clas.bytes+b.clasHandles.bytes+b.gas.bytes+b.gasHandle.bytes+b.gasArgs.bytes+b.tessellationInputs.bytes+b.tessellationStats.bytes+b.tessellationTotal.bytes;
        for(const auto& field:b.fields)s.bytes+=field->bytes;
    }
    for(const auto& spectrum:p.spectra)if(spectrum)s.bytes+=spectrum->initial.bytes+spectrum->transform.bytes+spectrum->workspace.bytes;
    return s;
}
}
