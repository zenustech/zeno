#include "ClusterRuntime.h"
#include <optix_function_table_definition.h>
#include <optix_stubs.h>
#include <optix_stack_size.h>
#include <cuda_runtime.h>
#include <nvrtc.h>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <fstream>
#include <iostream>
#include <cstring>
#include <cfloat>
#include <cstdlib>

static void check(cudaError_t r){if(r)throw std::runtime_error(cudaGetErrorString(r));}
static void check(OptixResult r){if(r)throw std::runtime_error("OptiX "+std::to_string(r));}
static void check(nvrtcResult r){if(r)throw std::runtime_error(nvrtcGetErrorString(r));}
template<class T> CUdeviceptr upload(const T* data,size_t count){CUdeviceptr p{};check(cudaMalloc(reinterpret_cast<void**>(&p),count*sizeof(T)));if(data)check(cudaMemcpy(reinterpret_cast<void*>(p),data,count*sizeof(T),cudaMemcpyHostToDevice));return p;}
template<class T> struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) Record{char header[OPTIX_SBT_RECORD_HEADER_SIZE];T data;};
struct Params{OptixTraversableHandle handle;const zeno_cluster_device::Cluster* clusters;float3 eye,u,v,w;float4* image;unsigned int width,height;};
static const char* shader=R"CUDA(
#include <optix.h>
#include <optix_device.h>
struct Vertex{float position[3],normal[3];};
struct Cluster{const Vertex* vertices;const unsigned char* indices;unsigned int lod,geometry,group;};
struct Params{OptixTraversableHandle handle;const Cluster* clusters;float3 eye,u,v,w;float4* image;unsigned int width,height;};
extern "C" {__constant__ Params params;}
static __device__ float3 add(float3 a,float3 b){return make_float3(a.x+b.x,a.y+b.y,a.z+b.z);}
static __device__ float3 mul(float3 a,float b){return make_float3(a.x*b,a.y*b,a.z*b);}
static __device__ float dot3(float3 a,float3 b){return a.x*b.x+a.y*b.y+a.z*b.z;}
static __device__ float3 norm(float3 a){return mul(a,rsqrtf(fmaxf(dot3(a,a),1.e-20f)));}
extern "C" __global__ void __raygen__main(){
 uint3 p=optixGetLaunchIndex();float x=(p.x+.5f)/params.width*2-1,y=1-(p.y+.5f)/params.height*2;
 float3 d=norm(add(params.w,add(mul(params.u,x),mul(params.v,y))));unsigned int r=0,g=0,b=0;
 optixTrace(params.handle,params.eye,d,.001f,1.e20f,0,255,OPTIX_RAY_FLAG_DISABLE_ANYHIT,0,1,0,r,g,b);
 params.image[p.y*params.width+p.x]=make_float4(__uint_as_float(r),__uint_as_float(g),__uint_as_float(b),1);
}
extern "C" __global__ void __miss__main(){optixSetPayload_0(__float_as_uint(.08f));optixSetPayload_1(__float_as_uint(.1f));optixSetPayload_2(__float_as_uint(.15f));}
extern "C" __global__ void __miss__shadow(){optixSetPayload_0(1);}
extern "C" __global__ void __closesthit__main(){
 const Cluster c=params.clusters[optixGetClusterId()];unsigned int tri=optixGetPrimitiveIndex();float2 uv=optixGetTriangleBarycentrics();
 const Vertex a=c.vertices[c.indices[tri*3]],b=c.vertices[c.indices[tri*3+1]],d=c.vertices[c.indices[tri*3+2]];
 float3 n=norm(make_float3(a.normal[0]*(1-uv.x-uv.y)+b.normal[0]*uv.x+d.normal[0]*uv.y,a.normal[1]*(1-uv.x-uv.y)+b.normal[1]*uv.x+d.normal[1]*uv.y,a.normal[2]*(1-uv.x-uv.y)+b.normal[2]*uv.x+d.normal[2]*uv.y));
 n=norm(optixTransformNormalFromObjectToWorldSpace(n));
 float3 ray=optixGetWorldRayDirection();if(dot3(n,ray)>0)n=mul(n,-1);
 float3 p=add(optixGetWorldRayOrigin(),mul(ray,optixGetRayTmax()));float3 light=norm(make_float3(.4f,.8f,.5f));unsigned int visible=0;
 optixTrace(params.handle,add(p,mul(n,fmaxf(.001f,optixGetRayTmax()*1.e-5f))),light,.001f,1.e20f,0,255,OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT|OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,0,1,1,visible);
 float value=.15f+.5f*fmaxf(dot3(n,light),0.f)*visible;
 optixSetPayload_0(__float_as_uint(value));optixSetPayload_1(__float_as_uint(value));optixSetPayload_2(__float_as_uint(value));
}
)CUDA";
int main(int argc,char** argv)try{
 if(argc<3)throw std::runtime_error("Usage: zeno_cluster_smoke asset.json output.ppm [frames] [poolMiB] [--static] [--loads N]");
 const int frames=argc>3?std::stoi(argv[3]):40;const uint64_t budget=uint64_t(argc>4?std::stoi(argv[4]):128)<<20;
 bool fixed=false;uint32_t loads=32;
 for(int i=5;i<argc;++i){if(std::strcmp(argv[i],"--static")==0)fixed=true;else if(std::strcmp(argv[i],"--loads")==0&&i+1<argc)loads=std::stoul(argv[++i]);else throw std::runtime_error("Unknown smoke option");}
 check(cudaFree(nullptr));check(optixInit());
 OptixDeviceContextOptions options{};options.validationMode=OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL;
 if(const char* validation=std::getenv("ZENO_OPTIX_VALIDATE");validation && validation[0]=='0')options.validationMode=OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_OFF;
 options.logCallbackFunction=[](unsigned int level,const char* tag,const char* message,void*){std::cerr<<"[OptiX "<<level<<" "<<tag<<"] "<<message<<std::endl;};options.logCallbackLevel=3;
 OptixDeviceContext context{};check(optixDeviceContextCreate(nullptr,&options,&context));
 {
 zeno::cluster::RuntimeSettings settings;settings.geometryBytes=budget;settings.clasBytes=budget;settings.loadsPerFrame=loads;
 zeno::cluster::Runtime runtime(argv[1],settings);
 auto asset=zeno::cluster::loadAsset(argv[1]);glm::vec3 lo(FLT_MAX),hi(-FLT_MAX);
 for(const auto& p:asset.placements){const auto m=glm::make_mat4(p.transform.data());for(uint32_t root:asset.geometries[p.geometry].roots){const auto& b=asset.geometries[p.geometry].groups[root].bounds;const auto c=glm::vec3(m*glm::vec4(b[0],b[1],b[2],1));float scale=0;for(int i=0;i<3;++i)scale+=glm::dot(glm::vec3(m[i]),glm::vec3(m[i]));const float r=b[3]*std::sqrt(scale);lo=glm::min(lo,c-glm::vec3(r));hi=glm::max(hi,c+glm::vec3(r));}}
 const glm::vec3 center=(lo+hi)*.5f;const float radius=std::max(glm::length(hi-lo)*.5f,.1f);
 nvrtcProgram program{};check(nvrtcCreateProgram(&program,shader,"cluster-smoke.cu",0,nullptr,nullptr));
 const std::string optixInclude=std::string("-I")+CLUSTER_OPTIX_INCLUDE,cudaInclude=std::string("-I")+CLUSTER_CUDA_INCLUDE;
 const char* flags[]={"--std=c++17","--gpu-architecture=compute_70",optixInclude.c_str(),cudaInclude.c_str(),"--generate-line-info"};
 const auto compile=nvrtcCompileProgram(program,5,flags);size_t logSize=0;nvrtcGetProgramLogSize(program,&logSize);std::string compileLog(logSize,'\0');nvrtcGetProgramLog(program,compileLog.data());if(compile)std::cerr<<compileLog;check(compile);
 size_t ptxSize=0;check(nvrtcGetPTXSize(program,&ptxSize));std::string ptx(ptxSize,'\0');check(nvrtcGetPTX(program,ptx.data()));check(nvrtcDestroyProgram(&program));
 OptixPipelineCompileOptions pipelineOptions{};pipelineOptions.allowClusteredGeometry=1;pipelineOptions.traversableGraphFlags=OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY;
 pipelineOptions.numPayloadValues=3;pipelineOptions.numAttributeValues=2;pipelineOptions.usesPrimitiveTypeFlags=OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;pipelineOptions.pipelineLaunchParamsVariableName="params";
 OptixModuleCompileOptions moduleOptions{};moduleOptions.optLevel=OPTIX_COMPILE_OPTIMIZATION_DEFAULT;moduleOptions.debugLevel=OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;
 OptixModule module{};char log[8192];size_t logBytes=sizeof(log);auto result=optixModuleCreate(context,&moduleOptions,&pipelineOptions,ptx.data(),ptx.size(),log,&logBytes,&module);if(result)std::cerr<<log;check(result);
 OptixProgramGroup groups[4]{};OptixProgramGroupOptions groupOptions{};
 for(int i=0;i<4;++i){OptixProgramGroupDesc desc{};if(i==0){desc.kind=OPTIX_PROGRAM_GROUP_KIND_RAYGEN;desc.raygen.module=module;desc.raygen.entryFunctionName="__raygen__main";}else if(i<3){desc.kind=OPTIX_PROGRAM_GROUP_KIND_MISS;desc.miss.module=module;desc.miss.entryFunctionName=i==1?"__miss__main":"__miss__shadow";}else{desc.kind=OPTIX_PROGRAM_GROUP_KIND_HITGROUP;desc.hitgroup.moduleCH=module;desc.hitgroup.entryFunctionNameCH="__closesthit__main";}logBytes=sizeof(log);check(optixProgramGroupCreate(context,&desc,1,&groupOptions,log,&logBytes,&groups[i]));}
 OptixPipeline pipeline{};OptixPipelineLinkOptions link{};link.maxTraceDepth=2;logBytes=sizeof(log);check(optixPipelineCreate(context,&pipelineOptions,&link,groups,4,log,&logBytes,&pipeline));
 OptixStackSizes sizes{};for(auto group:groups)check(optixUtilAccumulateStackSizes(group,&sizes,pipeline));unsigned int a,b,c;check(optixUtilComputeStackSizes(&sizes,2,0,0,&a,&b,&c));check(optixPipelineSetStackSize(pipeline,a,b,c,3));
 Record<int> records[4]{};for(int i=0;i<4;++i)check(optixSbtRecordPackHeader(groups[i],&records[i]));const auto sbtRecords=upload(records,4);
 OptixShaderBindingTable sbt{};sbt.raygenRecord=sbtRecords;sbt.missRecordBase=sbtRecords+sizeof(records[0]);sbt.missRecordCount=2;sbt.missRecordStrideInBytes=sizeof(records[0]);sbt.hitgroupRecordBase=sbtRecords+3*sizeof(records[0]);sbt.hitgroupRecordCount=1;sbt.hitgroupRecordStrideInBytes=sizeof(records[0]);
 Params params{};params.width=params.height=256;const auto image=upload<float4>(nullptr,256*256),deviceParams=upload<Params>(nullptr,1);params.image=reinterpret_cast<float4*>(image);
 double warmUpdateMs=0;uint32_t warmUpdates=0;
 for(int frame=0;frame<frames;++frame){
   const float distance=(fixed || frame<frames/3?3.f:frame<frames*2/3?.3f:3.f)*radius;
   const glm::vec3 eye=center+glm::vec3(distance*.7f,distance*.3f,distance);const auto w=glm::normalize(center-eye),u=glm::normalize(glm::cross(w,glm::vec3(0,1,0)))*.41421356f,v=glm::normalize(glm::cross(u,w))*.41421356f;
   const auto before=runtime.stats();
   const bool changed=runtime.update(context,0,{eye.x,eye.y,eye.z},256.f/(2.f*.41421356f));
   const auto after=runtime.stats();
   if(fixed && before.settled && after.staticSkips!=before.staticSkips+1)
       throw std::runtime_error("Settled static update did not take the early exit");
   if(after.staticSkips>before.staticSkips){
       if(changed || after.loads!=before.loads || after.evictions!=before.evictions || after.gasBuilds!=before.gasBuilds || after.iasBuilds!=before.iasBuilds || after.plans!=before.plans)
           throw std::runtime_error("Settled static update performed streaming/AS work");
       warmUpdateMs+=after.updateMs;++warmUpdates;
   }
   params.handle=runtime.handle();params.clusters=runtime.descriptors();params.eye=make_float3(eye.x,eye.y,eye.z);params.u=make_float3(u.x,u.y,u.z);params.v=make_float3(v.x,v.y,v.z);params.w=make_float3(w.x,w.y,w.z);
   check(cudaMemcpy(reinterpret_cast<void*>(deviceParams),&params,sizeof(params),cudaMemcpyHostToDevice));check(optixLaunch(pipeline,0,deviceParams,sizeof(params),&sbt,256,256,1));check(cudaStreamSynchronize(0));
 }
 std::vector<float4> pixels(256*256);check(cudaMemcpy(pixels.data(),reinterpret_cast<void*>(image),pixels.size()*sizeof(float4),cudaMemcpyDeviceToHost));
 std::ofstream output(argv[2],std::ios::binary);output<<"P6\n256 256\n255\n";for(auto p:pixels){unsigned char rgb[3];const float color[]={p.x,p.y,p.z};for(int i=0;i<3;++i)rgb[i]=uint8_t(std::clamp(std::sqrt(std::max(color[i],0.f))*255.f,0.f,255.f));output.write(reinterpret_cast<char*>(rgb),3);}if(!output)throw std::runtime_error("Cannot write smoke image");
 const auto stats=runtime.stats();std::cout<<"[CLAS smoke] loads="<<stats.loads<<" evictions="<<stats.evictions<<" plans="<<stats.plans<<" staticSkips="<<stats.staticSkips<<" settled="<<stats.settled<<" staticUpdateAverageMs="<<(warmUpdates?warmUpdateMs/warmUpdates:0)<<" image="<<argv[2]<<std::endl;
 check(cudaFree(reinterpret_cast<void*>(image)));check(cudaFree(reinterpret_cast<void*>(deviceParams)));check(cudaFree(reinterpret_cast<void*>(sbtRecords)));check(optixPipelineDestroy(pipeline));for(auto g:groups)check(optixProgramGroupDestroy(g));check(optixModuleDestroy(module));
 }
 check(optixDeviceContextDestroy(context));return 0;
}catch(const std::exception& e){std::cerr<<"[CLAS smoke] "<<e.what()<<std::endl;return 1;}
