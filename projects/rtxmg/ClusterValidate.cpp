#include "ClusterRuntime.h"
#include "ClusterSource.h"
#include "ClusterAttributes.h"
#include <cstdlib>
#include <cuda_runtime.h>
#include <glm/gtc/type_ptr.hpp>
#include <iostream>
#include <fstream>
#include <nvrtc.h>
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>
#include <set>

using namespace zeno::cluster;
static void check(cudaError_t r) {
    if (r)
        throw std::runtime_error(cudaGetErrorString(r));
}
static void check(OptixResult r) {
    if (r)
        throw std::runtime_error("OptiX " + std::to_string(r));
}
static void check(nvrtcResult r) {
    if (r)
        throw std::runtime_error(nvrtcGetErrorString(r));
}
struct Buffer {
    CUdeviceptr p{};
    explicit Buffer(size_t n) { check(cudaMalloc(reinterpret_cast<void **>(&p), n)); }
    ~Buffer() { cudaFree(reinterpret_cast<void *>(p)); }
    Buffer(const Buffer &) = delete;
    template <class T> void upload(const std::vector<T> &v) {
        check(
            cudaMemcpy(reinterpret_cast<void *>(p), v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
    }
};
struct Hit {
    float distance, nx, ny, nz;
    uint32_t instance, invalid, shadow;
};
struct Params {
    OptixTraversableHandle handle;
    const zeno_cluster_device::Cluster *clusters;
    float3 eye, u, v, w;
    Hit *hits;
    uint32_t width, height;
    float3 light;
    uint32_t traceShadow;
    OptixTraversableHandle mirror;
    Hit *mirrorHits;
    const Hit* shadowHits;
};
template <class T> struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) Record {
    char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};
static const char *shader = R"CUDA(
#include <optix.h>
#include <optix_device.h>
struct Vertex{float position[3],normal[3];};
struct Cluster{const Vertex* vertices;const unsigned char* indices;unsigned int lod,geometry,group,triangles;const void* attributes;const unsigned short* materials;};
struct Hit {float distance,nx,ny,nz;unsigned int instance,invalid,shadow;};
struct Params {OptixTraversableHandle handle;const Cluster* clusters;float3 eye,u,v,w;Hit* hits;unsigned int width,height;float3 light;unsigned int traceShadow;OptixTraversableHandle mirror;Hit* mirrorHits;const Hit* shadowHits;};
extern "C" {__constant__ Params params;}
static __device__ float3 sub(float3 a,float3 b){return make_float3(a.x-b.x,a.y-b.y,a.z-b.z);}
static __device__ float3 cross3(float3 a,float3 b){return make_float3(a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x);}
static __device__ float3 norm(float3 a){float k=rsqrtf(fmaxf(a.x*a.x+a.y*a.y+a.z*a.z,1.e-30f));return make_float3(a.x*k,a.y*k,a.z*k);}
extern "C" __global__ void __raygen__main(){
 uint3 p=optixGetLaunchIndex();unsigned int index=p.y*params.width+p.x;
 float x=(p.x+.5f)/params.width*2-1,y=1-(p.y+.5f)/params.height*2;
 float3 d=norm(make_float3(params.w.x+params.u.x*x+params.v.x*y,params.w.y+params.u.y*x+params.v.y*y,params.w.z+params.u.z*x+params.v.z*y));
 unsigned int result=index;params.hits[index]={-1,0,0,0,~0u,0};
 float3 origin=params.eye;
 if(params.mirror){
   unsigned int mirrorRay=1;
   optixTrace(params.mirror,origin,d,.001f,1.e20f,0,255,OPTIX_RAY_FLAG_DISABLE_ANYHIT,0,1,0,result,mirrorRay);
   Hit h=params.hits[index];params.mirrorHits[index]=h;
   params.hits[index]={-1,0,0,0,~0u,0,0};
   if(h.distance<0)return;
   float3 n=make_float3(h.nx,h.ny,h.nz);
   float dot=n.x*d.x+n.y*d.y+n.z*d.z;
   float3 incident=d;
   d=norm(make_float3(d.x-2*dot*n.x,d.y-2*dot*n.y,d.z-2*dot*n.z));
   float sign=dot<0?1.f:-1.f,epsilon=fmaxf(.0001f,h.distance*1.e-6f);
   origin=make_float3(origin.x+incident.x*h.distance+n.x*sign*epsilon,origin.y+incident.y*h.distance+n.y*sign*epsilon,origin.z+incident.z*h.distance+n.z*sign*epsilon);
 }
 unsigned int sceneRay=0;
 optixTrace(params.handle,origin,d,.001f,1.e20f,0,255,OPTIX_RAY_FLAG_DISABLE_ANYHIT,0,1,0,result,sceneRay);
 if(params.traceShadow && params.hits[index].distance>0){
   Hit h=params.shadowHits?params.shadowHits[index]:params.hits[index];float3 l=norm(params.light);
   float sign=h.nx*l.x+h.ny*l.y+h.nz*l.z>=0?1.f:-1.f;
   float epsilon=fmaxf(.0001f,h.distance*1.e-6f);
   float3 shadowOrigin=make_float3(origin.x+d.x*h.distance+h.nx*sign*epsilon,origin.y+d.y*h.distance+h.ny*sign*epsilon,origin.z+d.z*h.distance+h.nz*sign*epsilon);
   optixTrace(params.handle,shadowOrigin,l,epsilon,1.e20f,0,255,OPTIX_RAY_FLAG_ENFORCE_ANYHIT|OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT|OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,0,1,0,result,sceneRay);
 }
}
extern "C" __global__ void __miss__main(){}
extern "C" __global__ void __anyhit__shadow(){params.hits[optixGetPayload_0()].shadow=1;optixTerminateRay();}
extern "C" __global__ void __closesthit__main(){
 float3 v[3];unsigned int invalid=0;
 if(params.clusters && !optixGetPayload_1()){
   optixGetTriangleVertexData(v);
   unsigned int id=optixGetClusterId();if(id==OPTIX_CLUSTER_ID_INVALID)invalid=1;
   else {const Cluster c=params.clusters[id];unsigned int t=optixGetPrimitiveIndex();
     for(unsigned int j=0;j<3;++j){const Vertex a=c.vertices[c.indices[t*3+j]];
       for(unsigned int k=0;k<3;++k)if(fabsf(a.position[k]-(&v[j].x)[k])>1.e-5f*fmaxf(1.f,fabsf(a.position[k])))invalid=1;
     }
   }
 }else optixGetTriangleVertexData(optixGetGASTraversableHandle(),optixGetPrimitiveIndex(),optixGetSbtGASIndex(),0,v);
 float3 n=norm(optixTransformNormalFromObjectToWorldSpace(cross3(sub(v[1],v[0]),sub(v[2],v[0]))));
 params.hits[optixGetPayload_0()]={optixGetRayTmax(),n.x,n.y,n.z,optixGetInstanceId(),invalid};
}
)CUDA";

// Compare multisets, including winding and degenerate/duplicate source faces.
using Triangle = std::array<uint32_t, 9>;
static Triangle triangle(const Vertex *v, const uint32_t *index) {
    std::array<std::array<uint32_t, 3>, 3> p{};
    for (int j = 0; j < 3; ++j)
        for (int k = 0; k < 3; ++k) {
            float f = v[index[j]].position[k];
            if (f == 0)
                f = 0;
            std::memcpy(&p[j][k], &f, 4);
        }
    Triangle best{};
    best.fill(~0u);
    // Compare complete rotations, including equal-position vertices in a
    // degenerate face. Reversing winding is intentionally not equivalent.
    for (int first = 0; first < 3; ++first) {
        Triangle rotated{};
        for (int j = 0; j < 3; ++j)
            for (int k = 0; k < 3; ++k)
                rotated[j * 3 + k] = p[(first + j) % 3][k];
        best = std::min(best, rotated);
    }
    return best;
}
static void validateSource(const fs::path &manifest, const Asset &asset) {
    const auto meta = readJson(manifest);
    const auto path = fs::path(meta.at("source").get<std::string>());
    const auto gltf = readJson(path);
    Source source{gltf, path.parent_path(), {}};
    for (size_t g = 0; g < asset.geometries.size(); ++g) {
        std::vector<Triangle> expected, actual;
        for (const auto &prim :
             gltf.at("meshes").at(meta["meshes"][g]["sourceMesh"].get<size_t>()).at("primitives")) {
            auto positions = source.vectors(prim["attributes"]["POSITION"]);
            auto indices = source.indices(prim["indices"]);
            std::vector<Vertex> vertices(positions.size());
            for (size_t i = 0; i < positions.size(); ++i)
                std::memcpy(vertices[i].position, glm::value_ptr(positions[i]), 12);
            for (size_t i = 0; i < indices.size(); i += 3)
                expected.push_back(triangle(vertices.data(), indices.data() + i));
        }
        const auto &geometry = asset.geometries[g];
        for (uint32_t id = 0; id < geometry.groups.size(); ++id)
            if (geometry.groups[id].depth == 0) {
                auto page = readPage(geometry, id);
                for (const auto &c : page.clusters)
                    for (uint32_t i = 0; i < c.triangleCount; ++i) {
                        uint32_t index[3];
                        for (int j = 0; j < 3; ++j)
                            index[j] = c.vertexOffset + page.indices[c.indexOffset + i * 3 + j];
                        actual.push_back(triangle(page.vertices.data(), index));
                    }
            }
        std::sort(expected.begin(), expected.end());
        std::sort(actual.begin(), actual.end());
        if (expected != actual)
            throw std::runtime_error("Finest baked triangles differ from original source at geometry " +
                                     std::to_string(g));
        std::cout << "[reference source] geometry=" << g << " triangles=" << expected.size() << " exact=1\n";
        source.decoded.clear();
    }
}

// Compare source-corner UVs and tangent frames together with their material
// slots; independent position-only hashes cannot detect attribute permutation.
using AttributeTriangle=std::array<uint32_t,19>;
static AttributeTriangle attributeTriangle(const Vertex* vertices,const zeno_cluster_device::ShadingVertex* attributes,
                                          const uint32_t* indices,uint16_t material) {
    std::array<std::array<uint32_t,6>,3> corners{};
    for(int j=0;j<3;++j) {
        const auto index=indices[j];const float values[5]={vertices[index].position[0],vertices[index].position[1],vertices[index].position[2],attributes[index].uv[0],attributes[index].uv[1]};
        for(int k=0;k<5;++k){const float value=values[k]==0?0.f:values[k];std::memcpy(&corners[j][k],&value,4);}
        corners[j][5]=attributes[index].tangent;
    }
    AttributeTriangle best;best.fill(~0u);
    for(int first=0;first<3;++first) {
        AttributeTriangle next{};next[18]=material;
        for(int j=0;j<3;++j)for(int k=0;k<6;++k)next[j*6+k]=corners[(j+first)%3][k];
        best=std::min(best,next);
    }
    return best;
}
static void validateSourceAttributes(const fs::path& manifest,const Asset& asset) {
    const auto meta=readJson(manifest);auto sourcePath=fs::u8path(meta.at("source").get<std::string>());
    if(!sourcePath.is_absolute())sourcePath=manifest.parent_path()/sourcePath;
    const auto gltf=readJson(sourcePath);Source source{gltf,sourcePath.parent_path(),{}};
    for(size_t g=0;g<asset.geometries.size();++g) {
        const auto& geometry=asset.geometries[g];
        if(geometry.rtxmgShard)throw std::runtime_error("Exact attribute comparison requires unquantized custom UV pages");
        std::vector<AttributeTriangle> expected,actual;uint32_t material=0;
        for(const auto& primitive:gltf.at("meshes").at(meta["meshes"][g]["sourceMesh"].get<size_t>()).at("primitives")) {
            const auto& attributes=primitive.at("attributes");const auto positions=source.vectors(attributes.at("POSITION"));
            const auto indices=source.indices(primitive.at("indices"));
            const auto normals=attributes.contains("NORMAL")?source.vectors(attributes.at("NORMAL")):std::vector<glm::vec3>(positions.size());
            const auto uv=attributes.contains("TEXCOORD_0")?source.attributes(attributes.at("TEXCOORD_0"),2):std::vector<std::array<float,4>>{};
            const auto tangents=attributes.contains("TANGENT")?source.attributes(attributes.at("TANGENT"),4):std::vector<std::array<float,4>>{};
            std::vector<Vertex> vertices(positions.size());std::vector<zeno_cluster_device::ShadingVertex> shading(positions.size());
            for(size_t i=0;i<positions.size();++i) {
                std::memcpy(vertices[i].position,glm::value_ptr(positions[i]),12);
                if(!uv.empty()){shading[i].uv[0]=uv[i][0];shading[i].uv[1]=uv[i][1];shading[i].tangent=0x40000000u;}
                if(!tangents.empty())shading[i].tangent|=packTangent(glm::value_ptr(normals[i]),tangents[i].data());
            }
            for(size_t i=0;i<indices.size();i+=3)expected.push_back(attributeTriangle(vertices.data(),shading.data(),indices.data()+i,uint16_t(material)));
            ++material;
        }
        std::ifstream stream(geometry.pages,std::ios::binary);
        for(uint32_t id=0;id<geometry.groups.size();++id)if(geometry.groups[id].depth==0) {
            const auto page=readPage(geometry,id,stream);
            if(page.attributes.size()!=page.vertices.size()||page.materials.size()*3!=page.indices.size())throw std::runtime_error("Missing finest shading attributes");
            for(const auto& cluster:page.clusters)for(uint32_t i=0;i<cluster.triangleCount;++i) {
                uint32_t indices[3];for(int k=0;k<3;++k)indices[k]=cluster.vertexOffset+page.indices[cluster.indexOffset+i*3+k];
                actual.push_back(attributeTriangle(page.vertices.data(),page.attributes.data(),indices,page.materials.at(cluster.indexOffset/3+i)));
            }
        }
        std::sort(expected.begin(),expected.end());std::sort(actual.begin(),actual.end());
        if(expected!=actual)throw std::runtime_error("Finest UV/tangent/material association differs at geometry "+std::to_string(g));
        std::cout<<"[reference attributes] geometry="<<g<<" triangles="<<actual.size()<<" exact=1"<<std::endl;
        source.decoded.clear();
    }
}

// The mirror is fixed at source resolution so this test isolates LOD of the
// reflected scene. It is an ordinary triangle GAS, independent of cluster data.
static OptixTraversableHandle buildMirror(OptixDeviceContext context, const fs::path &path,
                                         std::vector<std::unique_ptr<Buffer>> &keep) {
    const auto spec = readJson(path);
    const fs::path sourcePath = spec.at("source").get<std::string>();
    const auto gltf = readJson(sourcePath);
    Source source{gltf, sourcePath.parent_path(), {}};
    const auto matrix = glm::make_mat4(spec.at("matrix").get<std::array<float,16>>().data());
    std::vector<float3> vertices;
    std::vector<uint3> triangles;
    for (const auto &primitive : gltf.at("meshes").at(spec.at("sourceMesh").get<size_t>()).at("primitives")) {
        const auto positions = source.vectors(primitive.at("attributes").at("POSITION"));
        const auto indices = source.indices(primitive.at("indices"));
        const auto base = uint32_t(vertices.size());
        for (auto position : positions) {
            const auto p = glm::vec3(matrix * glm::vec4(position,1));
            vertices.push_back(make_float3(p.x,p.y,p.z));
        }
        for (size_t i=0;i<indices.size();i+=3)
            triangles.push_back(make_uint3(base+indices[i],base+indices[i+1],base+indices[i+2]));
    }
    auto vb=std::make_unique<Buffer>(vertices.size()*sizeof(float3));
    auto ib=std::make_unique<Buffer>(triangles.size()*sizeof(uint3));
    vb->upload(vertices); ib->upload(triangles);
    OptixBuildInput input{};
    input.type=OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    auto &array=input.triangleArray;
    array.vertexBuffers=&vb->p;array.numVertices=uint32_t(vertices.size());
    array.vertexFormat=OPTIX_VERTEX_FORMAT_FLOAT3;array.vertexStrideInBytes=sizeof(float3);
    array.indexBuffer=ib->p;array.numIndexTriplets=uint32_t(triangles.size());
    array.indexFormat=OPTIX_INDICES_FORMAT_UNSIGNED_INT3;array.indexStrideInBytes=sizeof(uint3);
    uint32_t flag=OPTIX_GEOMETRY_FLAG_NONE;array.flags=&flag;array.numSbtRecords=1;
    OptixAccelBuildOptions options{};options.operation=OPTIX_BUILD_OPERATION_BUILD;
    options.buildFlags=OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS|OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
    OptixAccelBufferSizes sizes{};
    check(optixAccelComputeMemoryUsage(context,&options,&input,1,&sizes));
    auto gas=std::make_unique<Buffer>(sizes.outputSizeInBytes);Buffer scratch(sizes.tempSizeInBytes);
    OptixTraversableHandle handle{};
    check(optixAccelBuild(context,0,&options,&input,1,scratch.p,sizes.tempSizeInBytes,gas->p,sizes.outputSizeInBytes,&handle,nullptr,0));
    check(cudaStreamSynchronize(0));
    OptixInstance instance{};
    instance.transform[0]=instance.transform[5]=instance.transform[10]=1;
    instance.traversableHandle=handle;instance.visibilityMask=255;
    auto instances=std::make_unique<Buffer>(sizeof(instance));instances->upload(std::vector<OptixInstance>{instance});
    input={};input.type=OPTIX_BUILD_INPUT_TYPE_INSTANCES;
    input.instanceArray.instances=instances->p;input.instanceArray.numInstances=1;
    options.buildFlags=OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
    check(optixAccelComputeMemoryUsage(context,&options,&input,1,&sizes));
    auto top=std::make_unique<Buffer>(sizes.outputSizeInBytes);Buffer topScratch(sizes.tempSizeInBytes);
    check(optixAccelBuild(context,0,&options,&input,1,topScratch.p,sizes.tempSizeInBytes,top->p,sizes.outputSizeInBytes,&handle,nullptr,0));
    check(cudaStreamSynchronize(0));
    keep.push_back(std::move(vb));keep.push_back(std::move(ib));keep.push_back(std::move(gas));
    keep.push_back(std::move(instances));keep.push_back(std::move(top));
    std::cout << "[reference mirror] sourceMesh=" << spec.at("sourceMesh") << " triangles=" << triangles.size() << std::endl;
    return handle;
}
int main(int argc, char **argv) try {
    if (argc < 2)
        throw std::runtime_error("Usage: zeno_cluster_validate asset.json [--source] [--finest] [--frames N] "
                                 "[--pool MiB] [--pixels W H] [--camera scene.json] [--camera-path path.json] "
                                 "[--shadow-dir X Y Z] [--strict-instances] [--output directory] "
                                 "[--mirror mirror.json] [--error pixels]");
    fs::path manifest = argv[1], cameraPath, cameraSequencePath, outputPath, mirrorPath;
    bool original = false, finest = false, interactive = false, sourceOnly = false;
    bool traceShadow = false, strictInstances = false, sourceAttributesOnly = false;
    float3 light = make_float3(1, 1, 0);
    int frames = 1, width = 1920, height = 1080;
    uint64_t pool = 256ull << 20;
    float pixelError=2;
    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--source")
            original = true;
        else if (a == "--source-only")
            original = sourceOnly = true;
        else if (a == "--source-attributes-only")
            sourceAttributesOnly = true;
        else if (a == "--interactive")
            interactive = true;
        else if (a == "--finest")
            finest = true;
        else if (a == "--strict-instances")
            strictInstances = true;
        else if (a == "--output")
            outputPath = argv[++i];
        else if (a == "--mirror")
            mirrorPath = argv[++i];
        else if (a == "--error")
            pixelError = std::stof(argv[++i]);
        else if (a == "--shadow-dir") {
            traceShadow = true;
            light.x = std::stof(argv[++i]);
            light.y = std::stof(argv[++i]);
            light.z = std::stof(argv[++i]);
            if (!(light.x*light.x+light.y*light.y+light.z*light.z>0))
                throw std::runtime_error("Shadow direction must be nonzero");
        }
        else if (a == "--frames")
            frames = std::stoi(argv[++i]);
        else if (a == "--pool")
            pool = std::stoull(argv[++i]) << 20;
        else if (a == "--pixels") {
            width = std::stoi(argv[++i]);
            height = std::stoi(argv[++i]);
        } else if (a == "--camera")
            cameraPath = argv[++i];
        else if (a == "--camera-path")
            cameraSequencePath = argv[++i];
        else
            throw std::runtime_error("Unknown validation option");
    }
    const Asset asset = loadAsset(manifest);
    if(sourceAttributesOnly){validateSourceAttributes(manifest,asset);return 0;}
    nlohmann::json cameraSequence;
    if (!cameraSequencePath.empty()) {
        if (!cameraPath.empty()) throw std::runtime_error("Use one camera source");
        cameraSequence = readJson(cameraSequencePath).at("views");
        if (!cameraSequence.is_array() || cameraSequence.size() < size_t(frames))
            throw std::runtime_error("Camera path needs one view per reference frame");
    }
    if (original)
        validateSource(manifest, asset);
    if (sourceOnly)
        return 0;
    check(cudaFree(nullptr));
    check(optixInit());
    OptixDeviceContextOptions co{};
    co.validationMode = std::getenv("ZENO_OPTIX_VALIDATE") && std::getenv("ZENO_OPTIX_VALIDATE")[0] == '0'
                            ? OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_OFF
                            : OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL;
    co.logCallbackFunction = [](unsigned int, const char *tag, const char *message, void *) {
        std::cerr << "[OptiX " << tag << "] " << message << '\n';
    };
    co.logCallbackLevel = 3;
    OptixDeviceContext context{};
    check(optixDeviceContextCreate(nullptr, &co, &context));
    const auto cachePath = fs::absolute(".build/rtxmg-zorah/optix-cache-reference");
    fs::create_directories(cachePath);
    check(optixDeviceContextSetCacheLocation(context, cachePath.string().c_str()));
    {
        RuntimeSettings settings;
        settings.geometryBytes = settings.clasBytes = pool;
        settings.loadsPerFrame = 4096;
        settings.pixelError = pixelError;
        if (finest)
            settings.pixelError = 1.e-12f;
        Runtime runtime(manifest, settings);
        nvrtcProgram program{};
        check(nvrtcCreateProgram(&program, shader, "cluster-reference.cu", 0, nullptr, nullptr));
        std::string oi = std::string("-I") + CLUSTER_OPTIX_INCLUDE,
                    ci = std::string("-I") + CLUSTER_CUDA_INCLUDE;
        const char *flags[] = {"--std=c++17", "--gpu-architecture=compute_70", oi.c_str(), ci.c_str(),
                               "--generate-line-info"};
        auto status = nvrtcCompileProgram(program, 5, flags);
        if (status) {
            size_t n;
            nvrtcGetProgramLogSize(program, &n);
            std::string l(n, '\0');
            nvrtcGetProgramLog(program, l.data());
            std::cerr << l;
        }
        check(status);
        size_t n;
        check(nvrtcGetPTXSize(program, &n));
        std::string ptx(n, '\0');
        check(nvrtcGetPTX(program, ptx.data()));
        check(nvrtcDestroyProgram(&program));
        OptixPipelineCompileOptions po{};
        po.allowClusteredGeometry = 1;
        po.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY;
        po.numPayloadValues = 2;
        po.numAttributeValues = 2;
        po.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;
        po.pipelineLaunchParamsVariableName = "params";
        OptixModuleCompileOptions mo{};
        mo.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
        OptixModule module{};
        char log[8192];
        size_t lb = sizeof(log);
        check(optixModuleCreate(context, &mo, &po, ptx.data(), ptx.size(), log, &lb, &module));
        OptixProgramGroup pg[3]{};
        OptixProgramGroupOptions go{};
        for (int i = 0; i < 3; ++i) {
            OptixProgramGroupDesc d{};
            if (!i) {
                d.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
                d.raygen = {module, "__raygen__main"};
            } else if (i == 1) {
                d.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
                d.miss = {module, "__miss__main"};
            } else {
                d.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
                d.hitgroup.moduleCH = module;
                d.hitgroup.entryFunctionNameCH = "__closesthit__main";
                d.hitgroup.moduleAH = module;
                d.hitgroup.entryFunctionNameAH = "__anyhit__shadow";
            }
            lb = sizeof(log);
            check(optixProgramGroupCreate(context, &d, 1, &go, log, &lb, &pg[i]));
        }
        OptixPipeline pipeline{};
        OptixPipelineLinkOptions lo{};
        lo.maxTraceDepth = 1;
        lb = sizeof(log);
        check(optixPipelineCreate(context, &po, &lo, pg, 3, log, &lb, &pipeline));
        OptixStackSizes stack{};
        for (auto g : pg)
            check(optixUtilAccumulateStackSizes(g, &stack, pipeline));
        uint32_t x, y, z;
        check(optixUtilComputeStackSizes(&stack, 1, 0, 0, &x, &y, &z));
        check(optixPipelineSetStackSize(pipeline, x, y, z, 3));
        std::vector<Record<int>> records(3);
        for (int i = 0; i < 3; ++i)
            check(optixSbtRecordPackHeader(pg[i], &records[i]));
        Buffer rec(records.size() * sizeof(records[0]));
        rec.upload(records);
        OptixShaderBindingTable sbt{};
        sbt.raygenRecord = rec.p;
        sbt.missRecordBase = rec.p + sizeof(records[0]);
        sbt.missRecordCount = 1;
        sbt.missRecordStrideInBytes = sizeof(records[0]);
        sbt.hitgroupRecordBase = rec.p + sizeof(records[0]) * 2;
        sbt.hitgroupRecordCount = 1;
        sbt.hitgroupRecordStrideInBytes = sizeof(records[0]);
        const size_t rays = size_t(width) * height;
        Buffer output(rays * sizeof(Hit)), dp(sizeof(Params));
        std::unique_ptr<Buffer> mirrorOutput;
        std::vector<std::unique_ptr<Buffer>> mirrorStorage;
        const auto mirror = mirrorPath.empty() ? OptixTraversableHandle(0) : buildMirror(context,mirrorPath,mirrorStorage);
        if (mirror) mirrorOutput=std::make_unique<Buffer>(rays*sizeof(Hit));
        glm::vec3 center(0), eye, cameraUp(0, 1, 0);
        float radius = 0, fov = 45;
        for (const auto &p : asset.placements) {
            auto m = glm::make_mat4(p.transform.data());
            auto b = asset.geometries[p.geometry].groups[asset.geometries[p.geometry].roots[0]].bounds;
            center += glm::vec3(m * glm::vec4(b[0], b[1], b[2], 1));
        }
        center /= float(asset.placements.size());
        for (const auto &p : asset.placements) {
            auto m = glm::make_mat4(p.transform.data());
            for (auto r : asset.geometries[p.geometry].roots) {
                const auto &b = asset.geometries[p.geometry].groups[r].bounds;
                radius = std::max(
                    radius, glm::length(glm::vec3(m * glm::vec4(b[0], b[1], b[2], 1)) - center) +
                                b[3] * std::max({glm::length(glm::vec3(m[0])), glm::length(glm::vec3(m[1])),
                                                 glm::length(glm::vec3(m[2]))}));
            }
        }
        std::vector<ReferenceCut> previousCut;
        uint64_t previousSceneVersion = 0, previousCutVersion = 0;
        for (int frame = 0; frame < frames; ++frame) {
            eye = center + glm::vec3(radius * .7f, radius * .3f, radius) * (frame % 3 == 1   ? .3f
                                                                            : frame % 3 == 2 ? 1.5f
                                                                                             : 3.f);
            if (!cameraPath.empty() || !cameraSequencePath.empty()) {
                const auto view = cameraSequencePath.empty() ? readJson(cameraPath)["view"] : cameraSequence.at(frame);
                eye = glm::make_vec3(view["position"].get<std::array<float, 3>>().data());
                center = glm::make_vec3(view["lookat"].get<std::array<float, 3>>().data());
                if (view.contains("up"))
                    cameraUp = glm::make_vec3(view["up"].get<std::array<float, 3>>().data());
                fov = view["fov"];
                if (cameraSequencePath.empty()) eye.x += frame * .05f;
            }
            const float t = std::tan(fov * 3.14159265f / 360.f), projection = height / (2 * t);
            for (int i = 0; i < 20000; ++i) {
                runtime.update(context, 0, {eye.x, eye.y, eye.z}, projection, interactive);
                if (interactive || runtime.stats().settled)
                    break;
                if (i == 19999)
                    throw std::runtime_error("Streaming did not settle");
            }
            runtime.validateCut();
            auto cut = runtime.referenceCut();
            const auto revision = runtime.stats();
            if (frame && revision.sceneVersion <= previousSceneVersion + 1) {
                const bool same = cut.size() == previousCut.size() &&
                    std::equal(cut.begin(), cut.end(), previousCut.begin(), [](const auto& a, const auto& b) {
                        return a.geometry == b.geometry && a.placement == b.placement && a.groups == b.groups;
                    });
                if (same != (revision.cutVersion == previousCutVersion))
                    throw std::runtime_error("Cut revision disagrees with independently compared published geometry");
            }
            previousSceneVersion = revision.sceneVersion;
            previousCutVersion = revision.cutVersion;
            previousCut = cut;
            using CutKey = std::pair<uint32_t, std::map<uint32_t, std::vector<uint32_t>>>;
            std::map<CutKey, OptixTraversableHandle> cache;
            std::map<OptixTraversableHandle, uint64_t> triangleCounts;
            uint64_t referenceTriangles = 0, referenceClusters = 0;
            uint32_t minLod = ~0u, maxLod = 0;
            std::vector<std::unique_ptr<Buffer>> keep;
            std::vector<OptixInstance> instances;
            for (const auto &c : cut) {
                const CutKey key{c.geometry, c.groups};
                auto found = cache.find(key);
                OptixTraversableHandle gas{};
                if (found != cache.end())
                    gas = found->second;
                else {
                    std::vector<float3> vertices;
                    std::vector<uint3> indices;
                    for (const auto &[g, cs] : c.groups) {
                        auto page = readPage(asset.geometries[c.geometry], g);
                        for (auto id : cs) {
                            const auto &cl = page.clusters[id];
                            const uint32_t base = uint32_t(vertices.size());
                            for (uint32_t i = 0; i < cl.vertexCount; ++i) {
                                const auto &p = page.vertices[cl.vertexOffset + i];
                                vertices.push_back(make_float3(p.position[0], p.position[1], p.position[2]));
                            }
                            for (uint32_t i = 0; i < cl.triangleCount; ++i) {
                                const auto *ix = page.indices.data() + cl.indexOffset + i * 3;
                                indices.push_back(make_uint3(base + ix[0], base + ix[1], base + ix[2]));
                            }
                        }
                    }
                    auto vb = std::make_unique<Buffer>(vertices.size() * sizeof(float3)),
                         ib = std::make_unique<Buffer>(indices.size() * sizeof(uint3));
                    vb->upload(vertices);
                    ib->upload(indices);
                    OptixBuildInput in{};
                    in.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
                    in.triangleArray.vertexBuffers = &vb->p;
                    in.triangleArray.numVertices = uint32_t(vertices.size());
                    in.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
                    in.triangleArray.vertexStrideInBytes = sizeof(float3);
                    in.triangleArray.indexBuffer = ib->p;
                    in.triangleArray.numIndexTriplets = uint32_t(indices.size());
                    in.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
                    in.triangleArray.indexStrideInBytes = sizeof(uint3);
                    uint32_t flag = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;
                    in.triangleArray.flags = &flag;
                    in.triangleArray.numSbtRecords = 1;
                    OptixAccelBuildOptions o{};
                    o.operation = OPTIX_BUILD_OPERATION_BUILD;
                    o.buildFlags =
                        OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS | OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
                    OptixAccelBufferSizes s{};
                    check(optixAccelComputeMemoryUsage(context, &o, &in, 1, &s));
                    auto b = std::make_unique<Buffer>(s.outputSizeInBytes);
                    Buffer scratch(s.tempSizeInBytes);
                    check(optixAccelBuild(context, 0, &o, &in, 1, scratch.p, s.tempSizeInBytes, b->p,
                                          s.outputSizeInBytes, &gas, nullptr, 0));
                    check(cudaStreamSynchronize(0));
                    cache.emplace(key, gas);
                    triangleCounts.emplace(gas, indices.size());
                    keep.push_back(std::move(vb));
                    keep.push_back(std::move(ib));
                    keep.push_back(std::move(b));
                }
                referenceTriangles += triangleCounts.at(gas);
                for (const auto &[group, clusters] : c.groups) {
                    referenceClusters += clusters.size();
                    const auto depth = asset.geometries[c.geometry].groups[group].depth;
                    minLod = std::min(minLod, depth);
                    maxLod = std::max(maxLod, depth);
                }
                OptixInstance instance{};
                const auto &m = asset.placements[c.placement].transform;
                for (int r = 0; r < 3; ++r)
                    for (int col = 0; col < 4; ++col)
                        instance.transform[r * 4 + col] = m[col * 4 + r];
                instance.instanceId = c.placement;
                instance.visibilityMask = 255;
                instance.traversableHandle = gas;
                instances.push_back(instance);
            }
            const auto cutStats = runtime.stats();
            if (cutStats.renderTriangles != referenceTriangles ||
                cutStats.renderClusters != referenceClusters || cutStats.minLod != minLod ||
                cutStats.maxLod != maxLod)
                throw std::runtime_error("GPU cut diagnostics differ from reference geometry");
            Buffer il(instances.size() * sizeof(OptixInstance));
            il.upload(instances);
            OptixBuildInput in{};
            in.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
            in.instanceArray.instances = il.p;
            in.instanceArray.numInstances = uint32_t(instances.size());
            OptixAccelBuildOptions o{};
            o.operation = OPTIX_BUILD_OPERATION_BUILD;
            o.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
            OptixAccelBufferSizes s{};
            check(optixAccelComputeMemoryUsage(context, &o, &in, 1, &s));
            Buffer top(s.outputSizeInBytes), scratch(s.tempSizeInBytes);
            OptixTraversableHandle reference{};
            check(optixAccelBuild(context, 0, &o, &in, 1, scratch.p, s.tempSizeInBytes, top.p,
                                  s.outputSizeInBytes, &reference, nullptr, 0));
            const auto w = glm::normalize(center - eye),
                       u = glm::normalize(glm::cross(w, cameraUp)) * t * float(width) /
                           float(height),
                       v = glm::normalize(glm::cross(u, w)) * t;
            Params p{runtime.handle(),
                     runtime.descriptors(),
                     make_float3(eye.x, eye.y, eye.z),
                     make_float3(u.x, u.y, u.z),
                     make_float3(v.x, v.y, v.z),
                     make_float3(w.x, w.y, w.z),
                     reinterpret_cast<Hit *>(output.p),
                     uint32_t(width),
                     uint32_t(height), light, uint32_t(traceShadow),mirror,
                     mirrorOutput ? reinterpret_cast<Hit*>(mirrorOutput->p) : nullptr,nullptr};
            std::vector<Hit> actual(rays), expected(rays);
            for (int pass = 0; pass < 2; ++pass) {
                if (pass) {
                    p.handle = reference;
                    p.clusters = nullptr;
                }
                check(cudaMemcpy(reinterpret_cast<void *>(dp.p), &p, sizeof(p), cudaMemcpyHostToDevice));
                check(optixLaunch(pipeline, 0, dp.p, sizeof(p), &sbt, width, height, 1));
                check(cudaMemcpy((pass ? expected : actual).data(), reinterpret_cast<void *>(output.p),
                                 rays * sizeof(Hit), cudaMemcpyDeviceToHost));
            }
            const auto rawExpected=expected;
            size_t rawShadow=0;
            for(size_t i=0;i<rays;++i)rawShadow+=actual[i].shadow!=expected[i].shadow;
            if(traceShadow) {
                // Coplanar/coincident primary hits can legitimately select
                // different normals. Compare secondary visibility using the
                // same primary position/normal, not two different spawn rays.
                Buffer canonical(rays*sizeof(Hit));canonical.upload(actual);
                p.shadowHits=reinterpret_cast<const Hit*>(canonical.p);
                check(cudaMemcpy(reinterpret_cast<void*>(dp.p),&p,sizeof(p),cudaMemcpyHostToDevice));
                check(optixLaunch(pipeline,0,dp.p,sizeof(p),&sbt,width,height,1));
                check(cudaMemcpy(expected.data(),reinterpret_cast<void*>(output.p),rays*sizeof(Hit),cudaMemcpyDeviceToHost));
            }
            size_t hits = 0, mask = 0, depth = 0, normal = 0, instance = 0, invalid = 0, shadow = 0, occluded = 0;
            float maxRelativeDepth = 0;
            for (size_t i = 0; i < rays; ++i) {
                const auto a = actual[i], b = expected[i];
                invalid += a.invalid;
                if ((a.distance < 0) != (b.distance < 0)) {
                    ++mask;
                    continue;
                }
                if (b.distance < 0)
                    continue;
                ++hits;
                const float relative = std::abs(a.distance - b.distance) / std::max(1.f, b.distance);
                maxRelativeDepth = std::max(maxRelativeDepth, relative);
                if (relative > 2.e-5f)
                    ++depth;
                if (a.nx * b.nx + a.ny * b.ny + a.nz * b.nz < .999f)
                    ++normal;
                if (a.instance != b.instance)
                    ++instance;
                shadow += a.shadow != b.shadow;
                occluded += a.shadow != 0;
            }
            std::cout << "[reference rays] frame=" << frame << " pixels=" << width << 'x' << height
                      << " rays=" << rays << " hits=" << hits << " mask=" << mask << " depth=" << depth
                      << " normal=" << normal << " instance=" << instance << " descriptor=" << invalid
                      << " shadow=" << shadow << " rawShadow="<<rawShadow<<" occluded=" << occluded
                      << " maxRelativeDepth=" << maxRelativeDepth << std::endl;
            if (!outputPath.empty()) {
                fs::create_directories(outputPath);
                const auto name = "frame-" + std::to_string(frame);
                std::ofstream binary(outputPath / (name + ".hits"), std::ios::binary);
                binary.write(reinterpret_cast<const char*>(actual.data()), actual.size()*sizeof(Hit));
                if (!binary) throw std::runtime_error("Cannot write ray output");
                std::ofstream referenceBinary(outputPath/(name+".reference.hits"),std::ios::binary);
                referenceBinary.write(reinterpret_cast<const char*>(rawExpected.data()),rawExpected.size()*sizeof(Hit));
                if(!referenceBinary)throw std::runtime_error("Cannot write reference ray output");
                if (mirror) {
                    std::vector<Hit> primary(rays);
                    check(cudaMemcpy(primary.data(),reinterpret_cast<void*>(mirrorOutput->p),rays*sizeof(Hit),cudaMemcpyDeviceToHost));
                    std::ofstream mirrorBinary(outputPath / (name + ".mirror.hits"), std::ios::binary);
                    mirrorBinary.write(reinterpret_cast<const char*>(primary.data()),rays*sizeof(Hit));
                    if (!mirrorBinary) throw std::runtime_error("Cannot write mirror ray output");
                }
                Json report{{"width",width},{"height",height},{"stride",sizeof(Hit)},
                    {"fields",{"distance","nx","ny","nz","instance","invalid","shadow"}},
                    {"render_triangles",cutStats.renderTriangles},{"min_lod",minLod},{"max_lod",maxLod},
                    {"pixel_error",cutStats.effectivePixelError},{"mask_mismatches",mask},
                    {"depth_mismatches",depth},{"normal_mismatches",normal},{"instance_mismatches",instance},
                    {"shadow_mismatches",shadow},{"raw_shadow_mismatches",rawShadow},
                    {"canonical_shadow_origins",traceShadow},{"occluded",occluded},{"mirror",mirrorPath.string()}};
                std::ofstream metadata(outputPath / (name + ".json"));
                metadata << report.dump(2);
                if (!metadata) throw std::runtime_error("Cannot write ray metadata");
            }
            if (!hits || mask || depth || invalid || shadow || (strictInstances && instance))
                throw std::runtime_error("CLAS/reference ray comparison failed");
        }
        const auto result = runtime.stats();
        std::cout << "[reference result] versions=" << result.sceneVersion << " cutVersions=" << result.cutVersion << " loads=" << result.loads
                  << " evictions=" << result.evictions << " compactions=" << result.gasCompactions
                  << " settled=" << result.settled << std::endl;
        if (std::getenv("ZENO_CLAS_FORCE_GAS_COMPACTION") && frames >= 16 && interactive &&
            !result.gasCompactions)
            throw std::runtime_error("Forced GAS bank compaction was not exercised");
        check(optixPipelineDestroy(pipeline));
        for (auto g : pg)
            check(optixProgramGroupDestroy(g));
        check(optixModuleDestroy(module));
    }
    check(optixDeviceContextDestroy(context));
    return 0;
} catch (const std::exception &e) {
    std::cerr << "[reference] " << e.what() << std::endl;
    return 1;
}
