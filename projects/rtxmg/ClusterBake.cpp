#include "ClusterCache.h"
#include "ClusterSource.h"
#include <meshoptimizer.h>
#define CLUSTERLOD_IMPLEMENTATION
#include <clusterlod.h>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <fstream>
#include <iostream>
#include <unordered_map>
#include <unordered_set>
#include <cstring>
#include <functional>
#include <atomic>
#include <future>
#include <mutex>
#include <map>
#include <cwchar>

using namespace zeno::cluster;
using Json = nlohmann::json;
namespace fs = std::filesystem;

// Weld positions first so attribute seams do not become artificial borders.
// Lock open and nonmanifold edges; ordinary interior edges have two uses.
static std::vector<unsigned char> openBorderLocks(const std::vector<glm::vec3>& positions,
                                                const std::vector<uint32_t>& indices) {
    std::vector<unsigned int> remap(positions.size());
    meshopt_generatePositionRemap(remap.data(),glm::value_ptr(positions[0]),positions.size(),sizeof(glm::vec3));
    std::vector<uint64_t> edges; edges.reserve(indices.size());
    for(size_t i=0;i<indices.size();i+=3) for(size_t e=0;e<3;++e) {
        uint32_t a=remap[indices[i+e]],b=remap[indices[i+(e+1)%3]];
        if(a!=b) edges.push_back((uint64_t(std::min(a,b))<<32)|std::max(a,b));
    }
    std::sort(edges.begin(),edges.end());
    std::vector<unsigned char> locks(positions.size());
    for(size_t begin=0,end;begin<edges.size();begin=end) {
        for(end=begin+1;end<edges.size() && edges[end]==edges[begin];++end) {}
        if(end-begin!=2) locks[uint32_t(edges[begin]>>32)]=locks[uint32_t(edges[begin])]=meshopt_SimplifyVertex_Lock;
    }
    for(size_t i=0;i<locks.size();++i) locks[i]=locks[remap[i]];
    return locks;
}

struct BakeContext {
    std::ofstream& stream;
    const std::vector<glm::vec3>& positions;
    const std::vector<glm::vec3>& normals;
    Json& groups;
};
static int outputGroup(void* context, clodGroup group, const clodCluster* clusters, size_t count) {
    auto& c = *static_cast<BakeContext*>(context);
    Page page;
    Json refined = Json::array();
    uint32_t triangles=0;
    for(size_t i=0;i<count;++i) {
        const auto& in=clusters[i];
        unsigned int vertices[MaxVertices]; unsigned char indices[MaxTriangles*3];
        if(in.vertex_count>MaxVertices || in.index_count>MaxTriangles*3) throw std::runtime_error("Baker exceeded cluster limits");
        const size_t vertexCount=clodLocalIndices(vertices,indices,in.indices,in.index_count);
        Cluster cl{uint32_t(page.vertices.size()),uint32_t(page.indices.size()),uint16_t(vertexCount),uint16_t(in.index_count/3),in.refined};
        page.clusters.push_back(cl); refined.push_back(in.refined); triangles+=cl.triangleCount;
        for(size_t j=0;j<vertexCount;++j) {
            Vertex v{};
            std::memcpy(v.position,glm::value_ptr(c.positions.at(vertices[j])),12);
            std::memcpy(v.normal,glm::value_ptr(c.normals.at(vertices[j])),12);
            page.vertices.push_back(v);
        }
        page.indices.insert(page.indices.end(),indices,indices+in.index_count);
    }
    const uint64_t offset=uint64_t(c.stream.tellp());
    const auto bytes=writePage(c.stream,page);
    c.groups.push_back({{"offset",offset},{"bytes",bytes},{"vertices",page.vertices.size()},
                       {"triangles",triangles},{"depth",group.depth},{"refined",std::move(refined)},
                       {"bounds",{group.simplified.center[0],group.simplified.center[1],group.simplified.center[2],group.simplified.radius,group.simplified.error}}});
    return int(c.groups.size()-1);
}

int main(int argc,char** argv) try {
    fs::path source,selection,output;
    int meshOnly=-1,maxMeshes=0,jobs=1;bool includeInstances=false;
    bool preserveFolds=false,dilateBorders=false;
    bool fallbackSloppy=true,permissive=true;
    bool preserveOpenBorders=false;
    int errorClamped=-1;
    for(int i=1;i<argc;++i) {
        const std::string arg=argv[i];
        if(i+1>=argc) throw std::runtime_error("Missing command argument");
        if(arg=="--source") source=argv[++i];
        else if(arg=="--selection") selection=argv[++i];
        else if(arg=="--output") output=argv[++i];
        else if(arg=="--mesh") meshOnly=std::stoi(argv[++i]);
        else if(arg=="--max-meshes") maxMeshes=std::stoi(argv[++i]);
        else if(arg=="--jobs") jobs=std::stoi(argv[++i]);
        else if(arg=="--include-instances") includeInstances=std::stoi(argv[++i])!=0;
        else if(arg=="--preserve-folds") preserveFolds=std::stoi(argv[++i])!=0;
        else if(arg=="--dilate-borders") dilateBorders=std::stoi(argv[++i])!=0;
        else if(arg=="--fallback-sloppy") fallbackSloppy=std::stoi(argv[++i])!=0;
        else if(arg=="--permissive") permissive=std::stoi(argv[++i])!=0;
        else if(arg=="--error-clamped") errorClamped=std::stoi(argv[++i])!=0;
        else if(arg=="--preserve-open-borders") preserveOpenBorders=std::stoi(argv[++i])!=0;
        else throw std::runtime_error("Unknown bake option: "+arg);
    }
    if(source.empty() || output.empty()) throw std::runtime_error("Usage: zeno_cluster_bake --source scene.gltf --output cache-directory [--selection file] [--mesh id] [--max-meshes n]");
    source=fs::weakly_canonical(source); output=fs::weakly_canonical(output);
    // Source distributions, including their native caches, are immutable inputs.
    const auto sourceDirectory=source.parent_path();
    auto directoryPart=sourceDirectory.begin(),outputPart=output.begin();
    for(;directoryPart!=sourceDirectory.end() && outputPart!=output.end();++directoryPart,++outputPart) {
#ifdef _WIN32
        if(_wcsicmp(directoryPart->c_str(),outputPart->c_str())!=0) break;
#else
        if(*directoryPart!=*outputPart) break;
#endif
    }
    if(directoryPart==sourceDirectory.end())
        throw std::runtime_error("Bake output must be outside the source asset directory");
#if MESHOPTIMIZER_VERSION < 1030
    if(preserveFolds||dilateBorders||errorClamped>=0) throw std::runtime_error("Fold preservation, border dilation and error clamping controls require meshoptimizer 1.3 or newer");
#endif
    fs::create_directories(output);
    if(jobs<1||jobs>64)throw std::runtime_error("Bake jobs must be between 1 and 64");
    const Json scene=readJson(source);
    Source input{scene,source.parent_path(),{}};
    std::unordered_set<size_t> selectedNodes;
    if(!selection.empty()) {
        const auto selected=readJson(selection);
        for(size_t id:selected.at("selection").at("node_indices")) selectedNodes.insert(id);
    }
    else for(size_t i=0;i<input.scene.at("nodes").size();++i) {
        const auto& node=input.scene["nodes"][i];
        if(node.contains("mesh") && (includeInstances||!node.value("extensions",Json::object()).contains("EXT_mesh_gpu_instancing"))) selectedNodes.insert(i);
    }
    std::unordered_set<size_t> selectedMeshes;
    for(size_t id:selectedNodes) {
        const auto& node=input.scene.at("nodes").at(id);
        if(!includeInstances&&node.value("extensions",Json::object()).contains("EXT_mesh_gpu_instancing")) throw std::runtime_error("GPU instancing requires --include-instances 1");
        const size_t mesh=node.at("mesh"); if(meshOnly<0 || mesh==size_t(meshOnly)) selectedMeshes.insert(mesh);
    }
    std::vector<size_t> meshOrder(selectedMeshes.begin(),selectedMeshes.end()); std::sort(meshOrder.begin(),meshOrder.end());
    if(maxMeshes>0 && meshOrder.size()>size_t(maxMeshes)) meshOrder.resize(maxMeshes);
    std::map<std::string,std::vector<size_t>> aliases;
    for(size_t mesh:meshOrder)aliases[input.geometryKey(mesh)].push_back(mesh);
    std::unordered_map<size_t,size_t> canonical;
    meshOrder.clear();
    for(const auto& [key,ids]:aliases){
        size_t representative=ids.front();
        for(size_t id:ids)if(fs::exists(output/("mesh-"+std::to_string(id)+".json"))&&fs::exists(output/("mesh-"+std::to_string(id)+".pages"))){representative=id;break;}
        meshOrder.push_back(representative);for(size_t id:ids)canonical[id]=representative;
    }
    std::sort(meshOrder.begin(),meshOrder.end());
    uint64_t fingerprint=14695981039346656037ull;
    for(unsigned char byte:input.scene.dump()) {fingerprint^=byte; fingerprint*=1099511628211ull;}
    const Json bakeSettings{{"meshoptimizer",MESHOPTIMIZER_VERSION},{"maxVertices",MaxVertices},
        {"maxTriangles",MaxTriangles},{"partitionSize",8},{"fallbackSloppy",fallbackSloppy},
        {"permissive",permissive},{"errorClamped",errorClamped<0?MESHOPTIMIZER_VERSION>=1030:errorClamped!=0},
        {"normalWeight",0.5},{"preserveFolds",preserveFolds},{"dilateBorders",dilateBorders},
        {"preserveOpenBorders",preserveOpenBorders}};
    const std::string signature=std::to_string(fingerprint)+"-gray-clod-v2-"+bakeSettings.dump();
    Json asset{{"schema","zeno.rtxmg.asset"},{"version",CacheVersion},{"source",source.string()},
               {"sourceSignature",signature},{"bakeSettings",bakeSettings},{"meshes",Json::array()},{"nodes",Json::array()},{"gray",true},{"gpuInstancing",includeInstances}};
    std::unordered_map<size_t,size_t> geometryIndex;
    std::mutex logMutex;
    auto bakeMesh=[&](Source& input,size_t meshID) {
        const auto base="mesh-"+std::to_string(meshID);
        const fs::path metadata=output/(base+".json"),pages=output/(base+".pages");
        bool resume=false;
        if(fs::exists(metadata) && fs::exists(pages)) {
            const auto old=readJson(metadata);
            resume=old.value("signature","")==signature && old.value("pageBytes",uint64_t(0))==fs::file_size(pages);
        }
        if(!resume) {
            if(fs::space(output).available<(1ull<<30)) throw std::runtime_error("Less than 1 GiB free for cluster bake; completed meshes remain checkpointed");
            input.decoded.clear();
            std::ofstream stream(pages.string()+".tmp",std::ios::binary|std::ios::trunc);
            Json groups=Json::array();
            uint64_t lockedBoundaryVertices=0;
            const auto& mesh=input.scene.at("meshes").at(meshID);
            {std::lock_guard<std::mutex> lock(logMutex);std::cout<<"[cluster bake] mesh="<<meshID<<" name="<<mesh.value("name","")<<std::endl;}
            for(const auto& primitive:mesh.at("primitives")) {
                if(primitive.value("mode",4)!=4) throw std::runtime_error("Stage 1 only supports triangle meshes");
                const auto& attributes=primitive.at("attributes");
                auto positions=input.vectors(attributes.at("POSITION"));
                auto indices=input.indices(primitive.at("indices"));
                std::vector<glm::vec3> normals(positions.size(),glm::vec3(0));
                if(attributes.contains("NORMAL")) normals=input.vectors(attributes.at("NORMAL"));
                if(normals.size()!=positions.size()) throw std::runtime_error("Normal accessor count mismatch");
                for(uint32_t index:indices) if(index>=positions.size()) throw std::runtime_error("Source triangle index out of range");
                for(const auto& p:positions) for(int j=0;j<3;++j) if(!std::isfinite(p[j])) throw std::runtime_error("Non-finite source position");
                if(!attributes.contains("NORMAL")) {
                    for(size_t i=0;i<indices.size();i+=3) {
                        const auto n=glm::cross(positions[indices[i+1]]-positions[indices[i]],positions[indices[i+2]]-positions[indices[i]]);
                        for(int j=0;j<3;++j) normals[indices[i+j]]+=n;
                    }
                    for(auto& n:normals) if(glm::dot(n,n)>0) n=glm::normalize(n);
                }
                auto config=clodDefaultConfigRT(MaxTriangles); config.max_vertices=MaxVertices; config.partition_size=8;
                config.simplify_fallback_sloppy=fallbackSloppy;
                config.simplify_permissive=permissive;
#if MESHOPTIMIZER_VERSION >= 1030
                config.simplify_preserve_folds=preserveFolds;
                config.simplify_dilate_borders=dilateBorders;
                if(errorClamped>=0) config.simplify_error_clamped=errorClamped!=0;
#endif
                const float weights[3]={0.5f,0.5f,0.5f};
                clodMesh in{}; in.indices=indices.data(); in.index_count=indices.size(); in.vertex_count=positions.size();
                in.vertex_positions=glm::value_ptr(positions[0]); in.vertex_positions_stride=sizeof(glm::vec3);
                in.vertex_attributes=glm::value_ptr(normals[0]); in.vertex_attributes_stride=sizeof(glm::vec3);
                in.attribute_weights=weights; in.attribute_count=3;
                auto borderLocks=preserveOpenBorders?openBorderLocks(positions,indices):std::vector<unsigned char>{};
                in.vertex_lock=borderLocks.empty()?nullptr:borderLocks.data();
                lockedBoundaryVertices+=std::count(borderLocks.begin(),borderLocks.end(),meshopt_SimplifyVertex_Lock);
                BakeContext context{stream,positions,normals,groups};
                // outputGroup serializes positions immediately: border dilation
                // mutates this private vertex copy before producing coarser LODs.
                clodBuild(config,in,&context,outputGroup);
            }
            stream.close();
            std::unordered_set<size_t> children;
            for(const auto& g:groups) for(int id:g.at("refined")) if(id>=0) children.insert(size_t(id));
            Json roots=Json::array(); for(size_t i=0;i<groups.size();++i) if(!children.count(i)) roots.push_back(i);
            if(fs::exists(pages)) fs::remove(pages);
            fs::rename(pages.string()+".tmp",pages);
            Json result{{"version",CacheVersion},{"signature",signature},{"sourceMesh",meshID},
                        {"bakeSettings",bakeSettings},
                        {"lockedBoundaryVertices",lockedBoundaryVertices},
                        {"pages",pages.filename().string()},{"pageBytes",fs::file_size(pages)},{"pageCodec",4},
                        {"groups",std::move(groups)},{"roots",std::move(roots)}};
            writeJson(metadata.string()+".tmp",result);
            if(fs::exists(metadata)) fs::remove(metadata);
            fs::rename(metadata.string()+".tmp",metadata);
        }
        {std::lock_guard<std::mutex> lock(logMutex);std::cout<<"[cluster bake] completed mesh="<<meshID<<(resume?" resumed":"")<<std::endl;}
    };
    // Each worker decodes one mesh at a time; metadata remains shared/read-only.
    // Per-mesh checkpoints are independent, and manifest publication stays serial.
    std::atomic<size_t> next{0};std::atomic<bool> failed{false};
    std::vector<std::future<void>> workers;
    for(int i=0;i<jobs;++i)workers.push_back(std::async(std::launch::async,[&]{
        Source local{scene,source.parent_path(),{}};
        try {while(!failed){const size_t index=next.fetch_add(1);if(index>=meshOrder.size())break;bakeMesh(local,meshOrder[index]);}}
        catch(...){failed=true;throw;}
    }));
    std::exception_ptr error;
    for(auto& worker:workers)try{worker.get();}catch(...){if(!error)error=std::current_exception();}
    if(error)std::rethrow_exception(error);
    for(size_t meshID:meshOrder){
        geometryIndex[meshID]=asset["meshes"].size();
        asset["meshes"].push_back({{"sourceMesh",meshID},{"metadata","mesh-"+std::to_string(meshID)+".json"}});
    }
    const auto& nodes=input.scene.at("nodes");
    std::vector<int> parents(nodes.size(),-1),state(nodes.size(),0);
    std::vector<glm::mat4> world(nodes.size(),glm::mat4(1));
    for(size_t i=0;i<nodes.size();++i) for(size_t child:nodes[i].value("children",std::vector<size_t>{})) {
        if(child>=nodes.size() || parents[child]!=-1) throw std::runtime_error("Invalid glTF parent hierarchy");
        parents[child]=int(i);
    }
    std::function<glm::mat4(size_t)> transform=[&](size_t i)->glm::mat4 {
        if(state[i]==2) return world[i]; if(state[i]==1) throw std::runtime_error("Cyclic glTF hierarchy"); state[i]=1;
        const auto& n=nodes[i]; glm::mat4 local(1);
        if(n.contains("matrix")) {auto values=n["matrix"].get<std::array<float,16>>();std::memcpy(glm::value_ptr(local),values.data(),64);}
        else {
            const auto t=n.value("translation",std::array<float,3>{0,0,0}),s=n.value("scale",std::array<float,3>{1,1,1});
            const auto r=n.value("rotation",std::array<float,4>{0,0,0,1});
            local=glm::translate(glm::mat4(1),glm::vec3(t[0],t[1],t[2]))*glm::mat4_cast(glm::quat(r[3],r[0],r[1],r[2]))*glm::scale(glm::mat4(1),glm::vec3(s[0],s[1],s[2]));
        }
        world[i]=parents[i]<0?local:transform(parents[i])*local; state[i]=2; return world[i];
    };
    std::vector<size_t> nodeOrder(selectedNodes.begin(),selectedNodes.end()); std::sort(nodeOrder.begin(),nodeOrder.end());
    for(size_t id:nodeOrder) {
        const size_t sourceMesh=nodes[id].at("mesh"); if(!canonical.count(sourceMesh))continue;
        const auto geometry=geometryIndex.at(canonical.at(sourceMesh));
        const auto m=transform(id); std::array<float,16> values; std::memcpy(values.data(),glm::value_ptr(m),64);
        auto ext=nodes[id].value("extensions",Json::object());
        if(includeInstances&&ext.contains("EXT_mesh_gpu_instancing")){
            const auto& attributes=ext.at("EXT_mesh_gpu_instancing").at("attributes");
            size_t count=0;
            for(auto it=attributes.begin();it!=attributes.end();++it){const size_t n=scene.at("accessors").at(it.value().get<size_t>()).at("count");if(count&&count!=n)throw std::runtime_error("Instance accessor counts differ");count=n;}
            if(!count)throw std::runtime_error("Empty GPU instance attribute set");
            input.decoded.clear();
            auto translations=attributes.contains("TRANSLATION")?input.vectors(attributes.at("TRANSLATION")):std::vector<glm::vec3>(count,glm::vec3(0));
            auto scales=attributes.contains("SCALE")?input.vectors(attributes.at("SCALE")):std::vector<glm::vec3>(count,glm::vec3(1));
            auto rotations=attributes.contains("ROTATION")?input.rotations(attributes.at("ROTATION")):std::vector<glm::quat>(count,glm::quat(1,0,0,0));
            for(size_t instance=0;instance<count;++instance){
                const auto placed=m*glm::translate(glm::mat4(1),translations[instance])*glm::mat4_cast(rotations[instance])*glm::scale(glm::mat4(1),scales[instance]);
                std::memcpy(values.data(),glm::value_ptr(placed),64);
                asset["nodes"].push_back({{"mesh",geometry},{"sourceMesh",sourceMesh},{"sourceNode",id},{"sourceInstance",instance},{"matrix",values}});
            }
        }else asset["nodes"].push_back({{"mesh",geometry},{"sourceMesh",sourceMesh},{"sourceNode",id},{"matrix",values}});
    }
    if(asset["nodes"].empty()) throw std::runtime_error("No ordinary mesh placements selected");
    writeJson(output/"asset.json.tmp",asset);
    if(fs::exists(output/"asset.json")) fs::remove(output/"asset.json"); fs::rename(output/"asset.json.tmp",output/"asset.json");
    auto checked=loadAsset(output/"asset.json");
    for(const auto& geometry:checked.geometries) for(uint32_t root:geometry.roots) readPage(geometry,root);
    std::cout<<"[cluster bake] asset="<<(output/"asset.json")<<" geometries="<<checked.geometries.size()<<" placements="<<checked.placements.size()<<std::endl;
    return 0;
} catch(const std::exception& e) {std::cerr<<"[cluster bake] "<<e.what()<<std::endl;return 1;}
