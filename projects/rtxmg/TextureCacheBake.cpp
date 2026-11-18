#include "TextureTileCache.h"
#include <tinygltf/json.hpp>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <string>

int main(int argc,char** argv) try {
    namespace fs=std::filesystem;using namespace zeno::cluster::texturecache;
    fs::path trace,manifest,root;bool dry=false;uint64_t limit=8ull<<30;
    for(int i=1;i<argc;++i) {
        const std::string a=argv[i];
        if(a=="--trace" && i+1<argc)trace=fs::u8path(argv[++i]);
        else if(a=="--manifest" && i+1<argc)manifest=fs::u8path(argv[++i]);
        else if(a=="--output" && i+1<argc)root=fs::u8path(argv[++i]);
        else if(a=="--max-bytes" && i+1<argc)limit=std::stoull(argv[++i]);
        else if(a=="--dry-run")dry=true;
        else throw std::runtime_error("Usage: zeno_texture_cache_bake (--trace requests.json | --manifest asset.json) --output directory [--dry-run] [--max-bytes N]");
    }
    if((trace.empty()==manifest.empty())||root.empty())throw std::runtime_error("Specify exactly one of --trace or --manifest and an --output");
    nlohmann::json requests=nlohmann::json::array();
    if(!trace.empty()){std::ifstream input(trace);input>>requests;}
    else {
        nlohmann::json asset,gltf;std::ifstream input(manifest);input>>asset;
        auto source=fs::u8path(asset.at("source").get<std::string>());
        if(!source.is_absolute())source=manifest.parent_path()/source;
        std::ifstream scene(source);scene>>gltf;
        for(const auto& texture:gltf.at("textures")) {
            const auto path=source.parent_path()/fs::u8path(gltf.at("images").at(texture.at("source").get<size_t>()).at("uri").get<std::string>());
            const auto info=inspect(path);std::vector<uint32_t> levels;
            for(uint32_t l=0;l<info.levels;++l)levels.push_back(l);
            requests.push_back({{"path",path.u8string()},{"levels",levels}});
        }
    }
    struct Job {fs::path path;std::vector<uint32_t> levels;};std::vector<Job> jobs;uint64_t upper=0;
    std::map<fs::path,std::set<uint32_t>> sources;
    for(const auto& entry:requests) {
        const auto levels=entry.at("levels").get<std::vector<uint32_t>>();
        auto& selected=sources[fs::absolute(fs::u8path(entry.at("path").get<std::string>())).lexically_normal()];
        selected.insert(levels.begin(),levels.end());
    }
    for(const auto& entry:sources) {
        const auto& source=entry.first;auto info=inspect(source);auto chosen=entry.second;
        if(auto old=Reader::open(root,source,info))for(uint32_t l=0;l<info.levels;++l)if(old->contains(l))chosen.insert(l);
        std::vector<uint32_t> levels(chosen.begin(),chosen.end());upper+=estimate(info,levels);jobs.push_back({source,std::move(levels)});
    }
    std::cout<<"[texture cache estimate] textures="<<jobs.size()<<" upperBytes="<<upper<<" limitBytes="<<limit<<std::endl;
    if(upper>limit)throw std::runtime_error("Texture cache estimate exceeds the explicit disk limit");
    if(dry)return 0;
    fs::create_directories(root);
    BakeStats total{};
    for(size_t i=0;i<jobs.size();++i) {
        // Compression ratios vary widely. Reserve the worst-case size of the
        // current atomic file, rather than requiring the whole uncompressed
        // corpus to fit at once. A stopped bake preserves completed textures.
        if(fs::space(root).available<estimate(inspect(jobs[i].path),jobs[i].levels)+(16ull<<30))
            throw std::runtime_error("Next texture could leave less than 16 GiB free; completed cache files remain usable");
        const auto s=bake(jobs[i].path,root,jobs[i].levels);total.sourceBytes+=s.sourceBytes;total.decodedBytes+=s.decodedBytes;
        total.tiles+=s.tiles;total.fileBytes+=s.fileBytes;
        if(i%20==0 || i+1==jobs.size())std::cout<<"[texture cache bake] complete="<<i+1<<"/"<<jobs.size()<<" bytes="<<total.fileBytes<<std::endl;
    }
    std::cout<<nlohmann::json({{"textures",jobs.size()},{"sourceBytes",total.sourceBytes},{"decodedBytes",total.decodedBytes},
                             {"tiles",total.tiles},{"cacheBytes",total.fileBytes}}).dump()<<std::endl;
    return 0;
}catch(const std::exception& e){std::cerr<<"[texture cache] "<<e.what()<<'\n';return 1;}
