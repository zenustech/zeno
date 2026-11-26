#pragma once
#include <array>
#include <atomic>
#include <cmath>
#include <functional>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>
#include <tinygltf/json.hpp>

namespace zeno::mega {
struct PlacementInvalidation {
    std::atomic<bool> dirty{true};
    PlacementInvalidation()=default;
    PlacementInvalidation(PlacementInvalidation&& other) noexcept:dirty(other.dirty.load()){}
    PlacementInvalidation& operator=(PlacementInvalidation&& other) noexcept {dirty=other.dirty.load();return *this;}
    void operator=(bool value){dirty=value;}
    bool exchange(bool value){return dirty.exchange(value);}
};
using Affine=std::array<float,12>;
inline constexpr Affine Identity{1,0,0,0,0,1,0,0,0,0,1,0};
inline Affine compose(const Affine& parent,const Affine& local) {
    Affine out{};
    for(unsigned r=0;r<3;++r) {
        for(unsigned c=0;c<3;++c)
            for(unsigned k=0;k<3;++k)out[r*4+c]+=parent[r*4+k]*local[k*4+c];
        out[r*4+3]=parent[r*4+3];
        for(unsigned k=0;k<3;++k)out[r*4+3]+=parent[r*4+k]*local[k*4+3];
    }
    return out;
}
inline double determinant(const Affine& a) {
    return double(a[0])*(double(a[5])*a[10]-double(a[6])*a[9])-
           double(a[1])*(double(a[4])*a[10]-double(a[6])*a[8])+
           double(a[2])*(double(a[4])*a[9]-double(a[5])*a[8]);
}
inline bool visible(const Affine& a) {
    for(float v:a)if(!std::isfinite(v))throw std::runtime_error("Nonfinite scene transform");
    return determinant(a)!=0.; // Scene visibility is encoded as a zero matrix.
}
inline Affine inverse(const Affine& a) {
    const double d=determinant(a);
    if(d==0. || !std::isfinite(d))throw std::runtime_error("Singular scene transform");
    Affine b{};
    b[0]=float((double(a[5])*a[10]-double(a[6])*a[9])/d);
    b[1]=float((double(a[2])*a[9]-double(a[1])*a[10])/d);
    b[2]=float((double(a[1])*a[6]-double(a[2])*a[5])/d);
    b[4]=float((double(a[6])*a[8]-double(a[4])*a[10])/d);
    b[5]=float((double(a[0])*a[10]-double(a[2])*a[8])/d);
    b[6]=float((double(a[2])*a[4]-double(a[0])*a[6])/d);
    b[8]=float((double(a[4])*a[9]-double(a[5])*a[8])/d);
    b[9]=float((double(a[1])*a[8]-double(a[0])*a[9])/d);
    b[10]=float((double(a[0])*a[5]-double(a[1])*a[4])/d);
    for(unsigned r=0;r<3;++r)
        b[r*4+3]=-(b[r*4]*a[3]+b[r*4+1]*a[7]+b[r*4+2]*a[11]);
    for(float v:b)if(!std::isfinite(v))throw std::runtime_error("Unrepresentable inverse scene transform");
    return b;
}
inline std::array<float,3> point(const Affine& a,const std::array<float,3>& p) {
    return {a[0]*p[0]+a[1]*p[1]+a[2]*p[2]+a[3],
            a[4]*p[0]+a[5]*p[1]+a[6]*p[2]+a[7],
            a[8]*p[0]+a[9]*p[1]+a[10]*p[2]+a[11]};
}
struct PlacementSet {
    bool referenced=false;
    std::vector<Affine> world;
};
// Resolve world-space placements for streamed LOD selection. This does not
// change the retained acceleration hierarchy or flatten its physical parents.
template<class MatrixLookup>
std::map<std::string,PlacementSet> resolvePlacements(const nlohmann::json& scene,
    const std::set<std::string>& resources,MatrixLookup&& matrixLookup) {
    std::map<std::string,PlacementSet> result;
    for(const auto& name:resources)result[name];
    std::map<std::string,std::string> bindings;
    if(scene.contains("BasicRenderInstances"))
        for(const auto& item:scene.at("BasicRenderInstances").items()) {
            const auto name=item.value().value("Geom",std::string{});
            if(resources.count(name)){bindings[item.key()]=name;result[name].referenced=true;}
        }
    for(const char* category:{"Static","Dynamic"}) {
        const auto groupKey=std::string(category)+"RenderGroups",entryKey=std::string(category)+"Entries";
        if(!scene.contains(groupKey) || !scene.contains(entryKey))continue;
        const auto& groups=scene.at(groupKey);
        std::map<std::string,bool> reachable;
        std::set<std::string> visiting;
        std::function<bool(const std::string&)> reaches=[&](const std::string& name) {
            if(bindings.count(name))return true;
            const auto known=reachable.find(name);if(known!=reachable.end())return known->second;
            if(!groups.contains(name))return false;
            if(!visiting.insert(name).second)throw std::runtime_error("Cycle in scene groups: "+name);
            bool yes=false;
            for(const auto& child:groups.at(name).items())yes=reaches(child.key())||yes;
            visiting.erase(name);return reachable[name]=yes;
        };
        auto eachMatrix=[&](const nlohmann::json& keys,const Affine& parent,auto&& visit) {
            if(keys.empty()){visit(parent);return;}
            for(const auto& key:keys) {
                const auto* list=matrixLookup(key.template get<std::string>());
                if(!list)continue;
                for(const auto& local:*list) {
                    auto world=compose(parent,local);
                    if(visible(world))visit(world);
                }
            }
        };
        std::function<void(const std::string&,const Affine&)> walk=[&](const std::string& name,const Affine& world) {
            const auto binding=bindings.find(name);
            if(binding!=bindings.end()) {
                auto& placements=result.at(binding->second).world;
                if(placements.size()>=1048576)throw std::runtime_error("Too many streamed scene placements");
                placements.push_back(world);return;
            }
            if(!reaches(name))return;
            for(const auto& child:groups.at(name).items())if(reaches(child.key()))
                eachMatrix(child.value(),world,[&](const Affine& m){walk(child.key(),m);});
        };
        for(const auto& entry:scene.at(entryKey).items())if(reaches(entry.key()))
            eachMatrix(entry.value(),Identity,[&](const Affine& m){walk(entry.key(),m);});
    }
    // Legacy direct VIEW references remain root objects. A referenced but hidden
    // node must stay hidden instead of falling back to a root placement.
    for(auto& [_,p]:result)if(!p.referenced)p.world.push_back(Identity);
    return result;
}
} // namespace zeno::mega
