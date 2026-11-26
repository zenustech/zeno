#pragma once
#include <map>
#include <set>
#include <string>
#include <vector>
#include <functional>
#include <stdexcept>
#include <tinygltf/json.hpp>

namespace zeno::scene {
// Compiled only when the logical scene changes. Per-frame invalidation follows
// reverse edges, including every parent of an instanced/shared child.
class Dependencies {
public:
    struct Group {std::string name;bool isStatic=false;};
private:
    struct Node {Group group;std::vector<size_t> parents;bool dirty=false;};
    std::vector<Node> nodes;
    std::map<std::pair<bool,std::string>,size_t> groupIds;
    std::map<std::string,std::vector<size_t>> bindings,matrices;
    std::map<std::string,std::vector<std::string>> geometryBindings;
    std::vector<size_t> pending;
    void invalidate(size_t id) {
        if(nodes[id].dirty)return;
        nodes[id].dirty=true;pending.push_back(id);
        for(auto parent:nodes[id].parents)invalidate(parent);
    }
public:
    void compile(const nlohmann::json& scene) {
        nodes.clear();groupIds.clear();bindings.clear();matrices.clear();geometryBindings.clear();pending.clear();
        if(scene.contains("BasicRenderInstances"))for(const auto& b:scene.at("BasicRenderInstances").items())
            geometryBindings[b.value().value("Geom",std::string{})].push_back(b.key());
        for(bool isStatic:{false,true}) {
            const std::string groups=isStatic?"StaticRenderGroups":"DynamicRenderGroups";
            const std::string entries=isStatic?"StaticEntries":"DynamicEntries";
            if(!scene.contains(groups) || !scene.contains(entries))continue;
            std::map<std::string,size_t> ids;
            for(const auto& item:scene.at(groups).items()) {
                ids[item.key()]=nodes.size();groupIds[{isStatic,item.key()}]=nodes.size();nodes.push_back({{item.key(),isStatic},{},false});
            }
            const size_t top=nodes.size();groupIds[{isStatic,groups}]=top;nodes.push_back({{groups,isStatic},{},false});
            auto edge=[&](size_t parent,const std::string& child,const nlohmann::json& transforms) {
                const auto group=ids.find(child);
                if(group==ids.end())bindings[child].push_back(parent);
                else nodes[group->second].parents.push_back(parent);
                for(const auto& key:transforms)matrices[key.get<std::string>()].push_back(parent);
            };
            for(const auto& item:scene.at(groups).items())for(const auto& child:item.value().items())
                edge(ids.at(item.key()),child.key(),child.value());
            for(const auto& entry:scene.at(entries).items())edge(top,entry.key(),entry.value());
        }
        // Reject cycles before invalidation so malformed graphs cannot recurse
        // indefinitely. This walk is structural work, never a camera update.
        std::vector<unsigned char> state(nodes.size());
        std::function<void(size_t)> visit=[&](size_t id) {
            if(state[id]==1)throw std::runtime_error("Cycle in render groups: "+nodes[id].group.name);
            if(state[id]==2)return;
            state[id]=1;for(auto parent:nodes[id].parents)visit(parent);state[id]=2;
        };
        for(size_t id=0;id<nodes.size();++id)visit(id);
        for(size_t id=0;id<nodes.size();++id)invalidate(id);
    }
    bool references(const std::string& geometry) const {return geometryBindings.count(geometry)!=0;}
    void groupChanged(const std::string& key,bool isStatic) {
        const auto group=groupIds.find({isStatic,key});if(group!=groupIds.end())invalidate(group->second);
    }
    std::vector<Group> ancestors(const std::string& geometry) const {
        std::set<size_t> seen;
        std::function<void(size_t)> visit=[&](size_t id) {
            if(!seen.insert(id).second)return;
            for(auto parent:nodes[id].parents)visit(parent);
        };
        const auto source=geometryBindings.find(geometry);
        if(source!=geometryBindings.end())for(const auto& key:source->second) {
            const auto binding=bindings.find(key);
            if(binding!=bindings.end())for(auto id:binding->second)visit(id);
        }
        std::vector<Group> result;result.reserve(seen.size());
        for(auto id:seen)result.push_back(nodes[id].group);
        return result;
    }
    void bindingChanged(const std::string& key) {
        auto it=bindings.find(key);if(it!=bindings.end())for(auto id:it->second)invalidate(id);
    }
    void geometryChanged(const std::string& key) {
        auto it=geometryBindings.find(key);if(it!=geometryBindings.end())for(const auto& binding:it->second)bindingChanged(binding);
    }
    void matrixChanged(const std::string& key) {
        auto it=matrices.find(key);if(it!=matrices.end())for(auto id:it->second)invalidate(id);
    }
    std::vector<Group> takeDirty() {
        std::vector<Group> dirty;dirty.reserve(pending.size());
        for(auto id:pending){dirty.push_back(nodes[id].group);nodes[id].dirty=false;}
        pending.clear();return dirty;
    }
    bool empty() const {return pending.empty();}
};
} // namespace zeno::scene
