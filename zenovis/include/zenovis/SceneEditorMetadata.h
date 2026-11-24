#pragma once

#include <tinygltf/json.hpp>
#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace zenovis {

// The render hierarchy and its matrix buffers are registered separately. Keep
// editor descriptions cheap during playback and decode their numeric maps only
// when an editor command needs them.
class SceneEditorMetadata {
    using Json = nlohmann::json;
    struct Source {
        std::string text, key;
        Json topology;
        bool present=false, topologyDirty=false, detailsDirty=false;
        uint64_t topologyVersion=0;
    };
    std::array<Source,2> sources;
    std::array<uint64_t,2> publishedVersions{};
    std::vector<std::string> publishedLights;
    std::string publishedOwner;
    bool published=false;

    static std::string owner(const std::string& key) {
        // Frame-cache keys append a frame/stamp pair. Changing that pair does
        // not replace the graph node that owns the outline.
        const auto last=key.rfind(':');
        if(last!=std::string::npos && last!=0) {
            const auto frame=key.rfind(':',last-1);
            if(frame!=std::string::npos)return key.substr(0,frame);
        }
        return key;
    }
    static void refreshTopology(Source& source) {
        if(!source.topologyDirty)return;
        auto next=Json::parse(source.text,[](int depth,Json::parse_event_t event,Json& value) {
            // The outline consumes the hierarchy, not the potentially very
            // large instance matrices or instance-ID maps.
            return !(depth==1 && event==Json::parse_event_t::key &&
                (value=="node_to_matrix" || value=="node_to_id"));
        });
        if(next!=source.topology) {
            source.topology=std::move(next);
            ++source.topologyVersion;
        }
        source.topologyDirty=false;
    }
public:
    void stage(bool isStatic,std::string_view text,std::string key={}) {
        auto& source=sources[isStatic?0:1];
        // Static editor metadata follows the existing LoadAsset lifecycle.
        if(isStatic && source.present)return;
        source.key=std::move(key);
        if(!source.present || std::string_view(source.text)!=text) {
            // Own one reusable snapshot: producers may reuse or mutate their
            // StringObject after registration, so a borrowed view is unsafe.
            source.text.assign(text.data(),text.size());
            source.present=true;source.topologyDirty=true;source.detailsDirty=true;
        }
    }
    bool refreshDetails(bool isStatic,Json& output) {
        auto& source=sources[isStatic?0:1];
        if(!source.present)return false;
        const bool changed=source.detailsDirty || output.is_null();
        if(changed) {
            auto next=Json::parse(source.text);
            output=std::move(next);source.detailsDirty=false;
        }
        if(!isStatic)output["node_key"]=source.key;
        return changed;
    }
    std::optional<Json> outline(const std::vector<std::string>& lights,bool force=false) {
        for(auto& source:sources)refreshTopology(source);
        const std::array<uint64_t,2> versions{{sources[0].topologyVersion,sources[1].topologyVersion}};
        const auto currentOwner=owner(sources[1].key);
        if(!force && published && versions==publishedVersions && lights==publishedLights && currentOwner==publishedOwner)
            return {};
        Json message;
        for(size_t index=0;index<sources.size();++index) {
            const auto& source=sources[index];if(!source.present)continue;
            Json tree;
            tree["root_name"]=source.topology.at("root_name");
            tree["scene_tree"]=source.topology.at("scene_tree");
            if(index==1)tree["node_key"]=source.key;
            message[index==0?"StaticSceneTree":"DynamicSceneTree"]=std::move(tree);
        }
        message["Lights"]=lights;message["MessageType"]="SceneTree";
        publishedVersions=versions;publishedLights=lights;publishedOwner=currentOwner;published=true;
        return message;
    }
    void invalidateOutline() {published=false;}
    void clear() {*this={};}
};

} // namespace zenovis
