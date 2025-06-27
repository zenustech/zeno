#pragma once

#include <zenovis/Scene.h>
#include <zeno/utils/MapStablizer.h>
#include <zeno/utils/PolymorphicMap.h>
#include <zeno/utils/disable_copy.h>
#include <zeno/core/IObject.h>
#include <tinygltf/json.hpp>
#include <string>
#include <memory>
#include <map>

namespace zenovis {
using Json = nlohmann::json;

struct ObjectsManager : zeno::disable_copy {
    zeno::MapStablizer<zeno::PolymorphicMap<std::map<
        std::string, std::shared_ptr<zeno::IObject>>>> objects;

    std::unordered_map<std::string, std::shared_ptr<zeno::IObject>> cached_mesh;
    std::map<std::string, std::shared_ptr<zeno::IObject>> lightObjects;
    bool needUpdateLight = true;

    template <class T = void>
    auto pairs() const {
        return objects.pairs<T>();
    }

    template <class T = void>
    auto pairsShared() const {
        return objects.pairsShared<T>();
    }

    ObjectsManager();
    ~ObjectsManager();
    void clear_objects();
    bool load_objects(std::map<std::string, std::shared_ptr<zeno::IObject>> const &objs, std::string& runtype);

    std::optional<zeno::IObject*> get(std::string nid);
    std::map<std::string, std::shared_ptr<zeno::IObject>> ObjectsManager::objs_filter(std::map<std::string, std::shared_ptr<zeno::IObject>> const &objs);
    Json staticSceneTree;
    Json staticSceneDescriptor;
    Json dynamicSceneTree;
    Json dynamicSceneDescriptor;

    std::string str_staticSceneTree;
    std::string str_staticSceneDescriptor;
    std::string str_dynamicSceneTree;
    std::string str_dynamicSceneDescriptor;
    void update_scene_tree(const std::string& str);
    void update_scene_descriptor(const std::string& str);
};

}
