#pragma once

#include <zenovis/Scene.h>
#include <zeno/utils/MapStablizer.h>
#include <zeno/utils/PolymorphicMap.h>
#include <zeno/utils/disable_copy.h>
#include <zeno/core/IObject.h>
#include <string>
#include <memory>
#include <map>
#include <tinygltf/json.hpp>
using Json = nlohmann::json;

namespace zenovis {

struct ObjectsManager : zeno::disable_copy {
    zeno::MapStablizer<zeno::PolymorphicMap<std::map<
        std::string, std::shared_ptr<zeno::IObject>>>> objects;

    std::unordered_map<std::string, std::shared_ptr<zeno::IObject>> cached_mesh;
    std::map<std::string, std::shared_ptr<zeno::IObject>> lightObjects;
    bool needUpdateLight = true;
    std::string staticSceneTree;
    std::string dynamicSceneTree;

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
    std::map<std::string, std::shared_ptr<zeno::IObject>> scene_tree_to_descriptor(std::map<std::string, std::shared_ptr<zeno::IObject>> const &objs);

    std::optional<zeno::IObject*> get(std::string nid);
};

}
