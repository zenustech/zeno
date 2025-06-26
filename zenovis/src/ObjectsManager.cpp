#include <zenovis/ObjectsManager.h>
#include <zeno/types/UserData.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/ListObject.h>
#include <zenovis/bate/IGraphic.h>
#include <zeno/core/IObject.h>
#include <zeno/utils/log.h>
#include <zeno/utils/fileio.h>
#include <zeno/utils/string.h>
#include <set>
#include <cstdlib>
#include <ctime>
#include "zeno/extra/SceneAssmbler.h"
#include "zeno/types/PrimitiveObject.h"

namespace zenovis {

ObjectsManager::ObjectsManager() = default;
ObjectsManager::~ObjectsManager() = default;

std::map<std::string, std::shared_ptr<zeno::IObject>> ObjectsManager::scene_tree_to_descriptor(std::map<std::string, std::shared_ptr<zeno::IObject>> const &objs) {
    for (auto const &[key, obj] : objs) {
        if (obj->userData().get2("ResourceType", std::string("")) == "SceneTree") {
            auto json_str = obj->userData().get2<std::string>("json", "");
            if (json_str.size()) {
                Json json = Json::parse(json_str);
                if (json["type"] == "static") {
                    staticSceneTree = key;
                }
                else if (json["type"] == "dynamic") {
                    dynamicSceneTree = key;
                }
            }
        }
    }
    std::map<std::string, std::shared_ptr<zeno::IObject>> output;
    auto staticSceneList = std::make_shared<zeno::ListObject>();
    auto dynamicSceneList = std::make_shared<zeno::ListObject>();

    auto get_index = [] (std::string const &key) -> int {
        std::string count_str = key.substr(key.find(":LIST")+5);
        count_str = count_str.substr(0, count_str.find(':'));
        auto count = std::stoi(count_str);
        return count;
    };
    if (staticSceneTree.size()) {
        auto static_size = get_index(staticSceneTree) + 1;
        staticSceneList->arr.resize(static_size);
    }
    if (dynamicSceneTree.size()) {
        auto dynamic_size = get_index(dynamicSceneTree) + 1;
        dynamicSceneList->arr.resize(dynamic_size);
    }

    std::string static_prefix = staticSceneTree.substr(0, staticSceneTree.find(':'));
    std::string dynamic_prefix = dynamicSceneTree.substr(0, dynamicSceneTree.find(':'));
    for (auto const &[key, obj] : objs) {
        if (staticSceneTree.size() && zeno::starts_with(key, static_prefix)) {
            auto index = get_index(key);
            if (index >= staticSceneList->arr.size()) {
                zeno::log_info("index >= staticSceneList->arr.size(): {}, {}", index, staticSceneList->arr.size());
            }
            staticSceneList->arr[index] = obj;
        }
        else if (dynamicSceneTree.size() && zeno::starts_with(key, dynamic_prefix)) {
            auto index = get_index(key);
            if (index >= dynamicSceneList->arr.size()) {
                zeno::log_info("index >= dynamicSceneList->arr.size(): {}, {}", index, dynamicSceneList->arr.size());
            }
            dynamicSceneList->arr[index] = obj;
        }
        else {
            output[key] = obj;
        }
    }
    auto get_format = [] (std::string const &key) -> std::string {
        auto prefix = key.substr(0, key.find(":LIST")+5);
        std::string count_str = key.substr(key.find(":LIST")+5);
        auto postfix = count_str.substr(count_str.find(':'));
        return prefix+"{}"+postfix;
    };
    Json scene_descriptor_json;

    zeno::log_error("staticSceneTree: {}", staticSceneTree);
    if (!staticSceneList->arr.empty()) {
        auto json_str = staticSceneList->arr[staticSceneList->arr.size() - 1]->userData().get2<std::string>("json", "");
        auto scene_tree = zeno::get_scene_tree_from_list(staticSceneList);
        auto new_scene_tree = scene_tree->root_rename("SRG", std::nullopt);
        auto json = Json::parse(json_str);
        auto scene = json["flattened"]? new_scene_tree->to_flatten_structure(true) : new_scene_tree->to_layer_structure(true);
        auto format_str = get_format(staticSceneTree);
        for (auto i = 1; i < scene->arr.size() - 2; i++) {
            auto key = zeno::format(format_str, i);
            output[key] = scene->arr[i];
            auto obj_name = scene->arr[i]->userData().get2("ObjectName", std::string("None"));
//            zeno::log_error("obj_name: {}", obj_name);
        }
        auto scene_str = scene->arr[scene->arr.size() - 2]->userData().get2<std::string>("Scene");
        auto filename = zeno::replace_all(staticSceneTree, ":", "") + ".json";
        zeno::file_put_content("E:/fuck/"+filename, scene_str);
        auto scene_descriptor = Json::parse(scene_str);
        scene_descriptor_json["StaticRenderGroups"] = scene_descriptor["StaticRenderGroups"];
        scene_descriptor_json["BasicRenderInstances"].update(scene_descriptor["BasicRenderInstances"]);
    }
//    zeno::log_error("dynamicSceneTree: {}", dynamicSceneTree);
//    if (!dynamicSceneList->arr.empty()) {
//        auto json_str = dynamicSceneList->arr[dynamicSceneList->arr.size() - 1]->userData().get2<std::string>("json", "");
//        auto scene_tree = zeno::get_scene_tree_from_list(dynamicSceneList);
//        auto new_scene_tree = scene_tree->root_rename("DRG", std::nullopt);
//        auto json = Json::parse(json_str);
//        auto scene = json["flattened"]? new_scene_tree->to_flatten_structure(false) : new_scene_tree->to_layer_structure(false);
//        auto format_str = get_format(dynamicSceneTree);
//        for (auto i = 1; i < scene->arr.size() - 2; i++) {
//            auto key = zeno::format(format_str, i);
//            output[key] = scene->arr[i];
//        }
//        auto scene_str = scene->arr[scene->arr.size() - 2]->userData().get2<std::string>("Scene");
//        auto filename = zeno::replace_all(dynamicSceneTree, ":", "") + ".json";
////        zeno::file_put_content("E:/fuck/"+filename, scene_str);
//        auto scene_descriptor = Json::parse(scene_str);
//        scene_descriptor_json["DynamicRenderGroups"] = scene_descriptor["DynamicRenderGroups"];
//        scene_descriptor_json["BasicRenderInstances"].update(scene_descriptor["BasicRenderInstances"]);
//    }

    {
        auto scene_descriptor = std::make_shared<zeno::PrimitiveObject>();
        auto &ud = scene_descriptor->userData();
        ud.set2("ResourceType", std::string("SceneDescriptor"));
        ud.set2("Scene", std::string(scene_descriptor_json.dump()));
        std::srand(std::time(0));
        auto json_key = zeno::format("GeneratedJson:{}", std::rand());
        zeno::file_put_content("E:/fuck/Generated.json", ud.get2<std::string>("Scene"));
        output[json_key] = scene_descriptor;
    }

    return output;
}

bool ObjectsManager::load_objects(std::map<std::string, std::shared_ptr<zeno::IObject>> const &objs, std::string& runtype) {
//    auto objs = scene_tree_to_descriptor(objs_);
    bool inserted = false;
    auto ins = objects.insertPass();

    bool changed = false;
    for (auto const &[key, obj] : objs) {
        if (ins.may_emplace(key)) {
            changed = true;
            auto const &ud = obj->userData();
            if (
                    ud.get2<std::string>("ResourceType", "") == "Mesh"
                    && ud.has<std::string>("ObjectName")
            ) {
                auto obj_name = obj->userData().get2<std::string>("ObjectName");
                cached_mesh[obj_name] = obj;
            }
        }
    }
    if(changed){
        if (runtype == "LoadAsset" || runtype == "RunAll" || runtype == "RunLightCamera") {
            lightObjects.clear();
        }
    }
    for (auto const &[key, obj] : objs) {
        if (ins.may_emplace(key)) {
            std::shared_ptr<zeno::IObject> newobj = obj;

            if (auto prim_in = dynamic_cast<zeno::PrimitiveObject *>(newobj.get())) {
                auto isRealTimeObject = prim_in->userData().get2<int>("isRealTimeObject", 0);
                if(isRealTimeObject){
                    //printf("loading light object %s\n", key.c_str());
                    lightObjects[key] = newobj;
                }
            }
            ins.try_emplace(key, std::move(newobj));
            inserted = true;
        }
    }

    return inserted;
}

void ObjectsManager::clear_objects() {
    objects.clear();
    lightObjects.clear();
}

std::optional<zeno::IObject* > ObjectsManager::get(std::string nid) {
    for (auto &[key, ptr]: this->pairs()) {
        if (key != nid && ptr->userData().get2<std::string>("ObjectName", key) != nid) {
            continue;
        }
        return ptr;
    }
    if (cached_mesh.count(nid)) {
        return cached_mesh[nid].get();
    }

    return std::nullopt;
}

}
