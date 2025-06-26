#include <zenovis/ObjectsManager.h>
#include <zeno/types/UserData.h>
#include <zeno/types/PrimitiveObject.h>
#include <zenovis/bate/IGraphic.h>
#include <zeno/core/IObject.h>
#include <zeno/utils/log.h>
#include <set>
#include "zeno/utils/fileio.h"

namespace zenovis {

ObjectsManager::ObjectsManager() = default;
ObjectsManager::~ObjectsManager() = default;

std::map<std::string, std::shared_ptr<zeno::IObject>> ObjectsManager::objs_filter(std::map<std::string, std::shared_ptr<zeno::IObject>> const &objs) {
    std::map<std::string, std::shared_ptr<zeno::IObject>> output;

    for (auto const &[key, obj] : objs) {
        auto ResourceType = obj->userData().get2("ResourceType", std::string(""));
        if (ResourceType == "SceneTree") {
            auto json_str = obj->userData().get2<std::string>("json", "");
            if (json_str.size()) {
                Json json = Json::parse(json_str);
                if (json["type"] == "static") {
                    staticSceneTree = json;
                }
                else if (json["type"] == "dynamic") {
                    dynamicSceneTree = json;
                }
            }
        }
        else if (ResourceType == "SceneDescriptor") {
            auto json_str = obj->userData().get2<std::string>("Scene", "");
            if (json_str.size()) {
                Json json = Json::parse(json_str);
                if (json["type"] == "static") {
                    staticSceneDescriptor = json;
                }
                else if (json["type"] == "dynamic") {
                    dynamicSceneDescriptor = json;
                }
            }
        }
        else {
            output[key] = obj;
        };
    }
    {
        Json scene_descriptor_json;
        if (staticSceneDescriptor.empty() == false) {
            scene_descriptor_json["StaticRenderGroups"] = staticSceneDescriptor["StaticRenderGroups"];
            scene_descriptor_json["BasicRenderInstances"].update(staticSceneDescriptor["BasicRenderInstances"]);
        }
        if (dynamicSceneDescriptor.empty() == false) {
            scene_descriptor_json["DynamicRenderGroups"] = dynamicSceneDescriptor["DynamicRenderGroups"];
            scene_descriptor_json["BasicRenderInstances"].update(dynamicSceneDescriptor["BasicRenderInstances"]);
        }
        if (scene_descriptor_json.empty() == false) {
            auto scene_descriptor = std::make_shared<zeno::PrimitiveObject>();
            auto &ud = scene_descriptor->userData();
            ud.set2("ResourceType", std::string("SceneDescriptor"));
            ud.set2("Scene", std::string(scene_descriptor_json.dump()));
            std::srand(std::time(0));
            auto json_key = zeno::format("GeneratedJson:{}", std::rand());
//            zeno::file_put_content("E:/fuck/Generated.json", ud.get2<std::string>("Scene"));
            output[json_key] = scene_descriptor;
        }
    }
    return output;
}

bool ObjectsManager::load_objects(std::map<std::string, std::shared_ptr<zeno::IObject>> const &objs_, std::string& runtype) {
    auto objs = objs_filter(objs_);
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
#if 0
            std::string stampChange = obj->userData().get2<std::string>("stamp-change", "TotalChange");
            if (stampChange != "TotalChange") {
                auto begin = objects.m_curr.begin();
                if (begin != objects.m_curr.end()) {
                    const std::string& oldkey = key.substr(0, key.find_first_of(":")) + begin->first.substr(begin->first.find_first_of(":"));
                    auto it = objects.m_curr.find(oldkey);
                    if (it != objects.m_curr.end()) {
                        newobj = it->second;
                        newobj->userData().set2("stamp-change", stampChange);
                        newobj->userData().set2("stamp-base", obj->userData().get2<int>("stamp-base", -1));
                        if (stampChange == "UnChanged") {
                        } else if (stampChange == "DataChange") {
                            const std::string& stampDatachangehint = newobj->userData().get2<std::string>("stamp-dataChange-hint", "");
                            //根据stampDatachangehint用obj的data信息更新newobj
                        } else if (stampChange == "ShapeChange") {
                            //暂时并入Totalchange
                            //用obj的shape信息更新newobj
                        }
                    } else {
                        newobj = obj;
                    }
                }
            }
#endif
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
#if 0
    if (runtype != "RunAll" && runtype != "LoadAsset") {
        std::set<std::string> keys;
        for (auto& [k, _] : objs) {
            keys.insert(k.substr(0, k.find_first_of(":")));
        }
        std::string objruntype = runtype == "RunLightCamera" ? "lightCamera" :
            (runtype == "RunMaterial" ? "material" :
                (runtype == "RunMatrix" ? "matrix" : "normal"));
        for (auto& [k, obj] : objects.m_curr) {
            if (obj && obj->userData().get2<std::string>("objRunType", "normal") != objruntype && keys.find(k.substr(0, k.find_first_of(":"))) == keys.end()) {
                ins.try_emplace(k, std::move(obj)); //key保持不变，不会触发optx加载obj
            }
        }
    }
#endif
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
