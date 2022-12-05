#include <zenovis/ObjectsManager.h>
#include <zenovis/LiveManager.h>
#include <zeno/types/UserData.h>
#include <zeno/types/PrimitiveObject.h>
#include <zenovis/bate/IGraphic.h>
#include <zeno/core/IObject.h>
#include <zeno/utils/log.h>
#include <iostream>
namespace zenovis {

ObjectsManager::ObjectsManager() = default;
ObjectsManager::~ObjectsManager() = default;

bool ObjectsManager::load_objects(std::map<std::string, std::shared_ptr<zeno::IObject>> const &objs) {
    bool inserted = false;
    auto ins = objects.insertPass();

    for (auto const &[key, obj] : objs) {
        if (auto prim_in = dynamic_cast<zeno::PrimitiveObject *>(obj.get())) {
            auto isLiveObject = prim_in->userData().get2<int>("IsLiveObject", 0);
            if (isLiveObject) {
                if(scene->liveMan->primObject.get() != nullptr){
                    auto newKey = key + ":Live:"+std::to_string(scene->liveMan->verLoadCount);
                    if (ins.may_emplace(newKey)) {
                        ins.try_emplace(newKey, (scene->liveMan->primObject));
                        inserted = true;
                    }
                }
            }
        }
    }

    bool changed_light = false;
    for (auto const &[key, obj] : objs) {
        if (ins.may_emplace(key)) {
            changed_light = true;
        }
    }
    if(changed_light){
        lightObjects.clear();
    }
    for (auto const &[key, obj] : objs) {
        if (ins.may_emplace(key)) {
            if (auto prim_in = dynamic_cast<zeno::PrimitiveObject *>(obj.get())) {
                auto isRealTimeObject = prim_in->userData().get2<int>("isRealTimeObject", 0);
                if(isRealTimeObject){
                    //printf("loading light object %s\n", key.c_str());
                    lightObjects[key] = obj;
                }
            }
            ins.try_emplace(key, std::move(obj));
            inserted = true;
        }
    }
    return inserted;
}

void ObjectsManager::clear_objects() {
    objects.clear();
}

std::optional<zeno::IObject* > ObjectsManager::get(std::string nid) {
    for (auto &[key, ptr]: this->pairs()) {
        if (key != nid) {
            continue;
        }
        return ptr;
    }

    return std::nullopt;
}

}
