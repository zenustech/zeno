#include <zenovis/ObjectsManager.h>
#include <zeno/types/UserData.h>
#include <zeno/types/PrimitiveObject.h>
#include <zenovis/bate/IGraphic.h>
#include <zeno/core/IObject.h>
#include <zeno/utils/log.h>

namespace zenovis {

ObjectsManager::ObjectsManager() = default;
ObjectsManager::~ObjectsManager() = default;

void ObjectsManager::load_objects(std::map<std::string, std::shared_ptr<zeno::IObject>> const &objs) {
    auto ins = objects.insertPass();

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
                auto isLight = prim_in->userData().getLiterial<int>("isL", 0);
                if(isLight){
                    printf("loading light object\n");
                    lightObjects[key] = obj;
                }
            }
            ins.try_emplace(key, std::move(obj));
        }
    }
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
