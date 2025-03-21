#include <zenovis/ObjectsManager.h>
#include <zeno/types/UserData.h>
#include <zeno/types/PrimitiveObject.h>
#include <zenovis/bate/IGraphic.h>
#include <zeno/core/IObject.h>
#include <zeno/utils/log.h>

namespace zenovis {

ObjectsManager::ObjectsManager() = default;
ObjectsManager::~ObjectsManager() = default;

bool ObjectsManager::load_objects(std::map<std::string, std::shared_ptr<zeno::IObject>> const &objs) {
    bool inserted = false;
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
            std::shared_ptr<zeno::IObject> newobj = obj;
            std::string stampChange = obj->userData().get2<std::string>("stamp-change", "TotalChange");
            if (stampChange != "TotalChange") {
                auto begin = objects.m_curr.begin();
                const std::string& oldkey = key.substr(0, key.find_first_of(":")) + begin->first.substr(begin->first.find_first_of(":"));
                auto it = objects.m_curr.find(oldkey);
                if (it != objects.m_curr.end()) {
                    newobj = objects.m_curr.find(oldkey)->second;
                    newobj->userData().set2("stamp-change", stampChange);
                    newobj->userData().set2("stamp-base", obj->userData().get2<int>("stamp-base", -1));
                    if (stampChange == "UnChanged") {
                    } else if (stampChange == "DataChange") {
                        //TODO
                        //用obj的data信息更新newobj
                    } else if (stampChange == "ShapeChange") {
                        //TODO
                        //用obj的shape信息更新newobj
                    }
                } else {
                    newobj = obj;
                }
            }

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
        if (key != nid) {
            continue;
        }
        return ptr;
    }

    return std::nullopt;
}

}
