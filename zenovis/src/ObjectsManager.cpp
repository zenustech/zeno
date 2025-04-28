#include <zenovis/ObjectsManager.h>
#include <zeno/types/UserData.h>
#include <zeno/types/PrimitiveObject.h>
#include <zenovis/bate/IGraphic.h>
#include <zeno/core/IObject.h>
#include <zeno/utils/log.h>
#include <set>

namespace zenovis {

ObjectsManager::ObjectsManager() = default;
ObjectsManager::~ObjectsManager() = default;

bool ObjectsManager::load_objects(std::map<std::string, std::shared_ptr<zeno::IObject>> const &objs, std::string& runtype) {
    bool inserted = false;
    auto ins = objects.insertPass();

    bool changed = false;
    for (auto const &[key, obj] : objs) {
        if (ins.may_emplace(key)) {
            changed = true;
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
        if (key != nid) {
            continue;
        }
        return ptr;
    }

    return std::nullopt;
}

}
