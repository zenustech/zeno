#include <zenovis/ObjectsManager.h>
#include <zenovis/bate/IGraphic.h>
#include <zeno/core/IObject.h>
#include <zeno/utils/log.h>
#include <zeno/types/UserData.h>
#include <zeno/utils/vec.h>
#include <zeno/funcs/PrimitiveUtils.h>

namespace zenovis {

ObjectsManager::ObjectsManager() = default;
ObjectsManager::~ObjectsManager() = default;

void ObjectsManager::load_objects(std::map<std::string, std::shared_ptr<zeno::IObject>> const &objs) {
    auto ins = objects.insertPass();
    for (auto const &[key, obj] : objs) {
        auto &ud = obj->userData();
        if (ins.may_emplace(key)) {
            if (auto prim = dynamic_cast<zeno::PrimitiveObject *>(obj.get())) {
                auto [bmin, bmax] = zeno::primBoundingBox(prim);
                auto delta = bmax - bmin;
                float radius = std::max({delta[0], delta[1], delta[2]}) * 0.5f;
                zeno::vec3f center = (bmin + bmax) * 0.5f;
                ud.set("_bboxMin", std::make_shared<zeno::NumericObject>(bmin));
                ud.set("_bboxMax", std::make_shared<zeno::NumericObject>(bmax));
                ud.set("_bboxRadius", std::make_shared<zeno::NumericObject>(radius));
                ud.set("_bboxCenter", std::make_shared<zeno::NumericObject>(center));
            }
            zeno::log_debug("load_objects: loading object [{}] at {}", key, obj.get());
            ins.try_emplace(key, std::move(obj));
        }
    }
}

void ObjectsManager::clear_objects() {
    objects.clear();
}

}
