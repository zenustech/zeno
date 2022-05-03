#include <zenovis/ObjectsManager.h>
#include <zenovis/bate/IGraphic.h>
#include <zeno/core/IObject.h>
#include <zeno/utils/log.h>

namespace zenovis {

ObjectsManager::ObjectsManager() = default;
ObjectsManager::~ObjectsManager() = default;

void ObjectsManager::load_objects(std::map<std::string, std::shared_ptr<zeno::IObject>> const &objs) {
    auto ins = objects.insertPass();
    for (auto const &[key, obj] : objs) {
        if (ins.may_emplace(key)) {
            zeno::log_debug("load_object: loading object [{}] at {}", key, obj.get());
            ins.try_emplace(key, std::move(obj));
        }
    }
}

void ObjectsManager::clear_objects() {
    objects.clear();
}

}
