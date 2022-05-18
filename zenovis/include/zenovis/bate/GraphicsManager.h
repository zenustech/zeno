#pragma once

#include <map>
#include <vector>
#include <zeno/types/UserData.h>
#include <zeno/utils/MapStablizer.h>
#include <zeno/utils/PolymorphicMap.h>
#include <zeno/utils/log.h>
#include <zenovis/bate/IGraphic.h>
#include <zenovis/Scene.h>

namespace zenovis {

struct GraphicsManager {
    Scene *scene;

    zeno::MapStablizer<zeno::PolymorphicMap<std::map<
        std::string, std::unique_ptr<IGraphic>>>> graphics;

    explicit GraphicsManager(Scene *scene) : scene(scene) {
    }

    bool load_objects(std::vector<std::pair<std::string, zeno::IObject *>> const &objs) {
        auto ins = graphics.insertPass();
        for (auto const &[key, obj] : objs) {
            if (ins.may_emplace(key)) {
                obj->userData().set("nameid", std::make_shared<zeno::StringObject>(key));
                zeno::log_debug("load_object: loading graphics [{}]", key);
                auto ig = makeGraphic(scene, obj);
                zeno::log_debug("load_object: loaded graphics to {}", ig.get());
                ins.try_emplace(key, std::move(ig));
            }
        }
        return ins.has_changed();
    }
};

} // namespace zenovis
