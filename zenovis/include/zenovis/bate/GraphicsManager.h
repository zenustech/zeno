#pragma once

#include <unordered_map>
#include <vector>
#include <zeno/utils/MapStablizer.h>
#include <zeno/utils/PolymorphicMap.h>
#include <zeno/utils/log.h>
#include <zenovis/bate/IGraphic.h>
#include <zenovis/Scene.h>

namespace zenovis {

struct GraphicsManager {
    Scene *scene;

    zeno::MapStablizer<zeno::PolymorphicMap<std::unordered_map<
        std::string, std::unique_ptr<IGraphic>>>> graphics;

    explicit GraphicsManager(Scene *scene) : scene(scene) {
    }

    void load_objects(std::vector<std::pair<std::string, zeno::IObject *>> const &objs) {
        auto ins = graphics.insertPass();
        for (auto const &[key, obj] : objs) {
            auto ig = makeGraphic(scene, obj);
            zeno::log_trace("load_object: load graphics {}", ig.get());
            ins.try_emplace(key, std::move(ig));
        }
    }
};

} // namespace zenovis
