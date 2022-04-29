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
        std::shared_ptr<zeno::IObject>, std::unique_ptr<IGraphic>>>> graphics;

    explicit GraphicsManager(Scene *scene) : scene(scene) {
    }

    void load_objects(std::vector<std::shared_ptr<zeno::IObject>> const &objs) {
        auto ins = graphics.insertPass();
        for (auto const &obj : objs) {
            auto ig = makeGraphic(scene, obj.get());
            zeno::log_trace("load_object: load graphics {}", ig.get());
            ins.try_emplace(obj, std::move(ig));
        }
    }
};

} // namespace zenovis
