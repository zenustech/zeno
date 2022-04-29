#pragma once

#include <unordered_map>
#include <vector>
#include <zeno/utils/MapStablizer.h>
#include <zeno/utils/PolymorphicMap.h>
#include <zeno/utils/log.h>
#include <zenovis/zhxx/ZhxxIGraphic.h>
#include <zenovis/zhxx/ZhxxScene.h>

namespace zenovis::zhxx {

struct ZhxxGraphicsManager {
    ZhxxScene *scene;

    zeno::MapStablizer<zeno::PolymorphicMap<std::unordered_map<
        std::shared_ptr<zeno::IObject>, std::unique_ptr<ZhxxIGraphic>>>> graphics;

    explicit ZhxxGraphicsManager(ZhxxScene *scene_) : scene(scene_) {
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
