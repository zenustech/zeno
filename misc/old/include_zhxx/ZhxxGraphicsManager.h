#pragma once

#include <map>
#include <vector>
#include <zeno/utils/MapStablizer.h>
#include <zeno/utils/PolymorphicMap.h>
#include <zeno/utils/log.h>
#include <zenovis/zhxx/ZhxxIGraphic.h>
#include <zenovis/zhxx/ZhxxScene.h>

namespace zenovis::zhxx {

struct ZhxxGraphicsManager {
    ZhxxScene *scene;

    zeno::MapStablizer<zeno::PolymorphicMap<std::map<
        std::string, std::unique_ptr<ZhxxIGraphic>>>> graphics;

    explicit ZhxxGraphicsManager(ZhxxScene *scene) : scene(scene) {
    }

    void load_objects(std::vector<std::pair<std::string, zeno::IObject *>> const &objs) {
        auto ins = graphics.insertPass();
        for (auto const &[key, obj] : objs) {
            if (ins.may_emplace(key)) {
                auto ig = makeGraphic(scene, obj);
                zeno::log_debug("(zxx) load_object: load graphics {}", ig.get());
                ins.try_emplace(key, std::move(ig));
            }
        }
    }
};

} // namespace zenovis
