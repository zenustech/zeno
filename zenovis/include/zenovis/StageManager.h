#pragma once

#include <zenovis/Scene.h>
#include <zeno/core/IObject.h>
#include <zeno/utils/logger.h>
#include <zeno/utils/MapStablizer.h>
#include <zeno/utils/PolymorphicMap.h>

#include <string>
#include <memory>
#include <map>
#include <iostream>
#include <chrono>
#include <thread>
#include <functional>

struct ZenoStage;

namespace zenovis {

struct StageManager : zeno::disable_copy {
    /// ZenoObject (Editor) ---> UsdObject ---> ZenoObject (Convert)

    // ZenoObject (Editor)
    zeno::MapStablizer<zeno::PolymorphicMap<
        std::map<std::string, std::shared_ptr<zeno::IObject>>>> zenoObjects;

    // ZenoObject (Convert)
    zeno::MapStablizer<zeno::PolymorphicMap<
        std::map<std::string, std::shared_ptr<zeno::IObject>>>> convertObjects;

    // ZenoObject - Light
    std::map<std::string, std::shared_ptr<zeno::IObject>> zenoLightObjects;

    std::shared_ptr<ZenoStage> zenoStage;

    int increase_count = 0;

    StageManager();
    ~StageManager();

    template <class T = void>
    auto pairs() const {
        // XXX
        return zenoObjects.pairs<T>();
    }
    template <class T = void>
    auto pairsShared() const {
        return zenoObjects.pairsShared<T>();
    }

    bool load_objects(std::map<std::string, std::shared_ptr<zeno::IObject>> const &objs);
    std::optional<zeno::IObject*> get(std::string nid);
};
}