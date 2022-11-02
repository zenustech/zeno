#pragma once

#include <boost/predef/os.h>
#include <pxr/base/gf/camera.h>
#include <pxr/base/js/json.h>
#include <pxr/base/plug/plugin.h>
#include <pxr/base/tf/fileUtils.h>
#include <pxr/pxr.h>
#include <pxr/usd/usd/inherits.h>
#include <pxr/usd/usd/stage.h>
#include <pxr/usd/usdGeom/xformable.h>
#include <pxr/usd/usdGeom/camera.h>

#include <zenovis/Scene.h>
#include <zeno/core/IObject.h>
#include <zeno/utils/logger.h>
#include <zeno/utils/MapStablizer.h>
#include <zeno/utils/PolymorphicMap.h>

#include <string>
#include <memory>
#include <map>

namespace zenovis {

struct StageManager : zeno::disable_copy {
    zeno::MapStablizer<zeno::PolymorphicMap<
        std::map<std::string, std::shared_ptr<zeno::IObject>>>> zenoObjects;

    StageManager();
    ~StageManager();

    template <class T = void>
    auto pairs() const {
        return zenoObjects.pairs<T>();
    }

    template <class T = void>
    auto pairsShared() const {
        return zenoObjects.pairsShared<T>();
    }

    bool load_objects(std::map<std::string, std::shared_ptr<zeno::IObject>> const &objs);
};
}