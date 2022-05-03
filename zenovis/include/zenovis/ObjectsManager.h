#pragma once

#include <zenovis/Scene.h>
#include <zeno/utils/MapStablizer.h>
#include <zeno/utils/PolymorphicMap.h>
#include <zeno/utils/disable_copy.h>
#include <zeno/core/IObject.h>
#include <string>
#include <memory>
#include <map>

namespace zenovis {

struct ObjectsManager : zeno::disable_copy {
    zeno::MapStablizer<zeno::PolymorphicMap<std::map<
        std::string, std::shared_ptr<zeno::IObject>>>> objects;

    template <class T = void>
    auto values() const {
        return objects.values<T>();
    }

    template <class T = void>
    auto pairs() const {
        return objects.pairs<T>();
    }

    ObjectsManager();
    ~ObjectsManager();
    void clear_objects();
    void load_objects(std::map<std::string, std::shared_ptr<zeno::IObject>> const &objs);
};

}
