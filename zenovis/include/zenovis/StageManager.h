#pragma once

#include <boost/predef/os.h>
#include <pxr/base/gf/camera.h>
#include <pxr/base/js/json.h>
#include <pxr/base/plug/plugin.h>
#include <pxr/base/tf/fileUtils.h>
#include <pxr/pxr.h>
#include <pxr/usd/usdGeom/xform.h>
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

PXR_NAMESPACE_USING_DIRECTIVE

namespace zenovis {

struct PrimInfo{
    std::string pPath;
    std::shared_ptr<zeno::IObject> iObject;
};

struct StageManager : zeno::disable_copy {
    zeno::MapStablizer<zeno::PolymorphicMap<
        std::map<std::string, std::shared_ptr<zeno::IObject>>>> zenoObjects;

    UsdStageRefPtr stagePtr;

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

    int _UsdGeomMesh(const PrimInfo& primInfo);

    void _CreateUSDHierarchy(const SdfPath &path)
    {
        if (path == SdfPath::AbsoluteRootPath())
            return;
        _CreateUSDHierarchy(path.GetParentPath());
        UsdGeomXform::Define(stagePtr, path);
    }
};
}