#pragma once

#include <zeno/core/IObject.h>
#include <zeno/types/PrimitiveObject.h>
#include <Alembic/AbcGeom/Foundation.h>
#include <Alembic/AbcGeom/All.h>
#include <Alembic/AbcCoreAbstract/All.h>
#include <Alembic/AbcCoreOgawa/All.h>
#include <Alembic/AbcCoreHDF5/All.h>
#include <Alembic/Abc/ErrorHandler.h>
using Alembic::AbcGeom::ObjectVisibility;

namespace zeno {

struct CameraInfo {
    double _far;
    double _near;
    double focal_length;
    double horizontalAperture;
    double verticalAperture;
};

struct ABCTree : PrimitiveObject {
    std::string name;
    std::shared_ptr<PrimitiveObject> prim;
    Alembic::Abc::M44d xform = Alembic::Abc::M44d();
    std::shared_ptr<CameraInfo> camera_info;
    std::vector<std::shared_ptr<ABCTree>> children;
    ObjectVisibility visible = ObjectVisibility::kVisibilityDeferred;

    template <class Func>
    bool visitPrims(Func const &func) const {
        if constexpr (std::is_void_v<std::invoke_result_t<Func,
                      std::shared_ptr<PrimitiveObject> const &>>) {
            if (prim)
                func(prim);
            for (auto const &ch: children)
                ch->visitPrims(func);
        } else {
            if (prim)
                if (!func(prim))
                    return false;
            for (auto const &ch: children)
                if (!ch->visitPrims(func))
                    return false;
        }
        return true;
    }
};

}
