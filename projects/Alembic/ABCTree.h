#pragma once

#include <tinygltf/json.hpp>
using Json = nlohmann::ordered_json;

#include <zeno/core/IObject.h>
#include <zeno/types/PrimitiveObject.h>
#include <Alembic/AbcGeom/Foundation.h>
#include <Alembic/AbcGeom/All.h>
#include <Alembic/AbcCoreAbstract/All.h>
#include <Alembic/AbcCoreOgawa/All.h>
#include <Alembic/AbcCoreHDF5/All.h>
#include <Alembic/Abc/ErrorHandler.h>
#include "zeno/utils/log.h"

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
    std::string instanceSourcePath;

    Json get_scene_info(
        ObjectVisibility parent_visible = ObjectVisibility::kVisibilityVisible
        , Alembic::Abc::M44d parent_xform = Alembic::Abc::M44d()
    ) {
        Json json;
        ObjectVisibility cur_visible = visible == ObjectVisibility::kVisibilityDeferred? parent_visible: visible;
        json["visibility"] = int(cur_visible);
        json["node_name"] = name;
        json["instance_source_path"] = instanceSourcePath;
//        if (instanceSourcePath.size()) {
//            zeno::log_info("{}: {}", name, instanceSourcePath);
//        }
        auto r0 = Imath::V4d(1, 0, 0, 0) * xform;
        auto r1 = Imath::V4d(0, 1, 0, 0) * xform;
        auto r2 = Imath::V4d(0, 0, 1, 0) * xform;
        auto t  = Imath::V4d(0, 0, 0, 1) * xform;
        json["r0"] = {r0[0], r0[1], r0[2]};
        json["r1"] = {r1[0], r1[1], r1[2]};
        json["r2"] = {r2[0], r2[1], r2[2]};
        json["t"]  = {t[0], t[1], t[2]};
        auto mat = xform * parent_xform;
        {
            auto r0 = Imath::V4d(1, 0, 0, 0) * mat;
            auto r1 = Imath::V4d(0, 1, 0, 0) * mat;
            auto r2 = Imath::V4d(0, 0, 1, 0) * mat;
            auto t  = Imath::V4d(0, 0, 0, 1) * mat;
            json["gr0"] = {r0[0], r0[1], r0[2]};
            json["gr1"] = {r1[0], r1[1], r1[2]};
            json["gr2"] = {r2[0], r2[1], r2[2]};
            json["gt"]  = {t[0], t[1], t[2]};
        }
        json["children_name"] = Json::array();
        if (instanceSourcePath.empty()) {
            for (const auto &child: children) {
                auto cjson = child->get_scene_info(cur_visible, mat);
                auto name = cjson["node_name"];
                json["children_name"].push_back(name);
                json[name] = cjson;
            }
        }
        return json;
    }

    template <class Func>
    bool visitPrims(Func const &func, bool skip_instance = false) const {
        if (skip_instance && instanceSourcePath.size() > 0) {
            return true;
        }
        if constexpr (std::is_void_v<std::invoke_result_t<Func,
                      std::shared_ptr<PrimitiveObject> const &>>) {
            if (prim)
                func(prim);
            for (auto const &ch: children)
                ch->visitPrims(func, skip_instance);
        } else {
            if (prim)
                if (!func(prim))
                    return false;
            for (auto const &ch: children)
                if (!ch->visitPrims(func, skip_instance))
                    return false;
        }
        return true;
    }
};

}
