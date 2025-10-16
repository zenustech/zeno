#pragma once

#include <tinygltf/json.hpp>
using Json = nlohmann::json;

#include <zeno/core/IObject.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/utils/safe_dynamic_cast.h>
#include <memory>
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

struct ABCTree : IObject {
    std::string name;
    std::unique_ptr<zeno::PrimitiveObject> prim;
    Alembic::Abc::M44d xform = Alembic::Abc::M44d();
    std::unique_ptr<CameraInfo> camera_info;
    std::vector<std::unique_ptr<ABCTree>> children;
    ObjectVisibility visible = ObjectVisibility::kVisibilityDeferred;
    std::string instanceSourcePath;

    zany clone() const override {
        auto tree = std::make_unique<ABCTree>();
        tree->name = name;
        if (prim) {
            tree->prim = zeno::safe_uniqueptr_cast<PrimitiveObject>(prim->clone());
        }
        tree->xform = xform;
        if (camera_info) {
            tree->camera_info = std::make_unique<CameraInfo>(*camera_info);
        }
        tree->children.resize(children.size());
        for (int i = 0; i < children.size(); i++) {
            tree->children[i] = zeno::safe_uniqueptr_cast<ABCTree>(children[i]->clone());
        }
        tree->visible = visible;
        tree->instanceSourcePath = instanceSourcePath;
        return tree;
    }

    Json get_scene_info(
        ObjectVisibility parent_visible = ObjectVisibility::kVisibilityVisible
    ) {
        Json json;
        ObjectVisibility cur_visible = visible == ObjectVisibility::kVisibilityDeferred? parent_visible: visible;
        json["visibility"] = int(cur_visible);
        json["node_name"] = name;
        if (instanceSourcePath.size()) {
            json["instance_source_path"] = "/ABC" + instanceSourcePath;
        }
        if (prim) {
            json["mesh"] = "mesh";
        }
        auto r0 = Imath::V4d(1, 0, 0, 0) * xform;
        auto r1 = Imath::V4d(0, 1, 0, 0) * xform;
        auto r2 = Imath::V4d(0, 0, 1, 0) * xform;
        auto t  = Imath::V4d(0, 0, 0, 1) * xform;
        json["r0"] = {r0[0], r0[1], r0[2]};
        json["r1"] = {r1[0], r1[1], r1[2]};
        json["r2"] = {r2[0], r2[1], r2[2]};
        json["t"]  = {t[0], t[1], t[2]};
        json["children_name"] = Json::array();
        if (instanceSourcePath.empty()) {
            for (const auto &child: children) {
                auto cjson = child->get_scene_info(cur_visible);
                auto name = cjson["node_name"];
                json["children_name"].push_back(name);
                json[name] = cjson;
            }
        }
        return json;
    }

    template <class Func>
    bool visitPrims(Func const &func) const {
        if constexpr (std::is_void_v<std::invoke_result_t<Func,
                      std::unique_ptr<PrimitiveObject> const &>>) {
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
