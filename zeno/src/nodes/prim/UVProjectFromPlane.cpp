#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/NumericObject.h>
#include <stdexcept>
#include "zeno/utils/log.h"

namespace zeno {
struct UVProjectFromPlane : zeno::INode {
    virtual void apply() override {
        auto prim = get_input<PrimitiveObject>("prim");
        auto &uv = prim->verts.add_attr<vec3f>("uv");
        auto refPlane = get_input<PrimitiveObject>("refPlane");
        if (refPlane->verts.size() != 4) {
            zeno::log_error("refPlane must be 1 * 1 plane!");
            throw zeno::makeError("refPlane must be 1 * 1 plane!");
        }
        auto originPos = refPlane->verts[2];
        auto xOffset = refPlane->verts[0];
        auto yOffset = refPlane->verts[3];
//        zeno::log_info("xOffset:{}, originPos: {}", xOffset, originPos);
        auto uDir = zeno::normalize(xOffset - originPos);
        auto vDir = zeno::normalize(yOffset - originPos);
        auto uLength = zeno::length(xOffset - originPos);
        auto vLength = zeno::length(yOffset - originPos);
        auto n = zeno::cross(uDir, vDir);
//        zeno::log_info("uDir:{], uLength: {}, n: {}", uDir, uLength, n);
        for (auto i = 0; i < prim->size(); i++) {
            auto &vert = prim->verts[i];
            auto offset = vert - originPos;
            auto proj = offset - zeno::dot(offset, n) * n;
            auto u = zeno::clamp(zeno::dot(proj, uDir) / uLength, 0, 1);
            auto v = zeno::clamp(zeno::dot(proj, vDir) / vLength, 0, 1);
            uv[i] = zeno::vec3f(u,  v, 0);
        }

        set_output("outPrim", std::move(prim));
    }
};

ZENDEFNODE(UVProjectFromPlane, {
    {
        {"PrimitiveObject", "prim"},
        {"PrimitiveObject", "refPlane"},
    },
    {
        {"PrimitiveObject", "outPrim"}
    },
    {},
    {"primitive"},
});
}
