#include <zeno/zeno.h>
#include <zeno/utils/logger.h>
#include <zeno/utils/prim_ops.h>
#include <zeno/types/ListObject.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/StringObject.h>
#include <zeno/types/NumericObject.h>
#include "ABCTree.h"
#include <queue>
#include <utility>

namespace zeno {
namespace {

int count_alembic_prims(std::shared_ptr<zeno::ABCTree> abctree) {
    int count = 0;
    abctree->visitPrims([&] (auto const &p) {
        count++;
    });
    return count;
}

struct CountAlembicPrims : INode {
    virtual void apply() override {
        auto abctree = get_input<ABCTree>("abctree");
        std::shared_ptr<PrimitiveObject> prim;
        int count = count_alembic_prims(abctree);
        set_output("count", std::make_shared<NumericObject>(count));
    }
};

ZENDEFNODE(CountAlembicPrims, {
    {{"ABCTree", "abctree"}},
    {{"int", "count"}},
    {},
    {"alembic"},
});

std::shared_ptr<PrimitiveObject> get_alembic_prim(std::shared_ptr<zeno::ABCTree> abctree, int index) {
    std::shared_ptr<PrimitiveObject> prim;
    abctree->visitPrims([&] (auto const &p) {
        if (index == 0) {
            prim = p;
            return false;
        }
        index--;
        return true;
    });
    if (!prim) {
        throw Exception("index out of range in abctree");
    }
    return prim;
}
struct GetAlembicPrim : INode {
    virtual void apply() override {
        auto abctree = get_input<ABCTree>("abctree");
        int index = get_input<NumericObject>("index")->get<int>();
        std::shared_ptr<PrimitiveObject> prim = get_alembic_prim(abctree, index);
        set_output("prim", std::move(prim));
    }
};

ZENDEFNODE(GetAlembicPrim, {
    {{"ABCTree", "abctree"}, {"int", "index", "0"}},
    {{"PrimitiveObject", "prim"}},
    {},
    {"alembic"},
});

struct AllAlembicPrim : INode {
    virtual void apply() override {
        auto abctree = get_input<ABCTree>("abctree");
        auto prims = std::make_shared<zeno::ListObject>();
        abctree->visitPrims([&] (auto const &p) {
            auto np = std::static_pointer_cast<PrimitiveObject>(p->clone());
            prims->arr.push_back(np);
        });
        auto outprim = primitive_merge(prims);
        set_output("prim", std::move(outprim));
    }
};

ZENDEFNODE(AllAlembicPrim, {
    {{"ABCTree", "abctree"}},
    {{"PrimitiveObject", "prim"}},
    {},
    {"alembic"},
});

struct GetAlembicCamera : INode {
    virtual void apply() override {
        auto abctree = get_input<ABCTree>("abctree");
        std::queue<std::pair<Alembic::Abc::v12::M44d, std::shared_ptr<ABCTree>>> q;
        q.emplace(Alembic::Abc::v12::M44d(), abctree);
        Alembic::Abc::v12::M44d mat;
        std::optional<CameraInfo> cam_info;
        while (q.size() > 0) {
            auto [m, t] = q.front();
            q.pop();
            if (t->camera_info) {
                mat = m;
                cam_info = *(t->camera_info);
                break;
            }
            for (auto ch: t->children) {
                q.emplace(m * t->xform, ch);
            }
        }
        if (!cam_info.has_value()) {
            log_error("Not found camera!");
        }

        auto pos = Imath::V4d(0, 0, 0, 1) * mat;
        auto up = Imath::V4d(0, 1, 0, 0) * mat;
        auto right = Imath::V4d(1, 0, 0, 0) * mat;

        float h_fov = (float)std::atan(35.0 / (2.0 * cam_info.value().focal_length));

        set_output("pos", std::make_shared<NumericObject>(zeno::vec3f((float)pos.x, (float)pos.y, (float)pos.z)));
        set_output("up", std::make_shared<NumericObject>(zeno::vec3f((float)up.x, (float)up.y, (float)up.z)));
        set_output("right", std::make_shared<NumericObject>(zeno::vec3f((float)right.x, (float)right.y, (float)right.z)));

        set_output("half_fov", std::make_shared<NumericObject>(h_fov));
        set_output("near", std::make_shared<NumericObject>((float)cam_info.value()._near));
        set_output("far", std::make_shared<NumericObject>((float)cam_info.value()._far));
    }
};

ZENDEFNODE(GetAlembicCamera, {
    {{"ABCTree", "abctree"}},
    {
        "pos",
        "up",
        "right",
        "half_fov",
        "near",
        "far",
    },
    {},
    {"alembic"},
});

} // namespace
} // namespace zeno
