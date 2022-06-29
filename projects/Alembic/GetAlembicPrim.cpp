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
void dfs_abctree(
    std::shared_ptr<ABCTree> root,
    int parent_index,
    std::vector<std::shared_ptr<ABCTree>>& linear_abctrees,
    std::vector<int>& linear_abctree_parent
) {
    int self_index = linear_abctrees.size();
    linear_abctrees.push_back(root);
    linear_abctree_parent.push_back(parent_index);
    for (auto const &ch: root->children) {
        dfs_abctree(ch, self_index, linear_abctrees, linear_abctree_parent);
    }
}

std::shared_ptr<PrimitiveObject> get_xformed_prim(std::shared_ptr<zeno::ABCTree> abctree, int index) {
    std::vector<std::shared_ptr<ABCTree>> linear_abctrees;
    std::vector<int> linear_abctree_parent;
    dfs_abctree(abctree, -1, linear_abctrees, linear_abctree_parent);
    std::shared_ptr<PrimitiveObject> prim;
    std::vector<Alembic::Abc::M44d> transforms;
    for (auto i = 0; i < linear_abctrees.size(); i++) {
        auto const& abc_node = linear_abctrees[i];
        int parent_index = linear_abctree_parent[i];
        if (parent_index >= 0) {
            transforms.push_back(abc_node->xform * transforms[parent_index]);
        } else {
            transforms.push_back(abc_node->xform);
        }
        if (abc_node->prim) {
            if (index == 0) {
                prim = std::static_pointer_cast<PrimitiveObject>(abc_node->prim->clone());
                auto& mat = transforms.back();
                for (auto& p: prim->verts) {
                    auto pos = Imath::V4d(p[0], p[1], p[2], 1) * mat;
                    p = zeno::vec3f((float)pos.x, (float)pos.y, (float)pos.z);
                }
            }
            index--;
        }
    }
    return prim;
}

std::shared_ptr<zeno::ListObject>
get_xformed_prims(
    std::shared_ptr<zeno::ABCTree> abctree
) {
    auto prims = std::make_shared<zeno::ListObject>();
    std::vector<std::shared_ptr<ABCTree>> linear_abctrees;
    std::vector<int> linear_abctree_parent;
    dfs_abctree(abctree, -1, linear_abctrees, linear_abctree_parent);
    std::shared_ptr<PrimitiveObject> prim;
    std::vector<Alembic::Abc::M44d> transforms;
    for (auto i = 0; i < linear_abctrees.size(); i++) {
        auto const& abc_node = linear_abctrees[i];
        int parent_index = linear_abctree_parent[i];
        if (parent_index >= 0) {
            transforms.push_back(abc_node->xform * transforms[parent_index]);
        } else {
            transforms.push_back(abc_node->xform);
        }
        if (abc_node->prim) {
            prim = std::static_pointer_cast<PrimitiveObject>(abc_node->prim->clone());
            auto& mat = transforms.back();
            for (auto& p: prim->verts) {
                auto pos = Imath::V4d(p[0], p[1], p[2], 1) * mat;
                p = zeno::vec3f((float)pos.x, (float)pos.y, (float)pos.z);
            }
            prims->arr.push_back(prim);
        }
    }
    return prims;
}
struct GetAlembicPrim : INode {
    virtual void apply() override {
        auto abctree = get_input<ABCTree>("abctree");
        int index = get_input<NumericObject>("index")->get<int>();
        int use_xform = get_input<NumericObject>("use_xform")->get<int>();
        std::shared_ptr<PrimitiveObject> prim;
        if (use_xform) {
            prim = get_xformed_prim(abctree, index);
        } else {
            prim = get_alembic_prim(abctree, index);
        }
        set_output("prim", std::move(prim));
    }
};

ZENDEFNODE(GetAlembicPrim, {
    {
        {"ABCTree", "abctree"},
        {"int", "index", "0"},
        {"int", "use_xform", "0"}
    },
    {{"PrimitiveObject", "prim"}},
    {},
    {"alembic"},
});

struct AllAlembicPrim : INode {
    virtual void apply() override {
        auto abctree = get_input<ABCTree>("abctree");
        auto prims = std::make_shared<zeno::ListObject>();
        int use_xform = get_input<NumericObject>("use_xform")->get<int>();
        if (use_xform) {
            prims = get_xformed_prims(abctree);
        } else {
            abctree->visitPrims([&] (auto const &p) {
                auto np = std::static_pointer_cast<PrimitiveObject>(p->clone());
                prims->arr.push_back(np);
            });
        }
        auto outprim = prim_merge(prims);
        set_output("prim", std::move(outprim));
    }
};

ZENDEFNODE(AllAlembicPrim, {
    {
        {"ABCTree", "abctree"},
        {"int", "use_xform", "0"}
    },
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
                q.emplace(t->xform * m, ch);
            }
        }
        if (!cam_info.has_value()) {
            log_error("Not found camera!");
        }

        auto pos = Imath::V4d(0, 0, 0, 1) * mat;
        auto up = Imath::V4d(0, 1, 0, 0) * mat;
        auto right = Imath::V4d(1, 0, 0, 0) * mat;

        float h_fov = (float)std::atan(24.0 / (2.0 * cam_info.value().focal_length));

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
