#include <zeno/zeno.h>
#include <zeno/logger.h>
#include <zeno/ListObject.h>
#include <zeno/NumericObject.h>
#include <zeno/PrimitiveObject.h>
#include <zeno/utils/UserData.h>
#include <zeno/StringObject.h>

#include "skinning_iobject.h"

namespace{
using namespace zeno;

struct MakeEmptyBones : zeno::INode {
    virtual void apply() override {
        auto root = get_input<zeno::NumericObject>("root")->get<zeno::vec3f>();
        auto res = std::make_shared<zeno::PrimitiveObject>();

        res->resize(1);
        res->verts[0] = root;
        
        set_output("bones",std::move(res));
    }
};

ZENDEFNODE(MakeEmptyBones, {
    {"root"},
    {"bones"},
    {},
    {"Skinning"},
});

struct AddBone : zeno::INode {
    virtual void apply() override {
        auto bones = get_input<zeno::PrimitiveObject>("bones");
        auto node = get_input<zeno::NumericObject>("node")->get<zeno::vec3f>();
        auto connect_to = get_input<zeno::NumericObject>("conn_to")->get<int>();

        int idx = bones->size();

        bones->verts.emplace_back(node);
        bones->lines.emplace_back(connect_to,idx);

        set_output("bones_out",bones);
    }
};

ZENDEFNODE(AddBone, {
    {"bones","node","conn_to"},
    {"bones_out"},
    {},
    {"Skinning"},
});

struct EvalBonesAngle : zeno::INode {
    virtual void apply() override {
        auto bones = get_input<zeno::PrimitiveObject>("bones");
        auto bs_idx = get_input<zeno::NumericObject>("bs_idx")->get<zeno::vec2i>();

        const auto& B1 = bones->lines[bs_idx[0]];
        const auto& B2 = bones->lines[bs_idx[1]];

        auto B1Dir = bones->verts[B1[1]] - bones->verts[B1[0]];
        auto B2Dir = bones->verts[B2[1]] - bones->verts[B2[0]];

        float cosAngle = zeno::dot(B1Dir,B2Dir)/zeno::length(B1Dir)/zeno::length(B2Dir);
        auto res = std::make_shared<zeno::NumericObject>();
        res->set<float>(zeno::acos(cosAngle));

        set_output("res",std::move(res));
    }
};

ZENDEFNODE(EvalBonesAngle, {
    {"bones","bs_idx"},
    {"res"},
    {},
    {"Skinning"},
});

};