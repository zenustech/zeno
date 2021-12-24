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

};