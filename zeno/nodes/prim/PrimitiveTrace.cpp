#include <zeno/zeno.h>
#include <zeno/PrimitiveObject.h>
#include <zeno/NumericObject.h>
#include <zeno/vec.h>
#include <cstring>
#include <cstdlib>
#include <cassert>

namespace zeno {


struct PrimitiveTraceTrail : zeno::INode {
    std::shared_ptr<PrimitiveObject> trailPrim = std::make_shared<PrimitiveObject>();

    virtual void apply() override {
        auto parsPrim = get_input<PrimitiveObject>("parsPrim");

        auto &parsPos = parsPrim->attr<zeno::vec3f>("pos");
        auto &trailPos = trailPrim->add_attr<zeno::vec3f>("pos");
        int base = trailPos.size();
        int last_base = base - parsPos.size();
        trailPos.resize(base + parsPos.size());
        for (int i = 0; i < parsPos.size(); i++) {
            trailPos[base + i] = parsPos[i];
        }
        if (last_base > 0) {
            for (int i = 0; i < parsPos.size(); i++) {
                trailPrim->lines.emplace_back(base + i, last_base + i);
            }
        }

        set_output("prim", trailPrim);
    }
};

ZENDEFNODE(PrimitiveTraceTrail,
    { /* inputs: */ {
    {"primitive", "parsPrim"},
    }, /* outputs: */ {
    {"primitive", "trailPrim"},
    }, /* params: */ {
    }, /* category: */ {
    "primitive",
    }});


}
