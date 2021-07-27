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

        int base = trailPrim->size();
        int last_base = base - parsPrim->size();
        trailPrim->resize(base + parsPrim->size());

        for (auto const &[parsAttr, parsArr]: parsPrim->m_attrs) {
            std::visit([&, trailAttr = parsAttr] (auto const &parsArr) {
                using T = std::decay_t<decltype(parsArr[0])>;
                auto &trailArr = trailPrim->add_attr<T>(trailAttr);
                for (int i = 0; i < parsPrim->size(); i++) {
                    trailArr[base + i] = parsArr[i];
                }
            }, parsArr);
        }
        if (last_base > 0) {
            for (int i = 0; i < parsPrim->size(); i++) {
                trailPrim->lines.emplace_back(base + i, last_base + i);
            }
        }

        set_output("trailPrim", trailPrim);
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
