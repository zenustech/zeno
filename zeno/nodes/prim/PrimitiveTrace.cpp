#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/utils/vec.h>
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
    {"PrimitiveObject", "parsPrim"},
    }, /* outputs: */ {
    {"PrimitiveObject", "trailPrim"},
    }, /* params: */ {
    }, /* category: */ {
    "primitive",
    }});


struct PrimitiveCalcVelocity : zeno::INode {
    std::shared_ptr<PrimitiveObject> trailPrim = std::make_shared<PrimitiveObject>();
    std::vector<vec3f> last_pos;

    virtual void apply() override {
        auto prim = get_input<PrimitiveObject>("prim");
        auto dt = has_input("dt") ? get_input<NumericObject>("dt")->get<float>() : 0.04f;
        last_pos = prim->attr<vec3f>("pos");
        auto const &pos = prim->attr<vec3f>("pos");
        auto &vel = prim->add_attr<vec3f>("vel");

#pragma omp parallel for
        for (int i = 0; i < std::min(last_pos.size(), pos.size()); i++) {
            vel[i] = (pos[i] - last_pos[i]) / dt;
        }

        set_output("prim", std::move(prim));
    }
};

ZENDEFNODE(PrimitiveCalcVelocity,
    { /* inputs: */ {
    {"PrimitiveObject", "prim"},
    {"NumericObject:float", "dt"},
    }, /* outputs: */ {
    {"PrimitiveObject", "prim"},
    }, /* params: */ {
    }, /* category: */ {
    "primitive",
    }});


}
