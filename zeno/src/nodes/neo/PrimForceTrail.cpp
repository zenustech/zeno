#include <zeno/para/parallel_for.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/PrimitiveUtils.h>
#include <zeno/types/StringObject.h>
#include <zeno/types/CurveObject.h>
#include <zeno/utils/arrayindex.h>
#include <zeno/utils/variantswitch.h>
#include <zeno/extra/TempNode.h>
#include <zeno/core/INode.h>
#include <zeno/zeno.h>
#include <limits>

namespace zeno {
namespace {

static float lineUDF(vec3f a, vec3f b, vec3f p) {
    auto aboab = lengthSquared(b - a);
    if (aboab <= std::numeric_limits<float>::epsilon())
        return length(p - a);
    auto apoab = dot(p - a, b - a);
    if (apoab < 0)
        return length(p - a);
    else if (apoab > aboab)
        return length(p - b);
    auto apxab = length(cross(p - a, b - a));
    return apxab / aboab;
}

static vec3f lineGrad(vec3f a, vec3f b, vec3f p) {
    const float dx = 1e-6f;
    return normalizeSafe(vec3f(
        lineUDF(a, b, p + vec3f(dx, 0, 0)) - lineUDF(a, b, p - vec3f(dx, 0, 0)),
        lineUDF(a, b, p + vec3f(0, dx, 0)) - lineUDF(a, b, p - vec3f(0, dx, 0)),
        lineUDF(a, b, p + vec3f(0, 0, dx)) - lineUDF(a, b, p - vec3f(0, 0, dx))));
    //auto aboab = lengthSquared(b - a);
    //if (aboab <= std::numeric_limits<float>::epsilon())
        //return normalizeSafe(p - a);
    //auto apxab = cross(p - a, b - a);
    //auto udf = length(apxab) / aboab;
    //return udf;
}

struct PrimForceTrail : INode {
    virtual void apply() override {
        auto prim = get_input<PrimitiveObject>("prim");
        auto trailPrim = get_input<PrimitiveObject>("trailPrim");
        auto forceAttr = get_input2<std::string>("forceAttr");
        auto attractForce = get_input2<float>("attractForce");
        auto driftForce = get_input2<float>("driftForce");

        auto attractUDFCurve = functor_variant(has_input("attractUDFCurve") ? 1 : 0, [&] {
            return [] (float x) -> float {
                return x;
            };
        }, [&] {
            auto curve = get_input<CurveObject>("attractUDFCurve");
            return [=] (float x) -> float {
                return curve->eval(x);
            };
        });
        auto driftCoordCurve = functor_variant(has_input("driftCoordCurve") ? 1 : 0, [&] {
            return [] (float x) -> float {
                return 1.f;
            };
        }, [&] {
            auto curve = get_input<CurveObject>("driftCoordCurve");
            return [=] (float x) -> float {
                return curve->eval(x);
            };
        });

        std::visit([&] (auto const &attractUDFCurve, auto const &driftCoordCurve) {
            auto &forceArr = prim->verts.add_attr<vec3f>(forceAttr);
            parallel_for(prim->verts.size(), [&] (size_t i) {
                auto pos = prim->verts[i];

                int finind = -1;
                float finudf = std::numeric_limits<float>::max();
                for (int k = 0; k < trailPrim->lines.size(); k++) {
                    auto line = trailPrim->lines[k];
                    float curudf = lineUDF(
                            trailPrim->verts[line[0]],
                            trailPrim->verts[line[1]],
                            pos);
                    if (curudf < finudf) {
                        finudf = curudf;
                        finind = k;
                    }
                }

                vec3f force{};
                if (finind != -1) {
                    auto finline = trailPrim->lines[finind];
                    auto lpa = trailPrim->verts[finline[0]];
                    auto lpb = trailPrim->verts[finline[1]];
                    auto fingrad = lineGrad(lpa, lpb, pos);
                    auto fintang = normalizeSafe(lpb - lpa);
                    auto fincoord = clamp(length(cross(pos - lpa, lpb - lpa)), 0.f, 1.f);
                    force = driftForce * driftCoordCurve(fincoord) * fintang;
                    force -= attractForce * attractUDFCurve(finudf) * fingrad;
                }
                forceArr[i] = force;
            });
        }, attractUDFCurve, driftCoordCurve);

        set_output("prim", std::move(prim));
    }
};

ZENDEFNODE(PrimForceTrail, {
    {
    {"PrimitiveObject", "prim"},
    {"PrimitiveObject", "trailPrim"},
    {"string", "forceAttr", "force"},
    {"float", "attractForce", "0.5"},
    {"float", "driftForce", "1"},
    {"", "attractUDFCurve", ""},
    {"", "driftCoordCurve", ""},
    },
    {
    {"PrimitiveObject", "prim"},
    },
    {
    },
    {"primitive"},
});

}
}
