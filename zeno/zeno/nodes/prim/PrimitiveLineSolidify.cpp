#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/StringObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/utils/orthonormal.h>
#include <zeno/utils/variantswitch.h>
#include <zeno/utils/parallel.h>
#include <sstream>
#include <iostream>
#include <cmath>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace zeno {

struct PrimitiveLineSimpleLink : zeno::INode {
    virtual void apply() override {
        auto prim = get_input<zeno::PrimitiveObject>("prim");

        prim->lines.clear();
        intptr_t n = prim->verts.size();
        for (intptr_t i = 1; i < n; i++) {
            prim->lines.emplace_back(i - 1, i);
        }
        prim->lines.update();
        set_output("prim", std::move(prim));
    }
};


ZENDEFNODE(PrimitiveLineSimpleLink, {
    {
    {"PrimitiveObject", "prim"},
    },
    {
    {"PrimitiveObject", "prim"},
    },
    {
    },
    {"primitive"},
});


struct PrimitiveLineSolidify : zeno::INode {
    virtual void apply() override {
        auto prim = get_input<zeno::PrimitiveObject>("prim");
        auto count = get_input<zeno::NumericObject>("count")->get<int>();
        auto radius = get_input<zeno::NumericObject>("radius")->get<float>();
        auto radiusAttr = get_input<zeno::StringObject>("radiusAttr")->get();
        bool isTri = get_input2<bool>("isTri");

        intptr_t n = prim->verts.size();
        if (n >= 2) {

            prim->lines.clear();

            prim->verts.resize(count * n);

            std::vector<vec3f> directions(n);
#pragma omp parallel for
            for (intptr_t i = 1; i < n - 1; i++) {
                auto lastpos = prim->verts[i - 1];
                auto currpos = prim->verts[i];
                auto nextpos = prim->verts[i + 1];
                auto direction = normalize(nextpos - lastpos);
                directions[i] = direction;
            }
            directions[0] = normalize(prim->verts[1] - prim->verts[0]);
            directions[n - 1] = normalize(prim->verts[n - 1] - prim->verts[n - 2]);

            std::vector<float> sinang(count);
            std::vector<float> cosang(count);

            for (int a = 0; a < count; a++) {
                float ang = a * (float{M_PI} * 2 / count);
                sinang[a] = std::sin(ang) * radius;
                cosang[a] = std::cos(ang) * radius;
            }

            boolean_switch(!radiusAttr.empty(), [&] (auto has_radius_attr) {

                decltype(auto) radattr = [&] () -> decltype(auto) {
                    if constexpr (has_radius_attr.value)
                        return prim->attr<float>(radiusAttr);
                    else
                        return std::false_type{};
                }();

#pragma omp parallel for
            for (intptr_t i = 0; i < n; i++) {
                auto currpos = prim->verts[i];
                orthonormal orb(directions[i]);
                for (int a = 0; a < count; a++) {
                    auto offs = orb.tangent * sinang[a] + orb.bitangent * cosang[a];
                    if constexpr (has_radius_attr.value)
                        offs *= radattr[i];
                    prim->verts[i + n * a] = currpos + offs;
                }
            }

            });

            boolean_switch(isTri, [&] (auto isTri) {

                if constexpr (isTri.value)
                    prim->tris.resize((n - 1) * count * 2);
                else
                    prim->quads.resize((n - 1) * count);

#pragma omp parallel for
                for (intptr_t i = 0; i < n - 1; i++) {
                    for (int a = 0; a < count - 1; a++) {
                        int p1 = i + n * a;
                        int p2 = i + n * (a+1);
                        int p3 = i+1 + n * (a+1);
                        int p4 = i+1 + n * a;
                        if constexpr (isTri.value) {
                            prim->tris[(i * count + a)*2] = {p1, p2, p3};
                            prim->tris[(i * count + a)*2+1] = {p1, p3, p4};
                        } else {
                            prim->quads[i * count + a] = {p1, p2, p3, p4};
                        }
                    }
                    int p1 = i + n * (count-1);
                    int p2 = i + n * 0;
                    int p3 = i+1 + n * 0;
                    int p4 = i+1 + n * (count-1);
                    if constexpr (isTri.value) {
                        prim->tris[(i * count + count-1)*2] = {p1, p2, p3};
                        prim->tris[(i * count + count-1)*2+1] = {p1, p3, p4};
                    } else {
                        prim->quads[i * count + count-1] = {p1, p2, p3, p4};
                    }
                }

            });

            prim->verts.foreach_attr([&] (auto const &key, auto &attr) {
            });

        }

        set_output("prim", std::move(prim));
    }
};


ZENDEFNODE(PrimitiveLineSolidify, {
    {
    {"PrimitiveObject", "prim"},
    {"int", "count", "4"},
    {"float", "radius", "0.1"},
    {"string", "radiusAttr", ""},
    {"bool", "isTri", "1"},
    },
    {
    {"PrimitiveObject", "prim"},
    },
    {
    },
    {"primitive"},
});

}
