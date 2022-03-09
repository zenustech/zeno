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
#include "PrimitiveLineSort.h"
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
        bool sealEnd = get_input2<bool>("sealEnd");
        
        if (get_input2<bool>("lineSort"))
            primLineSort(prim.get());

        intptr_t n = prim->verts.size();
        if (n >= 2 && count >= 2) {

            prim->lines.clear();

            if (sealEnd) {
                prim->verts.resize(count * n + 2);
                prim->verts[count * n + 0] = prim->verts[0];
                prim->verts[count * n + 1] = prim->verts[n - 1];
            } else {
                prim->verts.resize(count * n);
            }

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
                sinang[a] = std::sin(ang) * 1;
                cosang[a] = std::cos(ang) * 1;
            }

            std::vector<vec3f> bidirections(n);
            orthonormal first_orb(directions[0]);
            directions[0] = first_orb.tangent;
            bidirections[0] = first_orb.bitangent;
            vec3f last_tangent = directions[0];
            for (intptr_t i = 1; i < n; i++) {
                orthonormal orb(directions[i], last_tangent);
                last_tangent = directions[i] = orb.tangent;
                bidirections[i] = orb.bitangent;
                //printf("%f %f %f\n", directions[i][0], directions[i][1], directions[i][2]);
                //printf("%f %f %f\n", bidirections[i][0], bidirections[i][1], bidirections[i][2]);
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
                auto tangent = directions[i];
                auto bitangent = bidirections[i];
                for (int a = 0; a < count; a++) {
                    auto offs = tangent * sinang[a] + bitangent * cosang[a];
                    if (prim->has_attr(radiusAttr))
                        offs *= prim->attr<float>(radiusAttr)[i];
                    else offs *= radius;
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

            if (sealEnd) {
                for (int a = 0; a < count - 1; a++) {
                    prim->tris.emplace_back(count * n, n * a, n * (a+1));
                    prim->tris.emplace_back(count * n + 1, n-1 + n * a, n-1 + n * (a+1));
                }
                prim->tris.emplace_back(count * n, n * (count-1), 0);
                prim->tris.emplace_back(count * n + 1, n-1 + n * (count-1), n-1);
                prim->tris.update();
            }

            prim->verts.foreach_attr([&] (auto const &key, auto &attr) {
                for (int a = 1; a < count; a++) {
                    intptr_t na = n * a;
                    for (intptr_t i = 0; i < n; i++) {
                        attr[i + na] = attr[i];
                    }
                }
                if (sealEnd) {
                    attr[count * n + 0] = attr[0];
                    attr[count * n + 1] = attr[n - 1];
                }
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
    {"bool", "sealEnd", "1"},
    {"bool", "lineSort", "1"},
    },
    {
    {"PrimitiveObject", "prim"},
    },
    {
    },
    {"primitive"},
});

}
