#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/StringObject.h>
#include <zeno/types/NumericObject.h>
#include <random>
#include <cmath>
#include <zeno/types/PrimitiveTools.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace zeno {

struct PrimitiveScatter : INode {
    virtual void apply() override {
        auto prim = get_input<PrimitiveObject>("prim");
        auto npoints = get_input<NumericObject>("npoints")->get<int>();
        auto seed = get_input<NumericObject>("seed")->get<int>();
        auto type = get_param<std::string>("type");
        auto retprim = std::make_shared<PrimitiveObject>();

        if (type == "tris" && prim->tris.size()) {
            float total = 0;
            std::vector<float> cdf(prim->tris.size());
            for (size_t i = 0; i < prim->tris.size(); i++) {
                auto const &ind = prim->tris[i];
                //std::cout << '?' << i << ' ' << ind[0] << std::endl;
                auto a = prim->verts[ind[0]];
                auto b = prim->verts[ind[1]];
                auto c = prim->verts[ind[2]];
                auto area = length(cross(c - a, c - b));
                total += area;
                cdf[i] = total;
            }
            auto inv_total = 1 / total;
            for (size_t i = 0; i < cdf.size(); i++) {
                cdf[i] *= inv_total;
            }

            std::mt19937 gen(seed);
            std::uniform_real_distribution<float> unif;

            retprim->verts.resize(npoints);
            for(auto key:prim->attr_keys())
            { 
                if(key!="pos")
                std::visit([&retprim, key](auto &&ref) {
                                using T = std::remove_cv_t<std::remove_reference_t<decltype(ref[0])>>;
                                retprim->add_attr<T>(key);
                            }, prim->attr(key));
            }
#if defined(__GNUC__) || defined(__clang__)
#pragma omp simd
#endif
            for (size_t i = 0; i < npoints; i++) {
                auto val = unif(gen);
                auto it = std::lower_bound(cdf.begin(), cdf.end(), val);
                size_t index = it - cdf.begin();
                index = std::min(index, prim->tris.size() - 1);
                auto const &ind = prim->tris[index];
                //std::cout << '!' << index << ' ' << ind[0] << std::endl;
                auto a = prim->verts[ind[0]];
                auto b = prim->verts[ind[1]];
                auto c = prim->verts[ind[2]];
                auto r1 = std::sqrt(unif(gen));
                auto r2 = unif(gen);
                auto p = (1 - r1) * a + (r1 * (1 - r2)) * b + (r1 * r2) * c;
                retprim->verts[i] = p;
                BarycentricInterpPrimitive(retprim.get(), prim.get(), i, ind[0], ind[1], ind[2], p, a, b, c);
            }

        } else if (type == "lines" && prim->lines.size()) {
            float total = 0;
            std::vector<float> cdf(prim->lines.size());
            for (size_t i = 0; i < prim->lines.size(); i++) {
                auto const &ind = prim->lines[i];
                //std::cout << '?' << i << ' ' << ind[0] << std::endl;
                auto a = prim->verts[ind[0]];
                auto b = prim->verts[ind[1]];
                auto area = length(b - a);
                total += area;
                cdf[i] = total;
            }
            auto inv_total = 1 / total;
            for (size_t i = 0; i < cdf.size(); i++) {
                cdf[i] *= inv_total;
            }

            std::mt19937 gen(seed);
            std::uniform_real_distribution<float> unif;

            retprim->verts.resize(npoints);
            for (size_t i = 0; i < npoints; i++) {
                auto val = unif(gen);
                auto it = std::lower_bound(cdf.begin(), cdf.end(), val);
                size_t index = it - cdf.begin();
                index = std::min(index, prim->lines.size() - 1);
                auto const &ind = prim->lines[index];
                //std::cout << '!' << index << ' ' << ind[0] << std::endl;
                auto a = prim->verts[ind[0]];
                auto b = prim->verts[ind[1]];
                auto r1 = unif(gen);
                auto p = a * (1 - r1) + b * r1;
                retprim->verts[i] = p;
            }

        }
        

        set_output("points", std::move(retprim));
    }
};


ZENDEFNODE(PrimitiveScatter, {
    {
    {"PrimitiveObject", "prim"},
    {"int", "npoints", "100"},
    {"int", "seed", "0"},
    },
    {
    {"PrimitiveObject", "points"},
    },
    {
    {"enum tris lines", "type", "tris"},
    },
    {"primitive"},
});

}
