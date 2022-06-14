#include <zeno/zeno.h>
#include <zeno/funcs/PrimitiveUtils.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/StringObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/para/parallel_for.h>
#include <zeno/para/parallel_scan.h>
#include <zeno/utils/variantswitch.h>
#include <zeno/utils/wangsrng.h>
#include <random>
#include <cmath>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace zeno {

ZENO_API std::shared_ptr<PrimitiveObject> primScatter(
    PrimitiveObject *prim, std::string type, int npoints, bool interpAttrs, int seed) {
    auto retprim = std::make_shared<PrimitiveObject>();

    if (seed == -1) seed = std::random_device{}();
    std::vector<float> cdf;
    
    if (type == "tris") {
        if (!prim->tris.size()) return retprim;
        cdf.resize(prim->tris.size());
        parallel_inclusive_scan_sum(prim->tris.begin(), prim->tris.end(), cdf.begin(), [&] (auto const &ind) {
            auto a = prim->verts[ind[0]];
            auto b = prim->verts[ind[1]];
            auto c = prim->verts[ind[2]];
            auto area = length(cross(c - a, c - b));
            return area;
        });
    } else if (type == "lines") {
        if (!prim->lines.size()) return retprim;
        cdf.resize(prim->lines.size());
        parallel_inclusive_scan_sum(prim->lines.begin(), prim->lines.end(), cdf.begin(), [&] (auto const &ind) {
            auto a = prim->lines[ind[0]];
            auto b = prim->lines[ind[1]];
            auto area = length(a - b);
            return area;
        });
    }

    auto inv_total = 1 / cdf.back();
    parallel_for((size_t)0, cdf.size(), [&] (size_t i) {
        cdf[i] *= inv_total;
    });

    retprim->verts.resize(npoints);

    if (!prim->verts.num_attrs()) {
        interpAttrs = false;
    }
    if (interpAttrs) {
        prim->verts.foreach_attr([&] (auto const &key, auto const &arr) {
            using T = std::decay_t<decltype(arr[0])>;
            retprim->add_attr<T>(key);
        });
    }

    if (type == "tris") {
        parallel_for((size_t)0, (size_t)npoints, [&] (size_t i) {
            wangsrng rng(seed, i);
            auto val = rng.next_float();
            auto it = std::lower_bound(cdf.begin(), cdf.end(), val);
            std::size_t index = it - cdf.begin();
            index = std::min(index, prim->tris.size() - 1);
            auto const &ind = prim->tris[index];
            auto a = prim->verts[ind[0]];
            auto b = prim->verts[ind[1]];
            auto c = prim->verts[ind[2]];
            auto r1 = std::sqrt(rng.next_float());
            auto r2 = rng.next_float();
            auto w1 = 1 - r1;
            auto w2 = r1 * (1 - r2);
            auto w3 = r1 * r2;
            auto p = w1 * a + w2 * b + w3 * c;
            retprim->verts[i] = p;
            if (interpAttrs) {
                prim->verts.foreach_attr([&] (auto const &key, auto const &arr) {
                    using T = std::decay_t<decltype(arr[0])>;
                    auto &retarr = retprim->attr<T>(key);
                    auto a = arr[ind[0]];
                    auto b = arr[ind[1]];
                    auto c = arr[ind[2]];
                    auto p = w1 * a + w2 * b + w3 * c;
                    retarr[i] = p;
                });
            }
        });
    } else if (type == "lines") {
        parallel_for((size_t)0, (size_t)npoints, [&] (size_t i) {
            wangsrng rng(seed, i);
            auto val = rng.next_float();
            auto it = std::lower_bound(cdf.begin(), cdf.end(), val);
            std::size_t index = it - cdf.begin();
            index = std::min(index, prim->lines.size() - 1);
            auto const &ind = prim->lines[index];
            auto a = prim->verts[ind[0]];
            auto b = prim->verts[ind[1]];
            auto r1 = rng.next_float();
            auto p = a * (1 - r1) + b * r1;
            retprim->verts[i] = p;
            if (interpAttrs) {
                prim->verts.foreach_attr([&] (auto const &key, auto const &arr) {
                    using T = std::decay_t<decltype(arr[0])>;
                    auto &retarr = retprim->attr<T>(key);
                    auto a = arr[ind[0]];
                    auto b = arr[ind[1]];
                    auto p = a * (1 - r1) + b * r1;
                    retarr[i] = p;
                });
            }
        });
    }

    return retprim;
}

namespace {

struct PrimScatter : INode {
    virtual void apply() override {
        auto prim = get_input<PrimitiveObject>("prim");
        auto type = get_input2<std::string>("type");
        auto npoints = get_input2<int>("npoints");
        auto interpAttrs = get_input2<bool>("interpAttrs");
        auto seed = get_input2<int>("seed");
        auto retprim = primScatter(prim.get(), type, npoints, interpAttrs, seed);
        set_output("parsPrim", retprim);
    }
};

ZENO_DEFNODE(PrimScatter)({
    {
        {"prim"},
        {"enum tris lines", "type", "tris"},
        {"int", "npoints", "100"},
        {"int", "interpAttrs", "1"},
        {"int", "seed", "-1"},
    },
    {
        {"parsPrim"},
    },
    {},
    {"primitive"},
});

}

}
