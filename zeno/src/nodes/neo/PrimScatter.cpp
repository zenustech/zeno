#include <zeno/zeno.h>
#include <zeno/funcs/PrimitiveUtils.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/StringObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/para/parallel_for.h>
#include <zeno/para/parallel_scan.h>
#define ZENO_NOTICKTOCK
#include <zeno/utils/ticktock.h>
#include <zeno/utils/variantswitch.h>
#include <zeno/utils/wangsrng.h>
#include <zeno/utils/tuple_hash.h>
#include <zeno/utils/log.h>
#include <unordered_map>
#include <random>
#include <cmath>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace zeno {

template <class T>
static void revamp_vector(std::vector<T> &arr, std::vector<int> const &revamp) {
    std::vector<T> newarr(arr.size());
    for (int i = 0; i < revamp.size(); i++) {
        newarr[i] = arr[revamp[i]];
    }
    std::swap(arr, newarr);
}

static void primPossionFilter(PrimitiveObject *prim, float minRadius) {
    if (minRadius <= 0) return;

    TICK(possion);
    float invRadius = 1.f / minRadius;
    std::unordered_map<vec3i, std::vector<int>, tuple_hash, tuple_equal> lut;
    for (int i = 0; i < prim->verts.size(); i++) {
        vec3i ipos(prim->verts[i] * invRadius);
        lut[ipos].push_back(i);
    }

    std::vector<uint8_t> erased(prim->verts.size());
    for (int i = 0; i < prim->verts.size(); i++) {
        if (erased[i])
            continue;
        vec3i ipos(prim->verts[i] * invRadius);
        erased[i] = [&] {
            for (int dz = -1; dz <= 1; dz++) {
                for (int dy = -1; dy <= 1; dy++) {
                    for (int dx = -1; dx <= 1; dx++) {
                        auto it = lut.find(ipos + vec3i(dx, dy, dz));
                        if (it != lut.end()) {
                            for (int j: it->second) {
                                if (j == i || erased[j])
                                    continue;
                                auto dis = prim->verts[i] - prim->verts[j];
                                if (length(dis) < minRadius)
                                    return true;
                            }
                        }
                    }
                }
            }
            return false;
        }();
    }

    std::vector<int> revamp(prim->verts.size());
    int nrevamp = 0;
    for (int i = 0; i < erased.size(); i++) {
        if (!erased[i])
            revamp[nrevamp++] = i;
    }
    revamp.resize(nrevamp);

    prim->verts.forall_attr([&] (auto const &key, auto &arr) {
        revamp_vector(arr, revamp);
    });
    prim->verts.resize(nrevamp);
    TOCK(possion);
}

ZENO_API std::shared_ptr<PrimitiveObject> primScatter(
    PrimitiveObject *prim, std::string type, std::string denAttr, float density, float minRadius, bool interpAttrs, int seed) {
    auto retprim = std::make_shared<PrimitiveObject>();

    if (seed == -1) seed = std::random_device{}();
    bool hasDenAttr = !denAttr.empty();
    TICK(scatter);
    
    std::vector<float> cdf;
    if (type == "tris") {
        if (!prim->tris.size()) return retprim;
        cdf.resize(prim->tris.size());
        parallel_inclusive_scan_sum(prim->tris.begin(), prim->tris.end(), cdf.begin(), [&] (auto const &ind) {
            auto a = prim->verts[ind[0]];
            auto b = prim->verts[ind[1]];
            auto c = prim->verts[ind[2]];
            auto area = length(cross(c - a, c - b));
            if (hasDenAttr) {
                auto &den = prim->verts.attr<float>(denAttr);
                auto da = den[ind[0]];
                auto db = den[ind[1]];
                auto dc = den[ind[2]];
                area *= std::abs(da + db + dc) / 3;
            }
            return area;
        });
    } else if (type == "lines") {
        if (!prim->lines.size()) return retprim;
        cdf.resize(prim->lines.size());
        parallel_inclusive_scan_sum(prim->lines.begin(), prim->lines.end(), cdf.begin(), [&] (auto const &ind) {
            auto a = prim->lines[ind[0]];
            auto b = prim->lines[ind[1]];
            auto area = length(a - b);
            if (hasDenAttr) {
                auto &den = prim->verts.attr<float>(denAttr);
                auto da = den[ind[0]];
                auto db = den[ind[1]];
                area *= std::abs(da + db) / 2;
            }
            return area;
        });
    }
    if (cdf.empty()) return retprim;

    auto npoints = (int)std::rint(cdf.back() * density);
    auto inv_total = 1 / cdf.back();
    parallel_for((size_t)0, cdf.size(), [&] (size_t i) {
        cdf[i] *= inv_total;
    });
    zeno::log_info("PrimScatter total npoints {}", npoints);

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
            size_t index = it - cdf.begin();
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
            size_t index = it - cdf.begin();
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

    TOCK(scatter);
    primPossionFilter(retprim.get(), minRadius);

    return retprim;
}

namespace {

struct PrimScatter : INode {
    virtual void apply() override {
        auto prim = get_input<PrimitiveObject>("prim");
        auto type = get_input2<std::string>("type");
        auto denAttr = get_input2<std::string>("denAttr");
        auto density = get_input2<float>("density");
        auto minRadius = get_input2<float>("minRadius");
        auto interpAttrs = get_input2<bool>("interpAttrs");
        auto seed = get_input2<int>("seed");
        auto retprim = primScatter(prim.get(), type, denAttr, density, minRadius, interpAttrs, seed);
        set_output("parsPrim", retprim);
    }
};

ZENO_DEFNODE(PrimScatter)({
    {
        {gParamType_Primitive, "prim", "", zeno::Socket_ReadOnly},
        {"enum tris lines", "type", "tris"},
        {gParamType_String, "denAttr", ""},
        {gParamType_Float, "density", "100"},
        {gParamType_Float, "minRadius", "0"},
        {gParamType_Bool, "interpAttrs", "1"},
        {gParamType_Int, "seed", "-1"},
    },
    {
        {gParamType_Primitive, "parsPrim"},
    },
    {},
    {"primitive"},
});

}

}
