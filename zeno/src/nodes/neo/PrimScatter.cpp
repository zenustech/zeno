#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/StringObject.h>
#include <zeno/types/NumericObject.h>
#include <random>
#include <cmath>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace zeno {

static float area(vec3f p1, vec3f p2, vec3f&p3) {
    zeno::vec3f e1 = p3 - p1;
    zeno::vec3f e2 = p2 - p1;
    zeno::vec3f areavec = cross(e1, e2);
    return 0.5f * sqrt(dot(areavec, areavec));
}

static vec3f calcbarycentric(vec3f p, vec3f vert1, vec3f vert2, vec3f vert3) {
    float a1 = area(p, vert2, vert3);
    float a2 = area(p, vert1, vert3);
    float a = area(vert1, vert2, vert3);
    float w1 = a1 / (a+1e-7f);
    float w2 = a2 / (a+1e-7f);
    float w3 = 1 - w1 - w2;
    return {w1, w2, w3};
}

ZENO_API std::shared_ptr<PrimitiveObject> primScatter(
    PrimitiveObject *prim, std::string type, int npoints, bool interpAttrs, int seed) {
    auto retprim = std::make_shared<PrimitiveObject>();

    if (seed == -1) seed = std::random_device{}();
    float total = 0;
    std::vector<float> cdf;
    
    if (type == "tris") {
        if (!prim->tris.size()) return retprim;
        cdf.resize(prim->tris.size());
        for (size_t i = 0; i < prim->tris.size(); i++) {
            auto const &ind = prim->tris[i];
            auto a = prim->verts[ind[0]];
            auto b = prim->verts[ind[1]];
            auto c = prim->verts[ind[2]];
            auto area = length(cross(c - a, c - b));
            total += area;
            cdf[i] = total;
        }
    } else if (type == "lines") {
        if (!prim->lines.size()) return retprim;
        cdf.resize(prim->lines.size());
        for (size_t i = 0; i < prim->lines.size(); i++) {
            auto const &ind = prim->lines[i];
            auto a = prim->lines[ind[0]];
            auto b = prim->lines[ind[1]];
            auto area = length(a - b);
            total += area;
            cdf[i] = total;
        }
    }

    auto inv_total = 1 / total;
    for (size_t i = 0; i < cdf.size(); i++) {
        cdf[i] *= inv_total;
    }

    retprim->verts.resize(npoints);

    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> unif;

    if (!prim->verts.num_attrs()) {
        interpAttrs = false;
    }

    if (type == "tris") {
        for (std::size_t i = 0; i < npoints; i++) {
            auto val = unif(gen);
            auto it = std::lower_bound(cdf.begin(), cdf.end(), val);
            std::size_t index = it - cdf.begin();
            index = std::min(index, prim->tris.size() - 1);
            auto const &ind = prim->tris[index];
            auto a = prim->verts[ind[0]];
            auto b = prim->verts[ind[1]];
            auto c = prim->verts[ind[2]];
            auto r1 = std::sqrt(unif(gen));
            auto r2 = unif(gen);
            auto w1 = 1 - r1;
            auto w2 = r1 * (1 - r2);
            auto w3 = r1 * r2;
            auto p = w1 * a + w2 * b + w3 * c;
            retprim->verts[i] = p;
            if (interpAttrs) {
                prim->verts.foreach_attr([&] (auto const &key, auto const &arr) {
                    using T = std::decay_t<decltype(arr[0])>;
                    auto &retarr = retprim->add_attr<T>(key);
                    auto a = arr[ind[0]];
                    auto b = arr[ind[1]];
                    auto c = arr[ind[2]];
                    auto p = w1 * a + w2 * b + w3 * c;
                    retarr[i] = p;
                });
            }
        }
    } else if (type == "lines") {
        for (std::size_t i = 0; i < npoints; i++) {
            auto val = unif(gen);
            auto it = std::lower_bound(cdf.begin(), cdf.end(), val);
            std::size_t index = it - cdf.begin();
            index = std::min(index, prim->lines.size() - 1);
            auto const &ind = prim->lines[index];
            auto a = prim->verts[ind[0]];
            auto b = prim->verts[ind[1]];
            auto r1 = unif(gen);
            auto p = a * (1 - r1) + b * r1;
            retprim->verts[i] = p;
            if (interpAttrs) {
                prim->verts.foreach_attr([&] (auto const &key, auto const &arr) {
                    using T = std::decay_t<decltype(arr[0])>;
                    auto &retarr = retprim->add_attr<T>(key);
                    auto a = arr[ind[0]];
                    auto b = arr[ind[1]];
                    auto p = a * (1 - r1) + b * r1;
                    retarr[i] = p;
                });
            }
        }
    }

    return retprim;
}

namespace {

struct PrimScatter : INode {
    virtual void apply() const {
        primScatter(prim, type, npoints, interpAttrs, seed);
    }
};

}

}
