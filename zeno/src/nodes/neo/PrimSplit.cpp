#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/para/parallel_reduce.h>

namespace zeno {

struct PrimSplit : INode {
  virtual void apply() override {
    auto prim = get_input<PrimitiveObject>("prim");

    auto polyred = parallel_reduce_sum(prim->polys.begin(), prim->polys.end(), [] (auto const &pol) { return pol[1]; });
    auto n = prim->points.size() + prim->lines.size() * 2 + prim->tris.size() * 3 + prim->quads.size() * 4 + polyred;
    prim->verts.forall_attr<AttrAcceptAll>([&] (auto &, auto &arr) {
        auto oldarr = std::move(arr);
        arr.resize(n);
        size_t b = 0;
        for (size_t i = 0; i < prim->points.size(); i++) {
            auto ind = prim->points[i];
            arr[b + i * 1 + 0] = oldarr[ind];
        }
        b += prim->points.size();
        for (size_t i = 0; i < prim->lines.size(); i++) {
            auto ind = prim->lines[i];
            arr[b + i * 2 + 0] = oldarr[ind[0]];
            arr[b + i * 2 + 1] = oldarr[ind[1]];
        }
        b += prim->lines.size() * 2;
        for (size_t i = 0; i < prim->tris.size(); i++) {
            auto ind = prim->tris[i];
            arr[b + i * 3 + 0] = oldarr[ind[0]];
            arr[b + i * 3 + 1] = oldarr[ind[1]];
            arr[b + i * 3 + 2] = oldarr[ind[2]];
        }
        b += prim->tris.size() * 3;
        for (size_t i = 0; i < prim->quads.size(); i++) {
            auto ind = prim->quads[i];
            arr[b + i * 4 + 0] = oldarr[ind[0]];
            arr[b + i * 4 + 1] = oldarr[ind[1]];
            arr[b + i * 4 + 2] = oldarr[ind[2]];
            arr[b + i * 4 + 3] = oldarr[ind[3]];
        }
        b += prim->quads.size() * 4;
        for (size_t i = 0; i < prim->polys.size(); i++) {
            auto pol = prim->polys[i];
            for (size_t l = pol[0]; l < pol[0] + pol[1]; l++) {
                arr[b++] = oldarr[prim->loops[l]];
            }
        }
    });
    prim->verts.resize(n);

    size_t b = 0;
    for (size_t i = 0; i < prim->points.size(); i++) {
        prim->points[i] = i + b;
    }
    b += prim->points.size();
    for (size_t i = 0; i < prim->lines.size(); i++) {
        prim->lines[i] = zeno::vec2i(0, 1) + i * 2 + b;
    }
    b += prim->lines.size() * 2;
    for (size_t i = 0; i < prim->tris.size(); i++) {
        prim->tris[i] = zeno::vec3i(0, 1, 2) + i * 3 + b;
    }
    b += prim->tris.size() * 3;
    for (size_t i = 0; i < prim->quads.size(); i++) {
        prim->quads[i] = zeno::vec4i(0, 1, 2, 3) + i * 4 + b;
    }
    b += prim->quads.size() * 4;
    for (size_t i = 0; i < prim->polys.size(); i++) {
        auto pol = prim->polys[i];
        for (size_t l = pol[0]; l < pol[0] + pol[1]; l++) {
            prim->loops[l] = b++;
        }
    }

    set_output("prim", get_input("prim"));
  }
};

ZENDEFNODE(PrimSplit, {
    {{"", "prim", "", zeno::Socket_ReadOnly}},
    {"prim"},
    {},
    {"primitive"},
});

}
