#include <zeno/types/PrimitiveObject.h>
#include <zeno/zeno.h>
#include <zeno/funcs/PrimitiveUtils.h>
using namespace zeno;

struct buildCloth : zeno::INode {
    void build(PrimitiveObject* prim)
    {
        //根据点重建边、面、quads
        primWireframe(prim);
        //         struct segment_less {
        //     bool operator()(vec2i const &a, vec2i const &b) const {
        //         return std::make_pair(std::min(a[0], a[1]), std::max(a[0], a[1]))
        //             < std::make_pair(std::min(b[0], b[1]), std::max(b[0], b[1]));
        //     }
        // };
        // std::set<vec2i, segment_less> segments;
        // auto append = [&] (int i, int j) {
        //     segments.emplace(i, j);
        // };
        // for (auto const &ind: prim->lines) {
        //     append(ind[0], ind[1]);
        // }
        // for (auto const &ind: prim->tris) {
        //     append(ind[0], ind[1]);
        //     append(ind[1], ind[2]);
        //     append(ind[2], ind[0]);
        // }
        // for (auto const &ind: prim->quads) {
        //     append(ind[0], ind[1]);
        //     append(ind[1], ind[2]);
        //     append(ind[2], ind[3]);
        //     append(ind[3], ind[0]);
        // }
    }

    virtual void apply() override {
        auto prim = get_input<PrimitiveObject>("prim");

        build(prim.get());

        set_output("outPrim", std::move(prim));
    }
};

ZENDEFNODE(
    buildCloth,{
        {"prim"},
        {"outPrim"},
        {},
        {"PBD"}
    }
)