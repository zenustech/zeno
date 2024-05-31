#include <stdexcept>
#include <zeno/funcs/PrimitiveUtils.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/StringObject.h>
#include <zeno/zeno.h>

namespace zeno {

static void BarycentricInterp(PrimitiveObject *_dst, const PrimitiveObject *_src, int i, int v0, int v1, int v2,
                              zeno::vec3f &pdst, zeno::vec3f &w, std::string &idTag, std::string &weightTag) {
    for (auto key : _src->attr_keys()) {
        if (key != "pos" && key != idTag && key != weightTag)
            std::visit(
                [i, v0, v1, v2, &pdst, &w](auto &&dst, auto &&src) {
                    using DstT = std::remove_cv_t<std::remove_reference_t<decltype(dst)>>;
                    using SrcT = std::remove_cv_t<std::remove_reference_t<decltype(src)>>;
                    if constexpr (std::is_same_v<DstT, SrcT>) {
                        auto val1 = src[v0];
                        auto val2 = src[v1];
                        auto val3 = src[v2];
                        auto val = w[0] * val1 + w[1] * val2 + w[2] * val3;

                        dst[i] = val;
                    } else {
                        throw std::runtime_error("the same attr of both primitives are of different types.");
                    }
                },
                _dst->attr(key), _src->attr(key));
    }
}
struct PrimBarycentricInterp : INode {
    virtual void apply() override {
        auto points = get_input<PrimitiveObject>("Particles");
        auto prim = get_input<PrimitiveObject>("MeshPrim");
        auto idTag = get_input2<std::string>("triIdTag");
        auto weightTag = get_input2<std::string>("weightTag");

        auto triIndex = points->attr<float>(idTag);
        auto wIndex = points->attr<zeno::vec3f>(weightTag);

        for (auto key : prim->attr_keys()) {
            if (key != "pos" && key != idTag && key != weightTag)
                std::visit(
                    [&points, key](auto &&ref) {
                        using T = std::remove_cv_t<std::remove_reference_t<decltype(ref[0])>>;
                        points->add_attr<T>(key);
                    },
                    prim->attr(key));
        }

        #pragma omp parallel for
        for (auto index = 0; index < points->size(); ++index) {
            auto tidx = prim->tris[(int)triIndex[index]];
            int v0 = (int)(tidx[0]), v1 = (int)(tidx[1]), v2 = (int)(tidx[2]);
            vec3f w = wIndex[index];
            BarycentricInterp(points.get(), prim.get(), index, v0, v1, v2, points->verts[index], w, idTag, weightTag);
        }

        set_output("Particles", get_input("Particles"));
    }
};

ZENDEFNODE(PrimBarycentricInterp, {
                                      {
                                          {"", "Particles", "", zeno::Socket_ReadOnly},
                                          {"", "MeshPrim", "", zeno::Socket_ReadOnly},
                                          {"string", "triIdTag", "bvh_id"},
                                          {"string", "weightTag", "bvh_ws"},
                                      },
                                      {"Particles"},
                                      {},
                                      {"primitive"},
                                  });

} // namespace zeno