#include "Structures.hpp"
#include "zensim/geometry/AnalyticLevelSet.h"
#include "zensim/omp/execution/ExecutionPolicy.hpp"
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/utils/vec.h>
#include <zeno/zeno.h>

namespace zeno {

static zs::Vector<zs::AABBBox<3, float>>
retrieve_bounding_volumes(zs::OmpExecutionPolicy &pol, const std::vector<vec3f> &pos, const std::vector<vec3i> &tris) {
    using namespace zs;
    using T = float;
    using bv_t = AABBBox<3, T>;
    constexpr auto space = execspace_e::openmp;
    zs::Vector<bv_t> ret{tris.size()};
    pol(range(tris.size()), [&, bvs = proxy<space>(ret)](int i) {
        using vec3 = zs::vec<float, 3>;
        auto inds = tris[i];
        auto x0 = vec3::from_array(pos[inds[0]]);
        bv_t bv{x0, x0};
        for (int d = 1; d != 3; ++d)
            merge(bv, vec3::from_array(pos[inds[d]]));
        bvs[i] = bv;
    });
    return ret;
}

struct PrimitiveProject : INode {
    virtual void apply() override {
        using bvh_t = zs::LBvh<3, int, float>;
        using bv_t = typename bvh_t::Box;

        auto prim = get_input<PrimitiveObject>("prim");
        auto targetPrim = get_input<PrimitiveObject>("targetPrim");
        auto limit = get_input2<float>("limit");
        auto nrmAttr = get_input2<std::string>("nrmAttr");
        auto side = get_input2<std::string>("side");

        int sideNo = 0;
        if (side == "closest")
            sideNo = 0;
        else if (side == "farthest")
            sideNo = 1;

        using namespace zs;
        constexpr auto space = execspace_e::openmp;
        auto pol = omp_exec();
        auto &pos = prim->attr<zeno::vec3f>("pos");
        const auto &targetPos = targetPrim->attr<zeno::vec3f>("pos");
        const auto &tris = targetPrim->tris.values;

        bvh_t targetBvh;
        auto tBvs = retrieve_bounding_volumes(pol, targetPos, tris);
        targetBvh.build(pol, tBvs);

        /// @note cut off hits farther than distance [limit]
        if (limit <= 0)
            limit = std::numeric_limits<float>::infinity();

        auto const &nrm = prim->attr<vec3f>(nrmAttr);

        pol(range(pos.size()), [&, bvh = proxy<space>(targetBvh)](size_t i) {
            using vec3 = zs::vec<float, 3>;
            auto ro = vec3::from_array(pos[i]);
            auto rd = vec3::from_array(nrm[i]).normalized();
            float dist = 0;
            bvh.ray_intersect(ro, rd, [&](int triNo) {
                auto tri = tris[triNo];
                auto t0 = vec3::from_array(targetPos[tri[0]]);
                auto t1 = vec3::from_array(targetPos[tri[1]]);
                auto t2 = vec3::from_array(targetPos[tri[2]]);
                if (auto d = ray_tri_intersect(ro, rd, t0, t1, t2); d < limit && d > dist) {
                    dist = d;
                }
            });
            pos[i] = (ro + dist * rd).to_array();
        });

        set_output("prim", std::move(prim));
    }
};

ZENDEFNODE(PrimitiveProject, {
                                 {
                                     {"PrimitiveObject", "prim"},
                                     {"PrimitiveObject", "targetPrim"},
                                     {"string", "nrmAttr", "nrm"},
                                     {"float", "limit", "0"},
                                     {"enum closest farthest", "side", "farthest"},
                                 },
                                 {
                                     {"PrimitiveObject", "prim"},
                                 },
                                 {},
                                 {"primitive"},
                             });

} // namespace zeno