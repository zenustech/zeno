#include "Structures.hpp"
#include "Utils.hpp"
#include "zensim/cuda/execution/ExecutionPolicy.cuh"
#include "zensim/omp/execution/ExecutionPolicy.hpp"
#include "zensim/types/Property.h"
#include "zensim/zpc_tpls/fmt/color.h"
#include "zensim/zpc_tpls/fmt/format.h"
#include <zeno/types/NumericObject.h>

namespace zeno {

struct MakeZSBuckets : zeno::INode {
    void apply() override {
        float radius = get_input<NumericObject>("radius")->get<float>();
        float radiusMin = has_input("radiusMin") ? get_input<NumericObject>("radiusMin")->get<float>() : 0.f;
        auto &pars = get_input<ZenoParticles>("ZSParticles")->getParticles();

        auto out = std::make_shared<ZenoIndexBuckets>();
        auto &ibs = out->get();

        using namespace zs;
        auto cudaPol = cuda_exec();
        spatial_hashing(cudaPol, pars, radius + radius, ibs);

        fmt::print("done building index buckets with {} entries, {} buckets\n", ibs.numEntries(), ibs.numBuckets());

        set_output("ZSIndexBuckets", std::move(out));
    }
};

ZENDEFNODE(MakeZSBuckets, {{{"ZSParticles"}, {"numeric:float", "radius"}, {"numeric:float", "radiusMin"}},
                           {"ZSIndexBuckets"},
                           {},
                           {"MPM"}});

struct MakeZSLBvh : zeno::INode {
    template <typename TileVecT>
    void buildBvh(zs::CudaExecutionPolicy &pol, TileVecT &verts, typename TileVecT::value_type radius,
                  ZenoLinearBvh::lbvh_t &bvh) {
        using namespace zs;
        using bv_t = typename ZenoLinearBvh::lbvh_t::Box;
        constexpr auto space = execspace_e::cuda;
        Vector<bv_t> bvs{verts.get_allocator(), verts.size()};
        pol(range(verts.size()),
            [verts = proxy<space>({}, verts), bvs = proxy<space>(bvs), radius] ZS_LAMBDA(int i) mutable {
                auto x = verts.template pack<3>("x", i);
                bv_t bv{x - radius, x + radius};
                bvs[i] = bv;
            });
        bvh.build(pol, bvs);
    }
    void apply() override {
        auto pars = get_input<ZenoParticles>("ZSParticles");
        float radius = get_input<NumericObject>("radius")->get<float>();

        auto out = std::make_shared<ZenoLinearBvh>();
        auto &bvh = out->get();

        using namespace zs;
        auto cudaPol = cuda_exec();
        if (pars->hasImage(ZenoParticles::s_particleTag))
            buildBvh(cudaPol, pars->getParticles<true>(), radius, bvh);
        else
            buildBvh(cudaPol, pars->getParticles(), radius, bvh);
        out->thickness = radius;

        set_output("ZSLBvh", std::move(out));
    }
};

ZENDEFNODE(MakeZSLBvh, {{{"ZSParticles"}, {"numeric:float", "radius"}}, {"ZSLBvh"}, {}, {"MPM"}});

} // namespace zeno