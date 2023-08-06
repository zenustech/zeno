#include "Structures.hpp"
#include "zensim/cuda/execution/ExecutionPolicy.cuh"
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <random>
#include <zeno/types/DummyObject.h>
#include <zeno/types/ListObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/UserData.h>
#include <zeno/utils/log.h>
#include <zeno/utils/parallel_reduce.h>
#include <zeno/utils/vec.h>
#include <zeno/zeno.h>

namespace zeno {

static constexpr const char *zs_bvh_tag = "zsbvh";

template <typename VPosRange, typename VIndexRange, typename Bv, int codim = 3>
static void retrieve_bounding_volumes(zs::CudaExecutionPolicy &pol, VPosRange &&posR, VIndexRange &&idxR,
                                      zs::Vector<Bv> &ret) {
    using namespace zs;
    using bv_t = Bv;
    constexpr auto space = execspace_e::cuda;
    pol(range(range_size(idxR)),
        [tris = idxR.begin(), pos = posR.begin(), bvs = view<space>(ret)] ZS_LAMBDA(int ei) mutable {
            auto inds = tris[ei];
            auto x0 = pos[inds[0]];
            bv_t bv{x0, x0};
            for (int d = 1; d != 3; ++d)
                merge(bv, pos[inds[d]]);
            bvs[ei] = bv;
        });
}

struct ComputeVertexAO : INode {
    void apply() override {
        auto prim = get_input2<PrimitiveObject>("prim");
        auto scene = get_input2<PrimitiveObject>("scene");
        auto nrmTag = get_input2<std::string>("nrm_tag");
        auto niters = get_input2<int>("sample_iters");

        using namespace zs;
        using v3 = zs::vec<float, 3>;
        using i3 = zs::vec<int, 3>;
        constexpr auto space = execspace_e::cuda;
        auto pol = cuda_exec();
        using bvh_t = ZenoLinearBvh::lbvh_t;
        using bv_t = typename bvh_t::Box;
        // zs::AABBBox<3, float>
        auto &sceneData = scene->userData();
        auto allocator = get_temporary_memory_source(pol);
        if (!sceneData.has<ZenoLinearBvh>(zs_bvh_tag)) {
            auto zsbvh = std::make_shared<ZenoLinearBvh>();
            zsbvh->thickness = 0;
            zsbvh->et = ZenoLinearBvh::surface;

            const auto &sceneTris = scene->tris.values;
            const auto &scenePos = scene->verts.values;
            zs::Vector<v3> pos{allocator, scenePos.size()};
            zs::Vector<i3> indices{sceneTris.size(), memsrc_e::device, 0};
            zs::copy(mem_device, (void *)pos.data(), (void *)scenePos.data(), sizeof(v3) * pos.size());
            zs::copy(mem_device, (void *)indices.data(), (void *)sceneTris.data(), sizeof(i3) * indices.size());

            auto &bvh = zsbvh->bvh;
            zs::Vector<bv_t> bvs{allocator, indices.size()};
            retrieve_bounding_volumes(pol, range(pos), range(indices), bvs);
            bvh.build(pol, bvs);
            sceneData.set(zs_bvh_tag, zsbvh);
        } else {
            const auto &sceneTris = scene->tris.values;
            const auto &scenePos = scene->verts.values;
            zs::Vector<v3> pos{allocator, scenePos.size()};
            zs::Vector<i3> indices{allocator, sceneTris.size()};
            zs::copy(mem_device, (void *)pos.data(), (void *)scenePos.data(), sizeof(v3) * pos.size());
            zs::copy(mem_device, (void *)indices.data(), (void *)sceneTris.data(), sizeof(i3) * indices.size());

            auto zsbvh = sceneData.get<ZenoLinearBvh>(zs_bvh_tag); // std::shared_ptr<>
            auto &bvh = zsbvh->bvh;
            zs::Vector<bv_t> bvs{indices.size(), memsrc_e::device, 0};
            retrieve_bounding_volumes(pol, range(pos), range(indices), bvs);
            bvh.refit(pol, bvs);
        }

        auto zsbvh = sceneData.get<ZenoLinearBvh>(zs_bvh_tag); // std::shared_ptr<>
        auto &bvh = zsbvh->bvh;

        set_output("prim", prim);
    }
};

ZENDEFNODE(ComputeVertexAO, {
                                {
                                    {"PrimitiveObject", "prim", ""},
                                    {"PrimitiveObject", "scene", ""},
                                    {"string", "nrm_tag", "nrm"},
                                    {"int", "sample_iters", "512"},
                                },
                                {"prim"},
                                {},
                                {"tracing"},
                            });

} // namespace zeno