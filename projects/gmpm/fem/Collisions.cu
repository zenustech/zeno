#include "../Structures.hpp"
#include "zensim/Logger.hpp"
#include "zensim/cuda/execution/ExecutionPolicy.cuh"
#include "zensim/types/Property.h"
#include <zeno/types/ListObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/StringObject.h>

namespace zeno {

struct BuildSpatialAccelerator : INode {
  void apply() override {
    auto zstets = get_input<ZenoParticles>("ZSParticles");
    auto &verts = zstets->getParticles();
    auto thickness = get_input2<float>("thickness");

    using namespace zs;
    constexpr auto space = execspace_e::cuda;

    auto cudaPol = cuda_exec().device(0);

    if (zstets->hasAuxData("surfaces")) {
      const auto &surfaces = (*zstets)["surfaces"];
      const auto numSurfs = surfaces.size();
      auto &surfBvh = zstets->bvh("surfaces");

      using bv_t = typename ZenoParticles::lbvh_t::Box;
      zs::Vector<bv_t> bvs{surfaces.get_allocator(), surfaces.size()};
#if 0
      cudaPol(range(numSurfs),
              [eles = proxy<space>({}, springs), bvs = proxy<space>(bvs),
               data = proxy<space>({}, vertData), vtag, eps, dt,
               thickness] __device__(int ei) mutable {
                auto inds = eles.pack<2>("inds", ei).reinterpret_bits<int>();
                auto dv0 = data.pack<3>(vtag, inds[0]) * eps;
                auto x0 = data.pack<3>("x0", inds[0]);
                auto dv1 = data.pack<3>(vtag, inds[1]) * eps;
                auto x1 = data.pack<3>("x0", inds[1]);
                x0 += dv0 * dt;
                x1 += dv1 * dt;
                auto [mi, ma] = get_bounding_box(x0, x1);
                bv_t bv{mi, ma};
                // thicken the box
                bv._min -= thickness * 2;
                bv._max += thickness * 2;
                bvs[ei] = bv;
              });
      surfBvh.build(cudaPol, bvs);
#endif
    }

    set_output("ZSParticles", zstets);
  }
};

ZENDEFNODE(BuildSpatialAccelerator,
           {
               {"ZSParticles", {"float", "thickness", "0.0"}},
               {"ZSParticles"},
               {},
               {"FEM"},
           });

} // namespace zeno