#include "../Structures.hpp"
#include "../Utils.hpp"
#include "zensim/Logger.hpp"
#include "zensim/cuda/execution/ExecutionPolicy.cuh"
#include "zensim/types/Property.h"
#include <zeno/types/ListObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/StringObject.h>

namespace zeno {

struct MaintainSpatialAccelerator : INode {
  void apply() override {
    auto zstets = get_input<ZenoParticles>("ZSParticles");
    auto thickness = get_input2<float>("thickness");

    using namespace zs;
    using bv_t = typename ZenoParticles::lbvh_t::Box;
    constexpr auto space = execspace_e::cuda;

    auto cudaPol = cuda_exec().device(0);

    const auto &verts = zstets->getParticles();
    if (zstets->isMeshPrimitive()) {
      const auto numDgr = zstets->numDegree();
      if (numDgr == 3 && zstets->category == ZenoParticles::surface) {
        const auto &elements = zstets->getQuadraturePoints();
        auto bvs = retrieve_bounding_volumes(cudaPol, verts, elements,
                                             wrapv<3>{}, thickness);
        if (zstets->hasBvh(ZenoParticles::s_elementTag)) {
          auto &bvh = zstets->bvh(ZenoParticles::s_elementTag);
          bvh.build(cudaPol, bvs);
        } else {
          auto &bvh = zstets->bvh(ZenoParticles::s_elementTag);
          bvh.refit(cudaPol, bvs);
        }
      }
    }
    if (zstets->hasAuxData(ZenoParticles::s_surfTriTag)) {
      const auto &surfaces = (*zstets)[ZenoParticles::s_surfTriTag];
      auto bvs = retrieve_bounding_volumes(cudaPol, verts, surfaces, wrapv<3>{},
                                           thickness);
      if (zstets->hasBvh(ZenoParticles::s_surfTriTag)) {
        auto &surfBvh = zstets->bvh(ZenoParticles::s_surfTriTag);
        surfBvh.build(cudaPol, bvs);
      } else {
        auto &surfBvh = zstets->bvh(ZenoParticles::s_surfTriTag);
        surfBvh.refit(cudaPol, bvs);
      }
    }
    if (zstets->hasAuxData(ZenoParticles::s_surfEdgeTag)) {
      const auto &boundaryEdges = (*zstets)[ZenoParticles::s_surfEdgeTag];
      auto bvs = retrieve_bounding_volumes(cudaPol, verts, boundaryEdges,
                                           wrapv<2>{}, thickness);

      if (zstets->hasBvh(ZenoParticles::s_surfEdgeTag)) {
        auto &beBvh = zstets->bvh(ZenoParticles::s_surfEdgeTag);
        beBvh.build(cudaPol, bvs);
      } else {
        auto &beBvh = zstets->bvh(ZenoParticles::s_surfEdgeTag);
        beBvh.refit(cudaPol, bvs);
      }
    }
    if (zstets->hasAuxData(ZenoParticles::s_surfVertTag)) {
      const auto &boundaryVerts = (*zstets)[ZenoParticles::s_surfVertTag];
      auto bvs = retrieve_bounding_volumes(cudaPol, verts, boundaryVerts,
                                           wrapv<1>{}, thickness);

      if (zstets->hasBvh(ZenoParticles::s_surfVertTag)) {
        auto &bvBvh = zstets->bvh(ZenoParticles::s_surfVertTag);
        bvBvh.build(cudaPol, bvs);
      } else {
        auto &bvBvh = zstets->bvh(ZenoParticles::s_surfVertTag);
        bvBvh.refit(cudaPol, bvs);
      }
    }
#if 0
    {
      auto &bvBvh = zstets->bvh(ZenoParticles::s_surfVertTag);
      const auto &boundaryVerts = (*zstets)[ZenoParticles::s_surfVertTag];
      cudaPol(range(1), [verts = proxy<space>({}, verts),
                         be = proxy<space>({}, boundaryVerts),
                         bvh = proxy<space>(bvBvh)] __device__(int) mutable {
        zs::vec<float, 3> p{0., 0.5f, 0.};
        bvh.iter_neighbors(p, [&](int svI) {
          auto vi = reinterpret_bits<int>(be("inds", svI));
          auto x = verts.pack<3>("x", vi);
          printf("(0, 0.5, 0) finds %d-th boundary vert [%d] (%f, %f, %f)\n",
                 svI, vi, x[0], x[1], x[2]);
        });
      });
    }
#endif

    set_output("ZSParticles", zstets);
  }
};

/// for discrete collision detections etc.
ZENDEFNODE(MaintainSpatialAccelerator,
           {
               {"ZSParticles", {"float", "thickness", "0.0"}},
               {"ZSParticles"},
               {},
               {"FEM"},
           });

} // namespace zeno