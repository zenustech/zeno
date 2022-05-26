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

struct VisualizeBvh : INode {
  void apply() override {
    auto zstets = get_input<ZenoParticles>("ZSParticles");
    auto thickness = get_input2<float>("thickness");
    auto level = get_input2<int>("level");
    auto target = get_param<std::string>("target");

    using namespace zs;
    using bv_t = typename ZenoParticles::lbvh_t::Box;
    constexpr auto space = execspace_e::cuda;

    auto cudaPol = cuda_exec().device(0);

    auto prim = std::make_shared<PrimitiveObject>();

    const auto &verts = zstets->getParticles();

    Vector<bv_t> extractedBvs{verts.get_allocator(), 1};
    Vector<int> bvCnt{verts.get_allocator(), 1};
    bvCnt.setVal(0);

    if (target == "surface") {
      const auto &surfaces = (*zstets)[ZenoParticles::s_surfTriTag];
      auto bvs = retrieve_bounding_volumes(cudaPol, verts, surfaces, wrapv<3>{},
                                           thickness, "x");
      if (!zstets->hasBvh(ZenoParticles::s_surfTriTag)) {
        auto &surfBvh = zstets->bvh(ZenoParticles::s_surfTriTag);
        surfBvh.build(cudaPol, bvs);
      } else {
        auto &surfBvh = zstets->bvh(ZenoParticles::s_surfTriTag);
        surfBvh.refit(cudaPol, bvs);
      }
      auto &surfBvh = zstets->bvh(ZenoParticles::s_surfTriTag);

      extractedBvs.resize(surfaces.size());
      cudaPol(range(surfaces.size()),
              [bvh = proxy<space>(surfBvh),
               extractedBvs = proxy<space>(extractedBvs),
               bvCnt = proxy<space>(bvCnt), expLevel = level,
               execTag = wrapv<space>{}] ZS_LAMBDA(int i) mutable {
                auto node = bvh._leafIndices[i];
                using bvh_t = RM_CVREF_T(bvh);
                using Ti = typename bvh_t::index_t;
                Ti level = 0;
                while (node != -1 && level != expLevel) {
                  if (auto par = bvh._parents[node]; node != par + 1)
                    node = par;
                  else
                    return;
                  ++level;
                }
                // is valid and the right child of its parent
                if (node != -1) {
                  auto no = atomic_add(execTag, &bvCnt[0], 1);
                  extractedBvs[no] = bvh.getNodeBV(node);
                }
              });
    } else if (target == "edge") {
      const auto &boundaryEdges = (*zstets)[ZenoParticles::s_surfEdgeTag];
      auto bvs = retrieve_bounding_volumes(cudaPol, verts, boundaryEdges,
                                           wrapv<2>{}, thickness, "x");

      if (!zstets->hasBvh(ZenoParticles::s_surfEdgeTag)) {
        auto &beBvh = zstets->bvh(ZenoParticles::s_surfEdgeTag);
        beBvh.build(cudaPol, bvs);
      } else {
        auto &beBvh = zstets->bvh(ZenoParticles::s_surfEdgeTag);
        beBvh.refit(cudaPol, bvs);
      }
      auto &beBvh = zstets->bvh(ZenoParticles::s_surfEdgeTag);

      extractedBvs.resize(boundaryEdges.size());
      cudaPol(range(boundaryEdges.size()),
              [bvh = proxy<space>(beBvh),
               extractedBvs = proxy<space>(extractedBvs),
               bvCnt = proxy<space>(bvCnt), expLevel = level,
               execTag = wrapv<space>{}] ZS_LAMBDA(int i) mutable {
                using bvh_t = RM_CVREF_T(bvh);
                using Ti = typename bvh_t::index_t;
                auto node = bvh._leafIndices[i];
                Ti level = 0;
                while (node != -1 && level != expLevel) {
                  if (auto par = bvh._parents[node]; node != par + 1)
                    node = par;
                  else
                    return;
                  ++level;
                }
                // is valid and the right child of its parent
                if (node != -1) {
                  auto no = atomic_add(execTag, &bvCnt[0], 1);
                  extractedBvs[no] = bvh.getNodeBV(node);
                }
              });
    } else if (target == "point") {
      const auto &boundaryVerts = (*zstets)[ZenoParticles::s_surfVertTag];
      auto bvs = retrieve_bounding_volumes(cudaPol, verts, boundaryVerts,
                                           wrapv<1>{}, thickness, "x");

      if (!zstets->hasBvh(ZenoParticles::s_surfVertTag)) {
        auto &bvBvh = zstets->bvh(ZenoParticles::s_surfVertTag);
        bvBvh.build(cudaPol, bvs);
      } else {
        auto &bvBvh = zstets->bvh(ZenoParticles::s_surfVertTag);
        bvBvh.refit(cudaPol, bvs);
      }
      auto &bvBvh = zstets->bvh(ZenoParticles::s_surfVertTag);

      extractedBvs.resize(boundaryVerts.size());
      cudaPol(range(boundaryVerts.size()),
              [bvh = proxy<space>(bvBvh),
               extractedBvs = proxy<space>(extractedBvs),
               bvCnt = proxy<space>(bvCnt), expLevel = level,
               execTag = wrapv<space>{}] ZS_LAMBDA(int i) mutable {
                using bvh_t = RM_CVREF_T(bvh);
                using Ti = typename bvh_t::index_t;
                auto node = bvh._leafIndices[i];
                Ti level = 0;
                while (node != -1 && level != expLevel) {
                  if (auto par = bvh._parents[node]; node != par + 1)
                    node = par;
                  else
                    return;
                  ++level;
                }
                // is valid and the right child of its parent
                if (node != -1) {
                  auto no = atomic_add(execTag, &bvCnt[0], 1);
                  extractedBvs[no] = bvh.getNodeBV(node);
                }
              });
    }
    {
      const std::size_t numExtractedBvs = bvCnt.getVal();
      fmt::print("extracted {} bvs\n", numExtractedBvs);
      extractedBvs.resize(numExtractedBvs);
      Vector<zs::vec<int, 2>> dLines{extractedBvs.get_allocator(),
                                     12 * numExtractedBvs};
      Vector<zs::vec<float, 3>> dVerts{extractedBvs.get_allocator(),
                                       8 * numExtractedBvs};
      cudaPol(range(numExtractedBvs),
              [extractedBvs = proxy<space>(extractedBvs),
               dVerts = proxy<space>(dVerts)] ZS_LAMBDA(int i) mutable {
                auto bv = extractedBvs[i];
                auto offset = i * 8;
                for (auto [x, y, z] : ndrange<3>(2))
                  dVerts[offset++] = bv.getVert(x, y, z);
#if 0
                auto mi = bv._min;
                auto len = bv._max - bv._min;
                for (auto [x, y, z] : ndrange<3>(2))
                  dVerts[offset++] =
                      zs::vec<float, 3>{x ? len[0] : 0, y ? len[1] : 0,
                                        z ? len[2] : 0} +
                      mi;
#endif
              });
      cudaPol(range(numExtractedBvs),
              [dLines = proxy<space>(dLines)] ZS_LAMBDA(int i) mutable {
                using ivec = zs::vec<int, 2>;
                auto voffset = i * 8;
                auto toffset = i * 12;
                dLines[toffset++] = ivec{0, 4} + voffset;
                dLines[toffset++] = ivec{1, 5} + voffset;
                dLines[toffset++] = ivec{2, 6} + voffset;
                dLines[toffset++] = ivec{3, 7} + voffset;
                dLines[toffset++] = ivec{0, 2} + voffset;
                dLines[toffset++] = ivec{1, 3} + voffset;

                dLines[toffset++] = ivec{4, 6} + voffset;
                dLines[toffset++] = ivec{5, 7} + voffset;
                dLines[toffset++] = ivec{0, 1} + voffset;
                dLines[toffset++] = ivec{2, 3} + voffset;
                dLines[toffset++] = ivec{4, 5} + voffset;
                dLines[toffset++] = ivec{6, 7} + voffset;
              });

#if 0
      cudaPol(range(numExtractedBvs),
              [extractedBvs = proxy<space>(extractedBvs),
               surfaces = proxy<space>({}, surfaces),
               dVerts = proxy<space>(dVerts)] ZS_LAMBDA(int i) mutable {
                auto bv = extractedBvs[i];
                auto tri = surfaces.template pack<3>("inds", i)
                               .template reinterpret_bits<int>();
                printf("%d-th bv (%d, %d, %d): [%f, %f, %f] - [%f, %f, %f]\n",
                       i, tri[0], tri[1], tri[2], bv._min[0], bv._min[1],
                       bv._min[2], bv._max[0], bv._max[1], bv._max[2]);
              });
#endif

      prim->resize(8 * numExtractedBvs);
      auto &pos = prim->verts.values;
      prim->lines.resize(12 * numExtractedBvs);
      auto &lines = prim->lines.values;

      static_assert(sizeof(zeno::vec3f) == sizeof(zs::vec<float, 3>) &&
                        sizeof(zeno::vec2i) == sizeof(zs::vec<int, 2>),
                    "vec3f size assumption broken");
      Resource::copy(zs::MemoryEntity{MemoryLocation{memsrc_e::host, -1},
                                      (void *)pos.data()},
                     zs::MemoryEntity{MemoryLocation{memsrc_e::device, 0},
                                      (void *)dVerts.data()},
                     sizeof(zeno::vec3f) * pos.size());
      Resource::copy(zs::MemoryEntity{MemoryLocation{memsrc_e::host, -1},
                                      (void *)lines.data()},
                     zs::MemoryEntity{MemoryLocation{memsrc_e::device, 0},
                                      (void *)dLines.data()},
                     sizeof(zeno::vec2i) * lines.size());
    }

    set_output("prim", prim);
  }
};

/// for discrete collision detections etc.
ZENDEFNODE(VisualizeBvh, {
                             {"ZSParticles",
                              {"float", "thickness", "0.0"},
                              {"int", "level", "0"}},
                             {"prim"},
                             {{"enum point edge surface", "target", "surface"}},
                             {"FEM"},
                         });

} // namespace zeno