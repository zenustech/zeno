#include <zeno/core/Graph.h>
#include <zeno/extra/assetDir.h>
#include <zeno/funcs/PrimitiveUtils.h>
#include <zeno/utils/log.h>
#include <zeno/zeno.h>

#include "zensim/container/Bht.hpp"
#include "zensim/omp/execution/ExecutionPolicy.hpp"

namespace zeno {

struct PolyReduceLite : INode {
  virtual void apply() override {
    auto prim = get_input<PrimitiveObject>("prim");

    using namespace zs;
    constexpr auto space = execspace_e::openmp;
    auto pol = omp_exec();

    auto &verts = prim->verts;
    const auto &pos = verts.values;
    std::vector<int> vertDiscard(pos.size());
    std::vector<std::vector<int>> vertTris(pos.size());

    std::vector<int> fas(pos.size());
    pol(enumerate(fas), [](int no, int &fa) { fa = no; });

    const auto &tris = prim->tris.values;
    std::vector<int> triDiscard(tris.size());

    /// extract all edges
    /// @note vtab._activeKeys are all the edges
    bht<int, 2, int> vtab{(tris.size() * 3) * 3 / 2};
    {
      vtab.reset(pol, true);
      pol(enumerate(tris),
          [&, vtab = proxy<space>(vtab)](int triNo, auto tri) mutable {
            int i = tri[2];
            for (int d = 0; d != 3; ++d) {
              int j = tri[d];
              vtab.insert(zs::vec<int, 2>{std::min(i, j), std::max(i, j)});
              i = j;
            }
          });
      if (vtab._buildSuccess.getVal() == 0)
        throw std::runtime_error("PolyReduceLite hash failed!!");
    }
    auto nEdges = vtab.size();

    {
      /// establish vert-face relations
      std::vector<std::mutex> vertMutex(pos.size());
      pol(enumerate(tris), [&](int triNo, auto tri) {
        for (auto vNo : tri) {
          std::lock_guard lk(vertMutex[vNo]);
          vertTris[vNo].push_back(triNo);
        }
      });
    }

    int nIters = get_input2<int>("iterations");

    auto triHasVert = [&tris](const auto &tri, int v) {
      for (auto vNo : tri)
        if (vNo == v)
          return true;
      return false;
    };
    for (int i = 0; i != nIters; ++i) {
      ///
      if (i == nEdges)
        break;
      /// evaluate vert curvatures
      ;
      /// sort edges for collapse
      ;
      // temporal measure
      int no = i;
      /// collapse the most prefered edge (uv -> u, u > v)
      auto edge = vtab._activeKeys[no];
      auto u = std::max(edge[0], edge[1]), v = std::min(edge[0], edge[1]);
      // 1. remove vert u
      fas[u] = v;
      vertDiscard[u] = 1;
      // 2. remove triangles containing u-v
      for (int triNo : vertTris[u]) {
        if (triDiscard[triNo])
          continue;
        if (triHasVert(tris[triNo], v)) {
          // delete this triangle
          triDiscard[triNo] = 1;
        }
      }
      // 3. remapping triangles verts u->v will be done afterwards
    }

    std::vector<int> offsets(pos.size());
    {
      auto &vertPreserve = vertDiscard;
      /// compact verts
      pol(vertPreserve, [](int &v) { v = !v; });
      exclusive_scan(pol, std::begin(vertPreserve), std::end(vertPreserve),
                     std::begin(offsets));
      auto nvs = offsets.back() + vertPreserve.back();
      fmt::print("{} verts to {} verts\n", pos.size(), nvs);
      RM_CVREF_T(prim->verts) newVerts(nvs);
      pol(range(pos.size()), [&](int i) {
        if (vertPreserve[i])
          newVerts.values[offsets[i]] = pos[i];
        // map verts
        auto fa = fas[i];
        for (auto gfa = fas[fa]; gfa != fa; gfa = fas[fa])
          fa = gfa;
        fas[i] = offsets[fa];
      });
      prim->verts = newVerts;
    }
    {
      auto &triPreserve = triDiscard;
      /// compact tris and map vert indices
      offsets.resize(tris.size());
      pol(triPreserve, [](int &v) { v = !v; });
      exclusive_scan(pol, std::begin(triPreserve), std::end(triPreserve),
                     std::begin(offsets));
      auto nts = offsets.back() + triPreserve.back();
      fmt::print("{} tris to {} tris\n", tris.size(), nts);
      RM_CVREF_T(prim->tris) newTris(nts);
      pol(range(tris.size()), [&](int i) {
        if (triPreserve[i]) {
          auto tri = tris[i];
          for (auto &v : tri)
            v = fas[v];
          newTris.values[offsets[i]] = tri;
        }
      });
      prim->tris = newTris;
    }

    set_output("prim", std::move(prim));
  }
};

ZENDEFNODE(PolyReduceLite,
           {
               {{"PrimitiveObject", "prim"}, {"int", "iterations", "100"}},
               {
                   {"PrimitiveObject", "prim"},
               },
               {},
               {"zs_geom"},
           });

} // namespace zeno