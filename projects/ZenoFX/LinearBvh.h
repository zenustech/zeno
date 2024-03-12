#pragma once

#include <zeno/core/INode.h>
#include <zeno/core/IObject.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/utils/vec.h>
#include <zeno/zeno.h>
#include <exception>
#include <stdexcept>
#include "SpatialUtils.hpp"
#if defined(_OPENMP)
#include <omp.h>
#endif

namespace zeno {

struct LBvh : IObjectClone<LBvh> {
  enum element_e { point = 0, line, tri, tet, unknown };
  template <element_e et>
  using element_t = std::integral_constant<element_e, et>;
  template <element_e et> static constexpr element_t<et> element_c{};

  using TV = vec3f;
  using Box = std::pair<TV, TV>;
  using Ti = int;
  using Tu = std::make_unsigned_t<Ti>;
  using BvFunc = std::function<Box(Ti)>;

  std::weak_ptr<const PrimitiveObject> primPtr;
  BvFunc getBv;
  std::vector<Box> sortedBvs;
  std::vector<Ti> auxIndices, levels, parents, leafIndices;
  float thickness{0};
  std::string radiusAttr{""};
  element_e eleCategory{element_e::point}; // element category

  LBvh() noexcept = default;
  LBvh(const std::shared_ptr<PrimitiveObject> &prim, float thickness = 0.f) {
    build(prim, thickness, radiusAttr);
  }
  template <element_e et>
  LBvh(const std::shared_ptr<PrimitiveObject> &prim, float thickness,
       element_t<et> t) {
    build(prim, thickness, radiusAttr, t);
  }
  template <element_e et>
  LBvh(const std::shared_ptr<PrimitiveObject> &prim, float thickness, std::string radiusAttr,
       element_t<et> t) {
    build(prim, thickness, radiusAttr, t);
  }

  std::size_t getNumLeaves() const noexcept { return leafIndices.size(); }
  std::size_t getNumNodes() const noexcept { return getNumLeaves() * 2 - 1; }
  BvFunc getBvFunc(const std::shared_ptr<PrimitiveObject> &prim) const;

  template <element_e et>
  void build(const std::shared_ptr<PrimitiveObject> &prim, float thickness, std::string radiusAttr, element_t<et>);

  void build(const std::shared_ptr<PrimitiveObject> &prim, float thickness, std::string radiusAttr);


  void refit();

  static bool intersect(const Box &box, const TV &p) noexcept {
    constexpr int dim = 3;
    for (Ti d = 0; d != dim; ++d)
      if (p[d] < box.first[d] || p[d] > box.second[d])
        return false;
    return true;
  }

  static bool intersect_radius(const Box &box, const TV &p, const float &radius) noexcept {
  constexpr int dim = 3;
  for (Ti d = 0; d != dim; ++d) {
    if (p[d] < (box.first[d] - radius) || p[d] > (box.second[d] + radius)) {
      return false;
    }
  }
  return true;
}

  static float distance(const Box &bv, const TV &x) {
    const auto &[mi, ma] = bv;
    TV center = (mi + ma) / 2;
    TV point = abs(x - center) - (ma - mi) / 2;
    float max = std::numeric_limits<float>::lowest();
    for (int d = 0; d != 3; ++d) {
      if (point[d] > max)
        max = point[d];
      if (point[d] < 0)
        point[d] = 0;
    }
    return (max < 0.f ? max : 0.f) + length(point);
  }
  static float distance(const TV &x, const Box &bv) { return distance(bv, x); }


  /// closest bounding box
  template <element_e et>
  TV find_nearest(TV const &pos, Ti &id, float &dist, element_t<et>) const;
  TV find_nearest(TV const &pos, Ti &id, float &dist) const;

  template <element_e et>
  TV find_nearest_with_uv(TV const &pos, TV const &uv, Ti &id, float &dist, float &uvDist, float distEps, element_t<et>) const;
  TV find_nearest_with_uv(TV const &pos, TV const &uv, Ti &id, float &dist, float &uvDist, float distEps = std::numeric_limits<float>::epsilon() * 4) const;

  template <typename SameGroupPred, element_e et = element_e::tri>
  TV find_nearest_within_group(TV const &pos, Ti &id, float &dist, SameGroupPred &&pred, 
                        element_t<et> = {}) const {
  std::shared_ptr<const PrimitiveObject> prim = primPtr.lock();
  if (!prim)
    throw std::runtime_error(
        "the primitive object referenced by lbvh not available anymore");
  const auto &refpos = prim->attr<vec3f>("pos");

  const Ti numNodes = sortedBvs.size();
  Ti node = 0;
  TV ws{0.f, 0.f, 0.f};
  TV wsTmp{0.f, 0.f, 0.f};
  while (node != -1 && node != numNodes) {
    Ti level = levels[node];
    // level and node are always in sync
    for (; level; --level, ++node)
      if (auto d = distance(sortedBvs[node], pos); d > dist)
        break;
    // leaf node check
    if (level == 0) {
      const auto eid = auxIndices[node];
      float d = std::numeric_limits<float>::max();
      if constexpr (et == element_e::point) {
        auto pt = prim->points[eid];
        if (pred(pt))
          d = dist_pp(refpos[pt], pos, wsTmp);
      } else if constexpr (et == element_e::line) {
        auto line = prim->lines[eid];
        if (pred(line[0]) && pred(line[1]))
          d = dist_pe(pos, refpos[line[0]], refpos[line[1]], wsTmp);
      } else if constexpr (et == element_e::tri) {
        auto tri = prim->tris[eid];
        if (pred(tri[0]) && pred(tri[1]) && pred(tri[2]))
          d = dist_pt(pos, refpos[tri[0]], refpos[tri[1]], refpos[tri[2]], wsTmp);
      } else if constexpr (et == element_e::tet) {
        auto tet = prim->quads[eid];
        if (pred(tet[0]) && pred(tet[1]) && pred(tet[2]) && pred(tet[3])) {
          if (auto dd =
                  dist_pt(pos, refpos[tet[0]], refpos[tet[1]], refpos[tet[2]], wsTmp);
              dd < d)
            d = dd;
          if (auto dd =
                  dist_pt(pos, refpos[tet[1]], refpos[tet[3]], refpos[tet[2]], wsTmp);
              dd < d)
            d = dd;
          if (auto dd =
                  dist_pt(pos, refpos[tet[0]], refpos[tet[3]], refpos[tet[2]], wsTmp);
              dd < d)
            d = dd;
          if (auto dd =
                  dist_pt(pos, refpos[tet[0]], refpos[tet[2]], refpos[tet[3]], wsTmp);
              dd < d)
            d = dd;
        }
      }
      if (d < dist) {
        id = eid;
        dist = d;
        ws = wsTmp;
      }
      node++;
    } else // separate at internal nodes
      node = auxIndices[node];
  }
  return ws;
}

  std::shared_ptr<PrimitiveObject> retrievePrimitive(Ti eid) const;
  vec3f retrievePrimitiveCenter(Ti eid, const TV &w) const;

  template <class F> void iter_neighbors(TV const &pos, F &&f) const {
    if (auto numLeaves = getNumLeaves(); numLeaves <= 2) {
      for (Ti i = 0; i != numLeaves; ++i) {
        if (intersect(sortedBvs[i], pos))
          f(auxIndices[i]);
      }
      return;
    }
    const Ti numNodes = sortedBvs.size();
    Ti node = 0;
    while (node != -1 && node != numNodes) {
      Ti level = levels[node];
      // level and node are always in sync
      for (; level; --level, ++node)
        if (!intersect(sortedBvs[node], pos))
          break;
      // leaf node check
      if (level == 0) {
        if (intersect(sortedBvs[node], pos))
          f(auxIndices[node]);
        node++;
      } else // separate at internal nodes
        node = auxIndices[node];
    }
  }

   template <class F> void iter_neighbors_radius(TV const &pos, const float &radius, F &&f) const {
    if (auto numLeaves = getNumLeaves(); numLeaves <= 2) {
      for (Ti i = 0; i != numLeaves; ++i) {
        if (intersect_radius(sortedBvs[i], pos, radius))
          f(auxIndices[i]);
      }
      return;
    }
    const Ti numNodes = sortedBvs.size();
    Ti node = 0;
    while (node != -1 && node != numNodes) {
      Ti level = levels[node];
      for (; level; --level, ++node)
        if (!intersect_radius(sortedBvs[node], pos, radius))
          break;
      if (level == 0) {
        if (intersect_radius(sortedBvs[node], pos, radius))
          f(auxIndices[node]);
        node++;
      } else
        node = auxIndices[node];
    }
  }


};

} // namespace zeno
