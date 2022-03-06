#pragma once

#include <zeno/core/INode.h>
#include <zeno/core/IObject.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/utils/vec.h>

namespace zeno {

struct LBvh : IObjectClone<LBvh> {
  enum element_e { point = 0, line, tri, tet };
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
  element_e eleCategory{element_e::point}; // element category

  LBvh() noexcept = default;
  LBvh(const std::shared_ptr<PrimitiveObject> &prim, float thickness = 0.f) {
    build(prim, thickness);
  }

  std::size_t getNumLeaves() const noexcept { return leafIndices.size(); }
  std::size_t getNumNodes() const noexcept { return getNumLeaves() * 2 - 1; }
  BvFunc getBvFunc(const std::shared_ptr<PrimitiveObject> &prim) const;

  void build(const std::shared_ptr<PrimitiveObject> &prim, float thickness);
  template <element_e et>
  void build(const std::shared_ptr<PrimitiveObject> &prim, float thickness,
             element_t<et>);
  void refit();

  static bool intersect(const Box &box, const TV &p) noexcept {
    constexpr int dim = 3;
    for (Ti d = 0; d != dim; ++d)
      if (p[d] < box.first[d] || p[d] > box.second[d])
        return false;
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
  void find_nearest(TV const &pos, Ti &id, float &dist) const;

  template <class F> void iter_neighbors(TV const &pos, F const &f) const {
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
};

} // namespace zeno
