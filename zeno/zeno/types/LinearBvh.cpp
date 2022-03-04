#include "LinearBvh.h"
#include <algorithm>
#include <atomic>
#include <exception>
#include <stdexcept>
#include <zeno/zeno.h>
// #if defined(_OPENMP)
#include <omp.h>
// #endif

namespace zeno {

void LBvh::build(const std::shared_ptr<PrimitiveObject> &prim,
                 float thickness) {
  this->primPtr = prim;
  this->thickness = thickness;

  Ti numLeaves = 0; // refpos.size();

  {
    // determine element category
    if (prim->quads.size() > 0) {
      this->eleCategory = element_e::tet;
      numLeaves = prim->quads.size();
    } else if (prim->tris.size() > 0) {
      this->eleCategory = element_e::tri;
      numLeaves = prim->tris.size();
    } else if (prim->lines.size() > 0) {
      this->eleCategory = element_e::line;
      numLeaves = prim->lines.size();
    } else if (prim->points.size() > 0) {
      this->eleCategory = element_e::point;
      numLeaves = prim->points.size();
    } else {
      this->eleCategory = element_e::point;
      numLeaves = prim->verts.size();
      prim->points.resize(numLeaves);
#pragma omp parallel for
      for (Ti i = 0; i < numLeaves; ++i)
        prim->points[i] = i;
    }
  }

  const auto &refpos = prim->attr<vec3f>("pos");
  const Ti numNodes = numLeaves + numLeaves - 1;
  sortedBvs.resize(numNodes);
  auxIndices.resize(numNodes);
  levels.resize(numNodes);
  parents.resize(numNodes);
  leafIndices.resize(numLeaves);

  constexpr int dim = 3;
  constexpr auto ma = std::numeric_limits<float>().max();
  constexpr auto mi = std::numeric_limits<float>().lowest();
  Box wholeBox{TV{ma, ma, ma}, TV{mi, mi, mi}};
  const auto defaultBox = wholeBox;

  /// whole box
  // should use reduce here
  for (Ti i = 0; i != numLeaves; ++i) {
    const auto &p = refpos[i];
    for (int d = 0; d != dim; ++d) {
      if (p[d] < wholeBox.first[d])
        wholeBox.first[d] = p[d];
      if (p[d] > wholeBox.second[d])
        wholeBox.second[d] = p[d];
    }
  }
  // printf("lbvh bounding box: %f, %f, %f - %f, %f, %f\n", wholeBox.first[0],
  // wholeBox.first[1], wholeBox.first[2], wholeBox.second[0],
  // wholeBox.second[1], wholeBox.second[2]);

  std::vector<std::pair<Tu, Ti>> records(numLeaves); // <mc, id>
  /// morton codes
  auto getMortonCode = [](const TV &p) -> Tu {
    auto expand_bits = [](Tu v) -> Tu { // expands lower 10-bits to 30 bits
      v = (v * 0x00010001u) & 0xFF0000FFu;
      v = (v * 0x00000101u) & 0x0F00F00Fu;
      v = (v * 0x00000011u) & 0xC30C30C3u;
      v = (v * 0x00000005u) & 0x49249249u;
      return v;
    };
    return (expand_bits((Tu)(p[0] * 1024.f)) << (Tu)2) |
           (expand_bits((Tu)(p[1] * 1024.f)) << (Tu)1) |
           expand_bits((Tu)(p[2] * 1024.f));
  };
  std::function<Box(Ti)> getBv;
  {
    const auto lengths = wholeBox.second - wholeBox.first;
    auto getUniformCoord = [&wholeBox, &lengths](const TV &p) {
      // https://newbedev.com/constexpr-variable-captured-inside-lambda-loses-its-constexpr-ness
      constexpr int dim = 3;
      auto offsets = p - wholeBox.first;
      for (int d = 0; d != dim; ++d)
        offsets[d] = std::clamp(offsets[d], (float)0, lengths[d]) / lengths[d];
      return offsets;
    };
    if (eleCategory == element_e::tet) {
      getBv = [&quads = prim->quads, &refpos, &defaultBox,
               thickness](Ti i) -> Box {
        auto quad = quads[i];
        Box bv = defaultBox;
        for (int j = 0; j != 4; ++j) {
          const auto &p = refpos[quad[j]];
          for (int d = 0; d != 3; ++d) {
            if (p[d] - thickness < bv.first[d])
              bv.first[d] = p[d] - thickness;
            if (p[d] + thickness > bv.second[d])
              bv.second[d] = p[d] + thickness;
          }
        }
        return bv;
      };
#pragma omp parallel for
      for (Ti i = 0; i < numLeaves; ++i) {
        auto quad = prim->quads[i];
        auto uc = getUniformCoord((refpos[quad[0]] + refpos[quad[1]] +
                                   refpos[quad[2]] + refpos[quad[3]]) /
                                  4);
        records[i] = std::make_pair(getMortonCode(uc), i);
      }
    } else if (eleCategory == element_e::tri) {
      getBv = [&tris = prim->tris, &refpos, &defaultBox,
               thickness](Ti i) -> Box {
        auto tri = tris[i];
        Box bv = defaultBox;
        for (int j = 0; j != 3; ++j) {
          const auto &p = refpos[tri[j]];
          for (int d = 0; d != 3; ++d) {
            if (p[d] - thickness < bv.first[d])
              bv.first[d] = p[d] - thickness;
            if (p[d] + thickness > bv.second[d])
              bv.second[d] = p[d] + thickness;
          }
        }
        return bv;
      };
#pragma omp parallel for
      for (Ti i = 0; i < numLeaves; ++i) {
        auto tri = prim->tris[i];
        auto uc = getUniformCoord(
            (refpos[tri[0]] + refpos[tri[1]] + refpos[tri[2]]) / 3);
        records[i] = std::make_pair(getMortonCode(uc), i);
      }
    } else if (eleCategory == element_e::line) {
      getBv = [&lines = prim->lines, &refpos, &defaultBox,
               thickness](Ti i) -> Box {
        auto line = lines[i];
        Box bv = defaultBox;
        for (int j = 0; j != 2; ++j) {
          const auto &p = refpos[line[j]];
          for (int d = 0; d != 3; ++d) {
            if (p[d] - thickness < bv.first[d])
              bv.first[d] = p[d] - thickness;
            if (p[d] + thickness > bv.second[d])
              bv.second[d] = p[d] + thickness;
          }
        }
        return bv;
      };
#pragma omp parallel for
      for (Ti i = 0; i < numLeaves; ++i) {
        auto line = prim->lines[i];
        auto uc = getUniformCoord((refpos[line[0]] + refpos[line[1]]) / 2);
        records[i] = std::make_pair(getMortonCode(uc), i);
      }
    } else if (eleCategory == element_e::point) {
      getBv = [&points = prim->points, &refpos, &defaultBox,
               thickness](Ti i) -> Box {
        auto point = points[i];
        Box bv = defaultBox;
        const auto &p = refpos[point];
        for (int d = 0; d != 3; ++d) {
          if (p[d] - thickness < bv.first[d])
            bv.first[d] = p[d] - thickness;
          if (p[d] + thickness > bv.second[d])
            bv.second[d] = p[d] + thickness;
        }
        return bv;
      };
#pragma omp parallel for
      for (Ti i = 0; i < numLeaves; ++i) {
        auto pi = prim->points[i];
        auto uc = getUniformCoord(refpos[pi]);
        records[i] = std::make_pair(getMortonCode(uc), i);
      }
    }
  }
  std::sort(std::begin(records), std::end(records));

  std::vector<Tu> splits(numLeaves);
  ///
  constexpr auto numTotalBits = sizeof(Tu) * 8;
  auto clz = [](Tu x) -> Tu {
    static_assert(std::is_same_v<Tu, unsigned int>,
                  "Tu should be unsigned int");
#if defined(_MSC_VER) || (defined(_WIN32) && defined(__INTEL_COMPILER))
    return __lzcnt((unsigned int)x);
#elif defined(__clang__) || defined(__GNUC__)
    return __builtin_clz((unsigned int)x);
#endif
  };
#pragma omp parallel for
  for (Ti i = 0; i < numLeaves; ++i) {
    if (i != numLeaves - 1)
      splits[i] = numTotalBits - clz(records[i].first ^ records[i + 1].first);
    else
      splits[i] = numTotalBits + 1;
  }
  ///
  std::vector<Box> leafBvs(numLeaves);
  std::vector<Box> trunkBvs(numLeaves - 1);
  std::vector<Ti> leafLca(numLeaves);
  std::vector<Ti> leafDepths(numLeaves);
  std::vector<Ti> trunkR(numLeaves - 1);
  std::vector<Ti> trunkLc(numLeaves - 1);

  std::vector<std::atomic<Tu>> trunkTopoMarks(numLeaves - 1);
  std::vector<std::atomic<Ti>> trunkBuildFlags(numLeaves - 1);
#pragma omp parallel for
  for (Ti i = 0; i < numLeaves - 1; ++i) {
    trunkTopoMarks[i] = 0;
    trunkBuildFlags[i] = 0;
  }

  {
    std::vector<Ti> trunkL(numLeaves - 1);
    std::vector<Ti> trunkRc(numLeaves - 1);
#pragma omp parallel for
    for (Ti idx = 0; idx < numLeaves; ++idx) {
      {
        // const auto &pos = refpos[records[idx].second];
        // leafBvs[idx] = Box{pos - thickness, pos + thickness};
        leafBvs[idx] = getBv(records[idx].second);
      }

      leafLca[idx] = -1, leafDepths[idx] = 1;
      Ti l = idx - 1, r = idx; ///< (l, r]
      bool mark{false};

      if (l >= 0)
        mark =
            splits[l] < splits[r]; ///< true when right child, false otherwise

      int cur = mark ? l : r;
      if (mark)
        trunkRc[cur] = idx, trunkR[cur] = idx,
        trunkTopoMarks[cur].fetch_or((Tu)0x00000002u);
      else
        trunkLc[cur] = idx, trunkL[cur] = idx,
        trunkTopoMarks[cur].fetch_or((Tu)0x00000001u);

      while (trunkBuildFlags[cur].fetch_add(1) == 1) {
        { // refit
          int lc = trunkLc[cur], rc = trunkRc[cur];
          const auto childMask = trunkTopoMarks[cur] & (Tu)3;
          const auto &leftBox = (childMask & 1) ? leafBvs[lc] : trunkBvs[lc];
          const auto &rightBox = (childMask & 2) ? leafBvs[rc] : trunkBvs[rc];
          Box bv{};
          for (int d = 0; d != dim; ++d) {
            bv.first[d] = leftBox.first[d] < rightBox.first[d]
                              ? leftBox.first[d]
                              : rightBox.first[d];
            bv.second[d] = leftBox.second[d] > rightBox.second[d]
                               ? leftBox.second[d]
                               : rightBox.second[d];
          }
          trunkBvs[cur] = bv;
        }
        trunkTopoMarks[cur] &= 0x00000007;

        l = trunkL[cur] - 1, r = trunkR[cur];
        leafLca[l + 1] = cur, leafDepths[l + 1]++;
        atomic_thread_fence(std::memory_order_acquire);

        if (l >= 0)
          mark =
              splits[l] < splits[r]; ///< true when right child, false otherwise
        else
          mark = false;

        if (l + 1 == 0 && r == numLeaves - 1) {
          // trunkPar(cur) = -1;
          trunkTopoMarks[cur] &= 0xFFFFFFFB;
          break;
        }

        int par = mark ? l : r;
        // trunkPar(cur) = par;
        if (mark) {
          trunkRc[par] = cur, trunkR[par] = r;
          trunkTopoMarks[par].fetch_and(0xFFFFFFFD);
          trunkTopoMarks[cur] |= 0x00000004;
        } else {
          trunkLc[par] = cur, trunkL[par] = l + 1;
          trunkTopoMarks[par].fetch_and(0xFFFFFFFE);
          trunkTopoMarks[cur] &= 0xFFFFFFFB;
        }
        cur = par;
      }
    }
  }

  std::vector<Ti> leafOffsets(numLeaves + 1);
  leafOffsets[0] = 0;
  for (Ti i = 1; i <= numLeaves; ++i)
    leafOffsets[i] = leafOffsets[i - 1] + leafDepths[i - 1];
  std::vector<Ti> trunkDst(numLeaves - 1);
  /// compute trunk order
  // [levels], [parents], [trunkDst]
#pragma omp parallel for
  for (Ti i = 0; i < numLeaves; ++i) {
    auto offset = leafOffsets[i];
    parents[offset] = -1;
    for (Ti node = leafLca[i], level = leafDepths[i]; --level;
         node = trunkLc[node]) {
      levels[offset] = level;
      parents[offset + 1] = offset;
      trunkDst[node] = offset++;
    }
  }
  // only left-branch-node's parents are set so far
  // levels store the number of node within the left-child-branch from bottom
  // up starting from 0

  /// reorder trunk
  // [sortedBvs], [auxIndices], [parents]
  // auxIndices here is escapeIndex (for trunk nodes)
#pragma omp parallel for
  for (Ti i = 0; i < numLeaves - 1; ++i) {
    const auto dst = trunkDst[i];
    const auto &bv = trunkBvs[i];
    // auto l = trunkL[i];
    auto r = trunkR[i];
    sortedBvs[dst] = bv;
    const auto rb = r + 1;
    if (rb < numLeaves) {
      auto lca = leafLca[rb]; // rb must be in left-branch
      auto brother = (lca != -1 ? trunkDst[lca] : leafOffsets[rb]);
      auxIndices[dst] = brother;
      if (parents[dst] == dst - 1)
        parents[brother] = dst - 1; // setup right-branch brother's parent
    } else
      auxIndices[dst] = -1;
  }

  /// reorder leaf
  // [sortedBvs], [auxIndices], [levels], [parents], [leafIndices]
  // auxIndices here is primitiveIndex (for leaf nodes)
#pragma omp parallel for
  for (Ti i = 0; i < numLeaves; ++i) {
    const auto &bv = leafBvs[i];
    // const auto leafDepth = leafDepths[i];

    auto dst = leafOffsets[i + 1] - 1;
    leafIndices[i] = dst;
    sortedBvs[dst] = bv;
    auxIndices[dst] = records[i].second;
    levels[dst] = 0;
    if (parents[dst] == dst - 1)
      parents[dst + 1] = dst - 1; // setup right-branch brother's parent
    // if (leafDepth > 1) parents[dst + 1] = dst - 1;  // setup right-branch
    // brother's parent
  }
}

/// closest bounding box
void LBvh::find_nearest(TV const &pos, Ti &id, float &dist) const {
  std::shared_ptr<const PrimitiveObject> prim = primPtr.lock();
  if (!prim)
    throw std::runtime_error(
        "the primitive object referenced by lbvh not available anymore");
  const auto &refpos = prim->attr<vec3f>("pos");

  const Ti numNodes = sortedBvs.size();
  Ti node = 0;
  while (node != -1 && node != numNodes) {
    Ti level = levels[node];
    // level and node are always in sync
    for (; level; --level, ++node)
      if (auto d = distance(sortedBvs[node], pos); d > dist)
        break;
    // leaf node check
    if (level == 0) {
      const auto pid = auxIndices[node];
      auto d = length(refpos[pid] - pos);
      if (d < dist) {
        id = pid;
        dist = d;
      }
      node++;
    } else // separate at internal nodes
      node = auxIndices[node];
  }
}

} // namespace zeno