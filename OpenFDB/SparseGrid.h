// vim: sw=2 sts=2 ts=2
#pragma once


#include <vector>
#include "MathVec.h"


namespace fdb {

static size_t linearize32(Vec3I const &idx) {
  return idx.x | (idx.y << 5) | (idx.z << 10);
}

static Vec3I unlinearize32(size_t off) {
  Vec3I idx;
  idx.x = off & 31;
  idx.y = (off >> 5) & 31;
  idx.z = off >> 10;
  return idx;
}

static size_t linearize16(Vec3I const &idx) {
  return idx.x | (idx.y << 4) | (idx.z << 8);
}

static Vec3I unlinearize16(size_t off) {
  Vec3I idx;
  idx.x = off & 15;
  idx.y = (off >> 4) & 15;
  idx.z = off >> 8;
  return idx;
}

template <class LeafT>
struct TreeNode {
  LeafT *mChildren[16 * 16 * 16];

  TreeNode() {
    for (int i = 0; i < 16 * 16 * 16; i++) {
      mChildren[i] = nullptr;
    }
  }

  ~TreeNode() {
    for (int i = 0; i < 16 * 16 * 16; i++) {
      if (mChildren[i])
        delete mChildren[i];
      mChildren[i] = nullptr;
    }
  }
};

template <class LeafT>
struct RootNode {
  TreeNode<LeafT> *mChildren[32 * 32 * 32];

  RootNode() {
    for (int i = 0; i < 32 * 32 * 32; i++) {
      mChildren[i] = nullptr;
    }
  }

  ~RootNode() {
    for (int i = 0; i < 32 * 32 * 32; i++) {
      if (mChildren[i])
        delete mChildren[i];
      mChildren[i] = nullptr;
    }
  }
};

template <class LeafT>
struct Grid {
  RootNode<LeafT> *root;

  Grid() {
    root = new RootNode<LeafT>;
  }

  ~Grid() {
    delete root;
    root = nullptr;
  }

  template <bool activate>
  LeafT *leafAt(Vec3H const &idx) const {
    auto *&tree = root->mChildren[linearize32(idx >> 4)];
    if (!tree) {
      if constexpr (!activate) return nullptr;
      tree = new TreeNode<LeafT>;
    }
    auto *&leaf = tree->mChildren[linearize16(idx & 15)];
    if (!leaf) {
      if constexpr (!activate) return nullptr;
      leaf = new LeafT;
    }
    return leaf;
  }

  auto iterLeaf() const {
    std::vector<std::pair<Vec3H, LeafT *>> res;
    for (int i = 0; i < 32 * 32 * 32; i++) {
      auto *tree = root->mChildren[i];
      if (!tree) continue;
      for (int j = 0; j < 16 * 16 * 16; j++) {
        auto *leaf = tree->mChildren[j];
        if (!leaf) continue;
        Vec3H tree_idx = unlinearize32(i);
        Vec3H leaf_idx = unlinearize16(j);
        Vec3H idx = (tree_idx << 4) | leaf_idx;
        res.push_back(std::make_pair(idx, leaf));
      }
    }
    return res;
  }
};


struct PointsLeaf {
  std::vector<Vec3H> mOffsets;

  void addPoint(Vec3H const &off) {
    mOffsets.push_back(off);
  }

  auto const &iterPoint() {
    return mOffsets;
  }

  PointsLeaf() = default;
  ~PointsLeaf() = default;
};

struct PointsGrid : Grid<PointsLeaf> {
  static Vec3i composeIndex(Vec3H const &idx, Vec3H const &off) {
    Vec3i pos = (((Vec3i)idx << 16) | (Vec3i)off) - (1 << 23);
    return pos;
  }

  static std::pair<Vec3H, Vec3H> decomposeIndex(Vec3i const &pos) {
    Vec3H idx = (pos + (1 << 24)) >> 16;
    Vec3H off = pos & 65535;
    return std::make_pair(idx, off);
  }

  void addPoint(Vec3i const &pos) const {
    Vec3H idx = (pos + (1 << 24)) >> 16;
    Vec3H off = pos & 65535;
    auto *leaf = leafAt<true>(idx);
    leaf->addPoint(off);
  }

  auto iterPoint() const {
    std::vector<Vec3i> res;
    for (auto [idx, leaf]: iterLeaf()) {
      for (auto const &off: leaf->iterPoint()) {
        Vec3i pos = (((Vec3i)idx << 16) | (Vec3i)off) - (1 << 24);
        res.push_back(pos);
      }
    }
    return res;
  }
};


}
