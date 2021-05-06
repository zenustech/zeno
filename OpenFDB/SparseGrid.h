// vim: sw=2 sts=2 ts=2
#pragma once


#include <vector>
#include "MathVec.h"


namespace fdb {


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
  TreeNode<LeafT> *mChildren[16 * 16 * 16];

  RootNode() {
    for (int i = 0; i < 16 * 16 * 16; i++) {
      mChildren[i] = nullptr;
    }
  }

  ~RootNode() {
    for (int i = 0; i < 16 * 16 * 16; i++) {
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
  LeafT *leafAt(Vec3I const &index) const {
    auto *&tree = root->mChildren[linearize16(index >> 4)];
    if (!tree) {
      if constexpr (!activate) return nullptr;
      tree = new TreeNode<LeafT>;
    }
    auto *&leaf = tree->mChildren[linearize16(index & 15)];
    if (!leaf) {
      if constexpr (!activate) return nullptr;
      leaf = new LeafT;
    }
    return leaf;
  }

  auto iterLeaf() const {
    std::vector<std::pair<Vec3I, LeafT *>> res;
    for (int i = 0; i < 16 * 16 * 16; i++) {
      auto *tree = root->mChildren[i];
      if (!tree) continue;
      for (int j = 0; j < 16 * 16 * 16; j++) {
        auto *leaf = tree->mChildren[j];
        if (!leaf) continue;
        Vec3I tree_idx = unlinearize16(i);
        Vec3I leaf_idx = unlinearize16(j);
        Vec3I idx = (tree_idx << 4) | leaf_idx;
        res.push_back(std::make_pair(idx, leaf));
      }
    }
    return res;
  }
};


struct PointsLeaf {
  std::vector<Vec3h> mOffsets;

  void addPoint(Vec3h const &off) {
    mOffsets.push_back(off);
  }

  auto const &iterPoint() {
    return mOffsets;
  }

  PointsLeaf() = default;
  ~PointsLeaf() = default;
};

struct PointsGrid : Grid<PointsLeaf> {
  void addPoint(Vec3I const &pos) const {
    Vec3I idx = pos >> 16;
    Vec3h off = pos & 65535;
    auto *leaf = leafAt<true>(idx);
    leaf->addPoint(off);
  }

  auto iterPoint() const {
    std::vector<Vec3I> res;
    for (auto [idx, leaf]: iterLeaf()) {
      for (auto const &off: leaf->iterPoint()) {
        Vec3I pos = (idx << 16) | off;
        res.push_back(pos);
      }
    }
    return res;
  }

  static constexpr unsigned int MAX_INDEX = 1 << 24;
};


}
