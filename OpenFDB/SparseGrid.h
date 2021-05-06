// vim: sw=2 sts=2 ts=2
#pragma once


#include <vector>
#include "MathVec.h"


namespace fdb {


static size_t linearize(int x, int y, int z, size_t n) {
  return x + n * (y + n * z);
}


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

  LeafT *leafAt(int x, int y, int z) const {
    auto *tree = root->mChildren[linearize(x >> 4, y >> 4, z >> 4, 16)];
    if (!tree) return nullptr;
    auto *leaf = tree->mChildren[linearize(x & 15, y & 15, z & 15, 16)];
    return leaf;
  }

  LeafT *touchLeafAt(int x, int y, int z) const {
    auto *&tree = root->mChildren[linearize(x >> 4, y >> 4, z >> 4, 16)];
    if (!tree) {
      tree = new TreeNode<LeafT>;
    }
    auto *&leaf = tree->mChildren[linearize(x & 15, y & 15, z & 15, 16)];
    if (!leaf) {
      leaf = new LeafT;
    }
    return leaf;
  }

  auto iterLeaf() const {
    std::vector<LeafT *> res;
    for (size_t i = 0; i < 16 * 16 * 16; i++) {
      auto *tree = root->mChildren[i];
      if (!tree) continue;
      for (size_t i = 0; i < 16 * 16 * 16; i++) {
        auto *leaf = tree->mChildren[i];
        if (leaf)
          res.push_back(leaf);
      }
    }
    return res;
  }
};

struct PointsGrid : Grid<PointsLeaf> {
  void addPoint(Vec3u const &pos) const {
    Vec3u idx = pos >> 16;
    Vec3h off = pos & 65535;
    auto *leaf = touchLeafAt(idx.x, idx.y, idx.z);
    leaf->addPoint(off);
  }

  auto iterPoint() const {
    std::vector<Vec3u> res;
    for (auto *leaf: iterLeaf()) {
      for (auto const &pos: leaf->iterPoint()) {
        res.push_back(pos);
      }
    }
    return res;
  }
};


}
