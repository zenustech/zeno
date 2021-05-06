// vim: sw=2 sts=2 ts=2
#pragma once


#include <cstddef>


namespace fdb::volume {


static size_t linearize(int x, int y, int z, size_t n) {
  return x + n * (y + n * z);
}


template <class T>
struct LeafNode {
  T mChildren[8 * 8 * 8];

  T *valueAt(int x, int y, int z) {
    return &mChildren[linearize(x, y, z, 8)];
  }

  LeafNode() = default;
  ~LeafNode() = default;
};


template <class T>
struct TreeNode {
  LeafNode<T> *mChildren[16 * 16 * 16];

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


template <class T>
struct RootNode {
  TreeNode<T> *mChildren[16 * 16 * 16];

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


template <class T>
struct Grid {
  RootNode<T> *root;

  Grid() {
    root = new RootNode<T>;
  }

  ~Grid() {
    delete root;
    root = nullptr;
  }

  LeafNode<T> *leafNodeAt(int x, int y, int z) const {
    auto *tree = root->mChildren[linearize(x >> 4, y >> 4, z >> 4, 16)];
    if (!tree) return nullptr;
    auto *leaf = tree->mChildren[linearize(x & 15, y & 15, z & 15, 16)];
    return leaf;
  }

  LeafNode<T> *touchLeafNodeAt(int x, int y, int z) const {
    auto *&tree = root->mChildren[linearize(x >> 4, y >> 4, z >> 4, 16)];
    if (!tree) {
      tree = new TreeNode<T>;
    }
    auto *&leaf = tree->mChildren[linearize(x & 15, y & 15, z & 15, 16)];
    if (!leaf) {
      leaf = new LeafNode<T>;
    }
    return leaf;
  }

  T *valueAt(int x, int y, int z) const {
    auto *leaf = leafNodeAt(x >> 3, y >> 3, z >> 3);
    if (!leaf) return nullptr;
    auto *value = leaf->valueAt(x & 7, y & 7, z & 7);
    return value;
  }

  T *touchValueAt(int x, int y, int z) const {
    auto *leaf = touchLeafNodeAt(x >> 3, y >> 3, z >> 3);
    auto *value = leaf->valueAt(x & 7, y & 7, z & 7);
    return value;
  }

  bool isActiveAt(int x, int y, int z) const {
    return valueAt(x, y, z) != nullptr;
  }

  T &operator()(int x, int y, int z) const {
    return *touchValueAt(x, y, z);
  }
};


}
