#pragma once

#include "DenseGrid.h"
#include "schedule.h"
#include <atomic>

namespace fdb::ppgrid {

template <typename T>
inline atomic_allocate_pointer(std::atomic<T *> &ptr) {
    static thread_local T *preallocated = nullptr;
    if (!preallocated) preallocated = new T;
    T *old_ptr = ptr;
    while (!ptr.compare_exchage_weak(old_ptr, preallocated));
    preallocated = new T;
}

template <typename T, typename CounterT>
inline atomic_allocate_pointer(T *&ptr, CounterT &cnt) {
}

template <typename T, int Log2Dim1 = 3, int Log2Dim2 = 4, int Log2Dim3 = 5, bool IsOffseted = true>
struct PPGrid {
    static constexpr int Log2Res = Log2Dim1 + Log2Dim2 + Log2Dim3;
    static constexpr int Log2ResX = Log2Res;
    static constexpr int Log2ResY = Log2Res;
    static constexpr int Log2ResZ = Log2Res;
    static constexpr bool Offseted = IsOffseted;
    using ValueType = T;

private:
    struct LeafNode {
        densegrid::DenseGrid<ValueType, Log2Dim1, false> m_data;  // 2 KiB
    };

    struct InternalNode {
        densegrid::DenseGrid<std::atomic<LeafNode *>, Log2Dim2, false> m_data;  // 32 KiB
        densegrid::DenseGrid<ValueType, Log2Dim2, false> m_tiles;  // 16 KiB

        ~InternalNode() {
            for (int i = 0; i < m_data.size(); i++) {
                if (m_data.data()[i]) {
                    delete m_data.data()[i];
                }
            }
        }
    };

    struct RootNode {
        densegrid::DenseGrid<std::atomic<InternalNode *>, Log2Dim3, IsOffseted> m_data;  // 256 KiB
        densegrid::DenseGrid<ValueType, Log2Dim3, IsOffseted> m_tiles;  // 128 KiB

        ~RootNode() {
            for (int i = 0; i < m_data.size(); i++) {
                if (m_data.data()[i]) {
                    delete m_data.data()[i];
                }
            }
        }
    };

    RootNode m_root;

protected:
    LeafNode *peek_leaf(vec3i ijk) const {
        auto *node = m_root.m_data.at(ijk >> Log2Dim2);
        if (!node) return nullptr;
        auto *leaf = node->m_data.at(ijk);
        return leaf;
    }

    LeafNode *touch_leaf_unsafe(vec3i ijk) {
        auto &node_a = m_root.m_data.at(ijk >> Log2Dim2);
        auto *node = node_a.load();
        if (!node) {
            node = new InternalNode;
            node_a.store(node);
        }
        auto &leaf_a = node->m_data.at(ijk);
        auto *leaf = leaf_a.load();
        if (!leaf) {
            leaf = new LeafNode;
            leaf_a.store(leaf);
        }
        return leaf;
    }

    LeafNode *touch_leaf(vec3i ijk) {
        auto *node = atomic_allocate_pointer(m_root.m_data.at(ijk >> Log2Dim2));
        auto *leaf = atomic_allocate_pointer(node->m_data.at(ijk));
        return leaf;
    }

public:
    ValueType get(vec3i ijk) const {
        auto *node = m_root.m_data.at(ijk >> Log2Dim2 + Log2Dim1);
        if (!node) return m_root.m_tiles.get(ijk >> Log2Dim2 + Log2Dim1);
        auto *leaf = node->m_data.at(ijk);
        if (!leaf) return node->m_tiles.get(ijk >> Log2Dim1);
        return leaf->m_data.get(ijk);
    }

    void set_unsafe(vec3i ijk, ValueType value) {
        auto *leaf = touch_leaf_unsafe(ijk >> Log2Dim1);
        leaf->m_data.set(ijk, value);
    }

    void set(vec3i ijk, ValueType value) {
        auto *leaf = touch_leaf(ijk >> Log2Dim1);
        leaf->m_data.set(ijk, value);
    }

    template <class Pol, class F>
    void foreach_leaf(Pol const &pol, F const &func) {
        m_root.m_data.foreach(pol, [&] (auto ijk3, auto *node) {
            if (node) {
                node->m_data.foreach(Serial{}, [&] (auto ijk2, auto *leaf) {
                    if (leaf) {
                        auto ijk = (ijk3 << Log2Dim2) + ijk2;
                        func(ijk, leaf);
                    }
                });
            }
        });
    }

    template <class Pol, class F>
    void foreach(Pol const &pol, F const &func) {
        foreach_leaf(pol, [&] (auto ijk23, auto *leaf) {
            leaf->m_data.foreach(Serial{}, [&] (auto ijk1, auto &value) {
                auto ijk = ijk1 | ijk23 << Log2Dim1;
                func(ijk, value);
            });
        });
    }
};

}
