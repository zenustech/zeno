#pragma once

#include "DenseGrid.h"
#include "schedule.h"
#include <atomic>

namespace fdb::ppgrid {

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
        densegrid::DenseGrid<ValueType, Log2Dim1> m_data;  // 2 KiB
    };

    struct InternalNode {
        densegrid::DenseGrid<LeafNode *, Log2Dim2> m_data;  // 32 KiB
        densegrid::DenseGrid<ValueType, Log2Dim2> m_tiles;  // 16 KiB

        ~InternalNode() {
            for (int i = 0; i < m_data.size(); i++) {
                if (m_data.data()[i]) {
                    delete m_data.data()[i];
                }
            }
        }
    };

    struct RootNode {
        using AtomicCounterType = std::atomic<
            std::conditional_t<(Log2Dim2 * 3 > 15), unsigned int,
            std::conditional_t<(Log2Dim2 * 3 > 7), unsigned short,
            unsigned char>>>;
        densegrid::DenseGrid<InternalNode *, Log2Dim3, IsOffseted> m_data;  // 256 KiB
        densegrid::DenseGrid<ValueType, Log2Dim3, IsOffseted> m_tiles;  // 128 KiB
        densegrid::DenseGrid<AtomicCounterType, Log2Dim3, IsOffseted> m_leafcnt;  // 64 KiB

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

    LeafNode *touch_leaf(vec3i ijk) {
        auto *&node = m_root.m_data.at(ijk >> Log2Dim2);
        if (!node)
            node = new InternalNode;
        auto *&leaf = node->m_data.at(ijk);
        if (!leaf) {
            ++m_root.m_leafcnt.at(ijk >> Log2Dim2);
            leaf = new LeafNode;
        }
        return leaf;
    }

    void delete_leaf(vec3i ijk) {
        auto *&node = m_root.m_data.at(ijk >> Log2Dim2);
        if (!node) return;
        auto *&leaf = node->m_data.at(ijk);
        if (leaf) {
            delete leaf;
            leaf = nullptr;
            if (!--m_root.m_leafcnt.at(ijk >> Log2Dim2)) {
                delete node;
                node = nullptr;
            }
        }
    }

    ValueType *get_value(vec3i ijk) const {
        auto *node = m_root.m_data.at(ijk >> Log2Dim2 + Log2Dim1);
        if (!node) return m_root.m_tiles.at(ijk >> Log2Dim2 + Log2Dim1);
        auto *leaf = node->m_data.at(ijk);
        if (!leaf) return node->m_tiles.at(ijk >> Log2Dim1);
        return &leaf->m_data.at(ijk);
    }

    ValueType *peek_value(vec3i ijk) const {
        auto *leaf = peek_leaf(ijk >> Log2Dim1);
        if (!leaf) return nullptr;
        return &leaf->m_data.at(ijk);
    }

    ValueType *touch_value(vec3i ijk) {
        auto *leaf = touch_leaf(ijk >> Log2Dim1);
        return &leaf->m_data.at(ijk);
    }

public:
    ValueType &at(vec3i ijk) {
        return *get_value(ijk);
    }

    ValueType const &at(vec3i ijk) const {
        return *get_value(ijk);
    }

    ValueType get(vec3i ijk) const {
        return *get_value(ijk);
    }

    void set(vec3i ijk, ValueType value) {
        *touch_value(ijk) = value;
    }

    template <class Pol, class F>
    void foreach_leaf(Pol const &pol, F const &func) const {
        int beg = 0, end = 1 << Log2Dim3;
        if constexpr (IsOffseted) {
            end >>= 1;
            beg = -end;
        }
        ndrange_for(pol, vec3i(beg), vec3i(end), [&] (auto ijk3) {
            auto *node = m_root.m_data.at(ijk3);
            if (node) {
                ndrange_for(Serial{}, vec3i(0), vec3i(1 << Log2Dim2), [&] (auto ijk2) {
                    auto *leaf = node->m_data.at(ijk2);
                    if (leaf) {
                        auto ijk = ijk3 << Log2Dim2 | ijk2;
                        func(ijk, leaf);
                    }
                });
            }
        });
    }

    template <class Pol, class F>
    void foreach(Pol const &pol, F const &func) const {
        foreach_leaf(pol, [&] (auto ijk23, auto *leaf) {
            leaf->m_data.foreach(Serial{}, [&] (auto ijk1, auto &value) {
                auto ijk = ijk1 | ijk23 << Log2Dim1;
                func(ijk, value);
            });
        });
    }
};

}
