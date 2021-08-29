#pragma once

#include "DenseGrid.h"
#include "schedule.h"
#include <atomic>

namespace fdb::ppgrid {

template <typename T, int Log2Dim1 = 3, int Log2Dim2 = 4, int Log2Dim3 = 5, int Log2Offs3 = 4>
struct PPGrid {
    static constexpr int Log2Res = Log2Dim1 + Log2Dim2 + Log2Dim3;
    static constexpr int Log2Offs = Log2Offs3 != -1 ? Log2Dim1 + Log2Dim2 + Log2Offs3 : -1;
    static constexpr int Log2ResX = Log2Res;
    static constexpr int Log2ResY = Log2Res;
    static constexpr int Log2ResZ = Log2Res;
    static constexpr int Log2OffsX = Log2Offs;
    static constexpr int Log2OffsY = Log2Offs;
    static constexpr int Log2OffsZ = Log2Offs;
    using ValueType = T;

private:
    struct LeafNode {
        densegrid::DenseGrid<float, Log2Dim1> m_data;
    };

    struct InternalNode {
        densegrid::DenseGrid<LeafNode *, Log2Dim2> m_data;

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
        densegrid::DenseGrid<InternalNode *, Log2Dim3, Log2Offs3> m_data;
        densegrid::DenseGrid<AtomicCounterType, Log2Dim3, Log2Offs3> m_counter;

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
    LeafNode *get_leaf_at(vec3i ijk) const {
        auto *node = m_root.m_data.at(ijk >> Log2Dim2);
        if (!node) return nullptr;
        auto *leaf = node->m_data.at(ijk);
        return leaf;
    }

    LeafNode *add_leaf_at(vec3i ijk) {
        auto *&node = m_root.m_data.at(ijk >> Log2Dim2);
        if (!node)
            node = new InternalNode;
        auto *&leaf = node->m_data.at(ijk);
        if (!leaf) {
            ++m_root.m_counter.at(ijk >> Log2Dim2);
            leaf = new LeafNode;
        }
        return leaf;
    }

    void del_leaf_at(vec3i ijk) {
        auto *&node = m_root.m_data.at(ijk >> Log2Dim2);
        if (!node) return;
        auto *&leaf = node->m_data.at(ijk);
        if (leaf) {
            delete leaf;
            leaf = nullptr;
            if (!--m_root.m_counter.at(ijk >> Log2Dim2)) {
                delete node;
                node = nullptr;
            }
        }
    }

    T *get_at(vec3i ijk) const {
        auto *leaf = get_leaf_at(ijk >> Log2Dim1);
        if (!leaf) return nullptr;
        return &leaf->m_data.at(ijk);
    }

    T *add_at(vec3i ijk) {
        auto *leaf = add_leaf_at(ijk >> Log2Dim1);
        return &leaf->m_data.at(ijk);
    }

    void del_at(vec3i ijk) {
        del_leaf_at(ijk >> Log2Dim1);
    }

public:
    T const &at(vec3i ijk) const {
        return *get_at(ijk);
    }

    T &at(vec3i ijk) {
        return *add_at(ijk);
    }

    ValueType get(vec3i ijk) const {
        auto ptr = this->get_at(ijk);
        return ptr ? *ptr : ValueType(0);
    }

    void set(vec3i ijk, ValueType value) {
        *add_at(ijk) = value;
    }

    template <class Pol, class F>
    void foreach_leaf(Pol const &pol, F const &func) const {
        ndrange_for(pol, vec3i(0), vec3i(1 << Log2Dim3), [&] (auto ijk3) {
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
