#pragma once

#include "DenseGrid.h"
#include <atomic>

namespace fdb::vdbgrid {

template <typename T, size_t Log2Dim1 = 3, size_t Log2Dim2 = 4, size_t Log2Dim3 = 5>
struct VDBGrid {
    static constexpr size_t Log2Res = Log2Dim1 + Log2Dim2 + Log2Dim3;
    static constexpr size_t Log2ResX = Log2Res;
    static constexpr size_t Log2ResY = Log2Res;
    static constexpr size_t Log2ResZ = Log2Res;
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
        densegrid::DenseGrid<InternalNode *, Log2Dim3> m_data;
        densegrid::DenseGrid<AtomicCounterType, Log2Dim3> m_counter;

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
    T *get_at(vec3i ijk) const {
        auto *node = m_root.m_data.at(ijk >> Log2Dim1 + Log2Dim2);
        if (!node) return nullptr;
        auto *leaf = node->m_data.at(ijk >> Log2Dim1);
        if (!leaf) return nullptr;
        return &leaf->m_data.at(ijk);
    }

    T *add_at(vec3i ijk) {
        auto *&node = m_root.m_data.at(ijk >> Log2Dim1 + Log2Dim2);
        if (!node)
            node = new InternalNode;
        auto *&leaf = node->m_data.at(ijk >> Log2Dim1);
        if (!leaf) {
            ++m_root.m_counter.at(ijk >> Log2Dim1 + Log2Dim2);
            leaf = new LeafNode;
        }
        return &leaf->m_data.at(ijk);
    }

    void del_at(vec3i ijk) {
        auto *&node = m_root.m_data.at(ijk >> Log2Dim1 + Log2Dim2);
        if (!node) return;
        auto *&leaf = node->m_data.at(ijk >> Log2Dim1);
        if (leaf) {
            delete leaf;
            leaf = nullptr;
            if (!--m_root.m_counter.at(ijk >> Log2Dim1 + Log2Dim2)) {
                delete node;
                node = nullptr;
            }
        }
    }

public:
    T const &at(vec3i ijk) const {
        return *get_at(ijk);
    }

    T &at(vec3i ijk) {
        return *add_at(ijk);
    }

    ValueType get(vec3i ijk) const {
        return *get_at(ijk);
    }

    void set(vec3i ijk, ValueType value) {
        *add_at(ijk) = value;
    }
};

}
