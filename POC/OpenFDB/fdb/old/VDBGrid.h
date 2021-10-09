#pragma once

#include <memory>
#include <atomic>
#include "Dense.h"

namespace fdb {

template <class T>
struct VDBGrid {
    using LeafNode = Dense<T, 8>;

    struct InternalNode {
        Dense<LeafNode *, 16> m_data;

        InternalNode() = default;
        ~InternalNode() = default;
        InternalNode(InternalNode const &) = delete;
        InternalNode &operator=(InternalNode const &) = delete;
        InternalNode(InternalNode &&) = default;
        InternalNode &operator=(InternalNode &&) = default;
    };

    struct RootNode {
        Dense<InternalNode *, 32> m_data;
        Dense<std::atomic<ushort>, 32> m_counter;

        RootNode() = default;
        ~RootNode() = default;
        RootNode(RootNode const &) = delete;
        RootNode &operator=(RootNode const &) = delete;
        RootNode(RootNode &&) = default;
        RootNode &operator=(RootNode &&) = default;

        LeafNode *add(Quint3 coor) {
            InternalNode *&node = m_data(coor >> 4);
            if (!node) {
                node = new InternalNode;
            }
            LeafNode *&leaf = node->m_data(coor & 15);
            if (!leaf) {
                leaf = new LeafNode;
                ++m_counter(coor >> 4);
            }
            return leaf;
        }

        [[nodiscard]] LeafNode *get(Quint3 coor) {
            InternalNode *&node = m_data(coor >> 4);
            if (!node) {
                return nullptr;
            }
            LeafNode *leaf = node->m_data(coor & 15);
            return leaf;
        }

        void del(Quint3 coor) {
            InternalNode *&node = m_data(coor >> 4);
            LeafNode *&leaf = node->m_data(coor & 15);
            if (leaf) {
                delete leaf;
                leaf = nullptr;
                if (--m_counter(coor >> 4) <= 0) {
                    delete node;
                }
            }
        }

        template <class Pol, class F>
        void foreach(Pol const &pol, F const &func) {
            m_data.foreach(pol, [&] (Quint3 coor1, InternalNode *node) {
                if (node) {
                    node->m_data.foreach(policy::Serial{}, [&] (Quint3 coor2, LeafNode *leaf) {
                        if (leaf) {
                            Quint3 coor = coor1 << 4 | coor2;
                            func(coor, leaf);
                        }
                    });
                }
            });
        }
    };

    std::unique_ptr<RootNode> m_root = std::make_unique<RootNode>();

    LeafNode *add(Quint3 coor) const {
        return m_root->add(coor);
    }

    [[nodiscard]] LeafNode *get(Quint3 coor) const {
        return m_root->get(coor);
    }

    void del(Quint3 coor) const {
        return m_root->del(coor);
    }

    [[nodiscard]] T read_at(Quint3 coor) const {
        auto leaf = get(coor >> 3);
        return leaf ? leaf->at(coor & 7) : T(0);
    }

    void write_at(Quint3 coor, T value) const {
        auto leaf = add(coor >> 3);
        leaf->at(coor & 7) = value;
    }

    template <class Pol, class F>
    void foreach(Pol const &pol, F const &func) const {
        return m_root->foreach(pol, func);
    }
};

}
