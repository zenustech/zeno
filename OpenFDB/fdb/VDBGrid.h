#pragma once

#include <memory>
#include "Dense.h"

namespace fdb {

namespace VDBGrid_details {

struct LeafNode {
    Dense<Qfloat, 8> m;
};

struct InternalNode {
    Dense<LeafNode *, 16> m;

    InternalNode() = default;
    ~InternalNode() = default;
    InternalNode(InternalNode const &) = delete;
    InternalNode &operator=(InternalNode const &) = delete;
    InternalNode(InternalNode &&) = default;
    InternalNode &operator=(InternalNode &&) = default;
};

struct RootNode {
    Dense<InternalNode *, 32> m;

    RootNode() = default;
    ~RootNode() = default;
    RootNode(RootNode const &) = delete;
    RootNode &operator=(RootNode const &) = delete;
    RootNode(RootNode &&) = default;
    RootNode &operator=(RootNode &&) = default;

    LeafNode *&operator()(Quint3 coor) {
        InternalNode *&node = m(coor >> 4);
        if (node) {
            node = new InternalNode;
        }
        return node->m(coor & 15);
    }

    template <class Pol, class F>
    void foreach(Pol const &pol, F const &func) {
        m.foreach(pol, [&] (Quint3 coor1, InternalNode *node) {
            if (node) {
                node->m.foreach(policy::Serial{}, [&] (Quint3 coor2, LeafNode *&leaf) {
                        Quint3 coor = coor1 << 4 | coor2;
                        func(coor, leaf);
                });
            }
        });
    }
};

struct VDBGrid {
    std::unique_ptr<RootNode> m_root = std::make_unique<RootNode>();

    LeafNode *&operator()(Quint3 coor) const {
        return m_root->operator()(coor);
    }

    template <class Pol, class F>
    LeafNode *&foreach(Pol const &pol, F const &func) const {
        return m_root->foreach(pol, func);
    }
};

}

using VDBGrid_details::VDBGrid;

}
