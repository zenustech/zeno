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

    LeafNode *&operator()(Qint3 coor) {
        InternalNode *&p = m(coor >> 4);
        if (p) {
            p = new InternalNode;
        }
        return p->m(coor & 15);
    }
};

struct VDBGrid {
    std::unique_ptr<RootNode> m_root = std::make_unique<RootNode>();
};

}

using VDBGrid_details::VDBGrid;

}
