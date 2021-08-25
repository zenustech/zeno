#pragma once

namespace fdb {

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
        return (*p)(coor & 15);
    }
};

struct VDBGrid {
    RootNode;
};

}
