#ifndef __NODE_STATUS_H__
#define __NODE_STATUS_H__

namespace zeno {

    enum NodeStatus : unsigned int
    {
        Null = 0,
        Cached = 1,
        Mute = 1 << 1,
        Once = 1 << 2,
        View = 1 << 3,
    };

    constexpr NodeStatus operator|(NodeStatus X, NodeStatus Y) {
        return static_cast<NodeStatus>(
            static_cast<unsigned int>(X) | static_cast<unsigned int>(Y));
    }

    constexpr NodeStatus operator&(NodeStatus X, NodeStatus Y) {
        return static_cast<NodeStatus>(
            static_cast<unsigned int>(X) & static_cast<unsigned int>(Y));
    }

    constexpr NodeStatus operator^(NodeStatus X, NodeStatus Y) {
        return static_cast<NodeStatus>(
            static_cast<unsigned int>(X) ^ static_cast<unsigned int>(Y));
    }
}

#endif