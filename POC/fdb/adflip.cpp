#include <functional>
#include "vec.h"

struct LeafNode {
    float m[8 * 8 * 8];
};

struct InternalNode {
    LeafNode* m[16 * 16 * 16];

    InternalNode() {
        for (int i = 0; i < 16 * 16 * 16; i++) {
            m[i] = nullptr;
        }
    }

    InternalNode(InternalNode const &) = delete;
    InternalNode &operator=(InternalNode const &) = delete;

    ~InternalNode() {
        for (int i = 0; i < 16 * 16 * 16; i++) {
            if (m[i]) {
                delete m[i];
                m[i] = nullptr;
            }
        }
    }
};

struct RootNode {
    InternalNode* m[32 * 32 * 32];

    RootNode() {
        for (int i = 0; i < 32 * 32 * 32; i++) {
            m[i] = nullptr;
        }
    }

    RootNode(RootNode const &) = delete;
    RootNode &operator=(RootNode const &) = delete;

    ~RootNode() {
        for (int i = 0; i < 32 * 32 * 32; i++) {
            if (m[i]) {
                delete m[i];
                m[i] = nullptr;
            }
        }
    }
};


void foreachLeaf
    ( RootNode* root
    , std::function<void(LeafNode*, fdb::vec3i)> cb
    ) {
    for (int i = 0; i < 32 * 32 * 32; i++) {
        int ix = i % 32 * 16, iy = i / 32 % 32 * 16, iz = i / 32 / 32 * 16;
        auto intern = root->m[i];
        if (!intern) continue;
        for (int z = iz; z < 16; z++) {
            for (int y = iy; y < 16; y++) {
                for (int x = ix; x < 16; x++) {
                    auto leaf = intern->m[i];
                    cb(leaf, fdb::vec3i(x, y, z));
                }
            }
        }
    }
}



