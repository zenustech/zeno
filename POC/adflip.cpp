#include <functional>

struct LeafNode {
    float m[8 * 8 * 8];
};

struct InternalNode {
    LeafNode* m[16 * 16 * 16];
};

struct RootNode {
    InternalNode* m[32 * 32 * 32];
};


void foreachLeaf
    ( RootNode* root
    , std::function<void(LeafNode*, int, int, int)> cb
    ) {
    for (int i = 0; i < 32 * 32 * 32; i++) {
        int ix = i % 32 * 16, iy = i / 32 % 32 * 16, iz = i / 32 / 32 * 16;
        auto intern = root->m[i];
        if (!intern) continue;
        for (int x = ix; x < 16; x++) {
            for (int y = iy; y < 16; y++) {
                for (int z = iz; z < 16; z++) {
                    auto leaf = intern->m[i];
                    cb(leaf, x, y, z);
                }
            }
        }
    }
}
