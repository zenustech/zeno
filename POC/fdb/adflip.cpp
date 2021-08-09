#include <functional>
#include <cstdio>
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

    InternalNode(InternalNode const&) = delete;
    InternalNode& operator=(InternalNode const&) = delete;

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

    RootNode(RootNode const&) = delete;
    RootNode& operator=(RootNode const&) = delete;

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
#pragma omp parallel for
    for (int i = 0; i < 32 * 32 * 32; i++) {
        int ix = i % 32 * 16, iy = i / 32 % 32 * 16, iz = i / 32 / 32 * 16;
        auto* intern = root->m[i];
        if (!intern) continue;
        for (int z = 0; z < 16; z++) {
            for (int y = 0; y < 16; y++) {
                for (int x = 0; x < 16; x++) {
                    auto j = x + y * 16 + z * 16 * 16;
                    auto* leaf = intern->m[j];
                    if (!leaf) continue;
                    cb(leaf, fdb::vec3i(ix + x, iy + y, iz + z));
                }
            }
        }
    }
}

LeafNode* getLeaf
    ( RootNode* root
    , fdb::vec3i coor
    ) {
    int x = coor[0] / 16, y = coor[1] / 16, z = coor[2] / 16;
    int i = x + y * 16 + z * 16 * 16;
    auto* intern = root->m[i];
    if (!intern) return nullptr;
    x = coor[0] % 16, y = coor[1] % 16, z = coor[2] % 16;
    int j = x + y * 16 + z * 16 * 16;
    auto* leaf = intern->m[j];
    return leaf;
}

LeafNode* addLeaf
    ( RootNode* root
    , fdb::vec3i coor
    ) {
    int x = coor[0] / 16, y = coor[1] / 16, z = coor[2] / 16;
    int i = x + y * 16 + z * 16 * 16;
    auto*& intern = root->m[i];
    if (!intern) intern = new InternalNode;
    x = coor[0] % 16, y = coor[1] % 16, z = coor[2] % 16;
    int j = x + y * 16 + z * 16 * 16;
    auto*& leaf = intern->m[j];
    if (!leaf) leaf = new LeafNode;
    return leaf;
}

void foreachElement
    ( LeafNode* leaf
    , std::function<void(float&, fdb::vec3i)> cb
    ) {
    for (int z = 0; z < 8; z++) {
        for (int y = 0; y < 8; y++) {
            for (int x = 0; x < 8; x++) {
                auto k = x + y * 8 + z * 8 * 8;
                cb(leaf->m[k], fdb::vec3i(x, y, z));
            }
        }
    }
}

struct Transform {
    fdb::vec3f bmin{-2048.f}, bmax{2048.f};

    fdb::vec3f to_world(fdb::vec3i leafCoor, fdb::vec3i elmCoor) {
        auto coor = leafCoor * 8 + elmCoor;
        return bmin + (bmax - bmin) / 4096.f * coor;
    }
};

int main() {
    auto* root = new RootNode;
    auto* tran = new Transform;
    addLeaf(root, {8, 9, 10});
    foreachLeaf(root, [&] (auto* leaf, auto leafCoor) {
        foreachElement(leaf, [&] (float& value, auto elmCoor) {
            auto pos = tran->to_world(leafCoor, elmCoor);
            value = pos[0];
        });
    });
    foreachLeaf(root, [&] (auto* leaf, auto) {
        foreachElement(leaf, [&] (float& value, auto) {
            printf("%f\n", value);
        });
    });
}



