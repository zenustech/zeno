#include <cstdint>
#include <cstddef>
#include <zinc/vec.h>

using namespace zinc;

template <class T>
struct AutoInit {
    T m_value{0};

    operator T *() const {
        return m_value;
    }

    auto &operator=(T const &p) {
        m_value = p;
        return *this;
    }
};


template <class T, size_t N>
struct Dense {
    AutoInit<T> m_data[N * N * N];

    Dense() = default;
    ~Dense() = default;
    Dense(Dense const &) = default;
    Dense &operator=(Dense const &) = default;
    Dense(Dense &&) = default;
    Dense &operator=(Dense &&) = default;

    [[nodiscard]] static uintptr_t linearize(vec3I coor) {
        //return dot(clamp(coor, 0, N-1), vec3L(1, N, N * N));
        return dot((coor + N) % N, vec3L(1, N, N * N));
        /*coor += N;
        uintptr_t i = dot((coor / 8) % (N/8), vec3L(1, N/8, N/8 * N/8));
        uintptr_t j = dot(coor % 8, vec3L(1, 8, 8 * 8));
        return 8*8*8 * i + j;*/
    }

    [[nodiscard]] T &operator()(vec3I coor) {
        uintptr_t i = linearize(coor);
        return m_data[i];
    }

    [[nodiscard]] T const &operator()(vec3I coor) const {
        uintptr_t i = linearize(coor);
        return m_data[i];
    }

    [[nodiscard]] decltype(auto) operator()(uint32_t x, uint32_t y, uint32_t z) {
        return operator()({x, y, z});
    }

    [[nodiscard]] decltype(auto) operator()(uint32_t x, uint32_t y, uint32_t z) const {
        return operator()({x, y, z});
    }
};


struct LeafNode {
    Dense<float, 8> m;
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

    void operator()(vec3I coor) {
    }
};


int main() {
}
