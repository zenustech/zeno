#pragma once

#include "HashMap_vec3i.h"
#include "Array.h"

namespace fdb {

template <class T, size_t N>
struct HashGrid {
    static_assert(0);
};

template <class T>
struct HashGrid<T, 8> {
    static_assert(std::is_constructible<T>::value);
    static_assert(std::is_trivially_copy_constructible<T>::value);
    static_assert(std::is_trivially_copy_assignable<T>::value);
    static_assert(std::is_trivially_move_constructible<T>::value);
    static_assert(std::is_trivially_move_assignable<T>::value);
    static_assert(std::is_trivially_destructible<T>::value);

    struct Leaf {
        Array<T, 8 * 8 * 8> m_data{};
        uint8_t m_mask[8 * 8]{};

        inline FDB_DEVICE bool is_on(vec3i coord) {
            size_t i = coord[1] | coord[2] << 3;
            return m_mask[i] & (1 << coord[0]);
        }

        inline FDB_DEVICE void turn_on(vec3i coord) {
            size_t i = coord[1] | coord[2] << 3;
            m_mask[i] |= 1 << coord[0];
        }

        inline FDB_DEVICE void turn_off(vec3i coord) {
            size_t i = coord[1] | coord[2] << 3;
            m_mask[i] &= ~(1 << coord[0]);
        }

        inline FDB_DEVICE T &at(vec3i coord) {
            size_t i = coord[0] | coord[1] << 3 | coord[2] << 6;
            return m_data[i];
        }

        inline FDB_DEVICE T &turn_on_at(vec3i coord) {
            size_t i = coord[1] | coord[2] << 3;
            m_mask[i] |= 1 << coord[0];
            return m_data[i << 3 | coord[0]];
        }

        template <class Func>
        inline FDB_DEVICE void foreach(Func func) {
            for (int i = 0; i < 8 * 8 * 8; i++) {
                int x = i & 0x7;
                if (leaf.m_mask[i >> 3] & (1 << x)) {
                    int y = (i >> 3) & 0x7;
                    int z = (i >> 6) & 0x7;
                    vec3i coord(x, y, z);
                    func(std::as_const(coord), leaf.m_data[i]);
                }
            }
        }

        template <class Func>
        inline FDB_DEVICE void foreach_with_off(Func func) {
            for (int i = 0; i < 8 * 8 * 8; i++) {
                int x = i & 0x7;
                int y = (i >> 3) & 0x7;
                int z = (i >> 6) & 0x7;
                vec3i coord(x, y, z);
                func(std::as_const(coord), leaf.m_data[i]);
            }
        }
    };

    HashMap<vec3i, Leaf> m_grid;

    inline FDB_CONSTEXPR size_t capacity() const {
        return m_grid.capacity();
    }

    inline void reserve(size_t n) {
        m_grid.reserve(n);
    }

    inline void clear() {
        m_grid.clear();
    }

    struct View {
        typename HashMap<vec3i, Leaf>::View m_view;

        View(HashGrid const &parent)
            : m_view(parent.m_grid.view())
        {}

        template <class Kernel>
        inline void parallel_foreach(Kernel kernel, ParallelConfig cfg = {256, 2}) const {
            parallel_foreach_leaf([=] FDB_DEVICE (vec3i leaf_coord, Leaf &leaf) {
                leaf.foreach([&] (vec3i sub_coord, T &val) {
                    vec3i coord = leaf_coord << 3 | sub_coord;
                    kernel(std::as_const(coord), leaf.m_data[i]);
                });
            }, cfg);
        }

        template <class Kernel>
        inline void parallel_foreach_leaf(Kernel kernel, ParallelConfig cfg = {256, 2}) const {
            m_view.parallel_foreach([=] FDB_DEVICE (vec3i leaf_coord, Leaf &leaf) {
                kernel(std::as_const(leaf_coord), leaf);
            }, cfg);
        }

        inline FDB_DEVICE Leaf *probe_leaf(vec3i coord) const {
            return m_view.find(coord);
        }

        inline FDB_DEVICE Leaf &touch_leaf(vec3i coord) const {
            return *m_view.touch(coord);
        }

        inline FDB_DEVICE Leaf &get_leaf(vec3i coord) const {
            return *m_view.find(coord);
        }

        template <class Func>
        inline FDB_DEVICE void foreach_in_leaf(vec3i coord, Func func) const {
            auto *leaf = probe_leaf(coord);
            if (leaf) {
                leaf->foreach(func);
            }
        }

        inline FDB_DEVICE T *probe(vec3i coord) const {
            auto *leaf = probe_leaf(coord >> 3);
            if (!leaf)
                return nullptr;
            coord &= 0x7;
            if (!leaf->is_on(coord))
                return nullptr;
            return &leaf->at(coord);
        }

        inline FDB_DEVICE T &operator[](vec3i coord) const {
            auto *leaf = &touch_leaf(coord >> 3);
            return leaf->turn_on_at(coord & 0x7);
        }

        inline FDB_DEVICE T &operator()(vec3i coord) const {
            auto *leaf = &get_leaf(coord >> 3);
            return leaf->at(coord & 0x7);
        }
    };

    View view() const {
        return *this;
    }
};

#if 0 // wip
template <class T>
struct HashGrid<T, 4> {
    static_assert(std::is_constructible<T>::value);
    static_assert(std::is_trivially_copy_constructible<T>::value);
    static_assert(std::is_trivially_copy_assignable<T>::value);
    static_assert(std::is_trivially_move_constructible<T>::value);
    static_assert(std::is_trivially_move_assignable<T>::value);
    static_assert(std::is_trivially_destructible<T>::value);

    struct Leaf {
        Array<T, 4 * 4 * 4> m_data{};
        uint8_t m_mask[4 * 2]{};

        inline FDB_DEVICE bool is_on(vec3i coord) {
            size_t i = coord[1] | coord[2] << 3;
            return m_mask[i] & (1 << coord[0]);
        }

        inline FDB_DEVICE void turn_on(vec3i coord) {
            size_t i = coord[1] | coord[2] << 3;
            m_mask[i] |= 1 << coord[0];
        }

        inline FDB_DEVICE void turn_off(vec3i coord) {
            size_t i = coord[1] | coord[2] << 3;
            m_mask[i] &= ~(1 << coord[0]);
        }

        inline FDB_DEVICE T &at(vec3i coord) {
            size_t i = coord[0] | coord[1] << 3 | coord[2] << 6;
            return m_data[i];
        }

        inline FDB_DEVICE T &turn_on_at(vec3i coord) {
            size_t i = coord[1] | coord[2] << 3;
            m_mask[i] |= 1 << coord[0];
            return m_data[i << 3 | coord[0]];
        }

        template <class Func>
        inline FDB_DEVICE void foreach(Func func) {
            for (int i = 0; i < 8 * 8 * 8; i++) {
                int x = i & 0x7;
                if (leaf.m_mask[i >> 3] & (1 << x)) {
                    int y = (i >> 3) & 0x7;
                    int z = (i >> 6) & 0x7;
                    vec3i coord(x, y, z);
                    func(std::as_const(coord), leaf.m_data[i]);
                }
            }
        }

        template <class Func>
        inline FDB_DEVICE void foreach_with_off(Func func) {
            for (int i = 0; i < 8 * 8 * 8; i++) {
                int x = i & 0x7;
                int y = (i >> 3) & 0x7;
                int z = (i >> 6) & 0x7;
                vec3i coord(x, y, z);
                func(std::as_const(coord), leaf.m_data[i]);
            }
        }
    };

    HashMap<vec3i, Leaf> m_grid;

    inline FDB_CONSTEXPR size_t capacity() const {
        return m_grid.capacity();
    }

    inline void reserve(size_t n) {
        m_grid.reserve(n);
    }

    inline void clear() {
        m_grid.clear();
    }

    struct View {
        typename HashMap<vec3i, Leaf>::View m_view;

        View(HashGrid const &parent)
            : m_view(parent.m_grid.view())
        {}

        template <class Kernel>
        inline void parallel_foreach(Kernel kernel, ParallelConfig cfg = {256, 2}) const {
            parallel_foreach_leaf([=] FDB_DEVICE (vec3i leaf_coord, Leaf &leaf) {
                leaf.foreach([&] (vec3i sub_coord, T &val) {
                    vec3i coord = leaf_coord << 3 | sub_coord;
                    kernel(std::as_const(coord), leaf.m_data[i]);
                });
            }, cfg);
        }

        template <class Kernel>
        inline void parallel_foreach_leaf(Kernel kernel, ParallelConfig cfg = {256, 2}) const {
            m_view.parallel_foreach([=] FDB_DEVICE (vec3i leaf_coord, Leaf &leaf) {
                kernel(std::as_const(leaf_coord), leaf);
            }, cfg);
        }

        inline FDB_DEVICE Leaf *probe_leaf(vec3i coord) const {
            return m_view.find(coord);
        }

        inline FDB_DEVICE Leaf &touch_leaf(vec3i coord) const {
            return *m_view.touch(coord);
        }

        inline FDB_DEVICE Leaf &get_leaf(vec3i coord) const {
            return *m_view.find(coord);
        }

        template <class Func>
        inline FDB_DEVICE void foreach_in_leaf(vec3i coord, Func func) const {
            auto *leaf = probe_leaf(coord);
            if (leaf) {
                leaf->foreach(func);
            }
        }

        inline FDB_DEVICE T *probe(vec3i coord) const {
            auto *leaf = probe_leaf(coord >> 3);
            if (!leaf)
                return nullptr;
            coord &= 0x7;
            if (!leaf->is_on(coord))
                return nullptr;
            return &leaf->at(coord);
        }

        inline FDB_DEVICE T &operator[](vec3i coord) const {
            auto *leaf = &touch_leaf(coord >> 3);
            return leaf->turn_on_at(coord & 0x7);
        }

        inline FDB_DEVICE T &operator()(vec3i coord) const {
            auto *leaf = &get_leaf(coord >> 3);
            return leaf->at(coord & 0x7);
        }
    };

    View view() const {
        return *this;
    }
};
#endif

}
