#pragma once

#include <vector>
#include <array>

#if 0
template <class T, size_t N>
struct Array {
    static constexpr size_t ChunkSize = 1024;
    static constexpr size_t Dimension = N;
    using ValueType = T;

    struct Chunk {
        std::array<std::array<T, ChunkSize>, N> m;
    };

    std::vector<Chunk> m_data;
    size_t m_size = 0;

    size_t size() const {
        return m_size;
    }

    void resize(size_t size) {
        m_data.resize((size + ChunkSize - 1) / ChunkSize);
        for (size_t i = m_size; i < size; i++) {
            for (int j = 0; j < N; j++) {
                at(i, j) = T();
            }
        }
        m_size = size;
    }

    T const *_M_at(size_t index, size_t compid) const {
        return &m_data[index / ChunkSize].m[compid][index % ChunkSize];
    }

    T const &at(size_t index, size_t compid) const {
        return *_M_at(index, compid);
    }

    T &at(size_t index, size_t compid) {
        return *const_cast<T *>(_M_at(index, compid));
    }

    std::vector<Chunk> &chunks() {
        return m_data;
    }

    std::vector<Chunk> const &chunks() const {
        return m_data;
    }
};
#else
template <class T, size_t N>
struct Array {
    static constexpr size_t Dimension = N;
    using ValueType = T;

    std::array<std::vector<T>, N> m_data;
    size_t m_size = 0;

    size_t size() const {
        return m_size;
    }

    void resize(size_t size) {
        for (auto &comp: m_data) {
            comp.resize(size);
        }
        m_size = size;
    }

    T const *_M_at(size_t index, size_t compid) const {
        return &m_data[compid][index];
    }

    T const *data(size_t compid) const {
        return _M_at(0, compid);
    }

    T *data(size_t compid) {
        return const_cast<T *>(_M_at(0, compid));
    }

    T const &at(size_t index, size_t compid) const {
        return *_M_at(index, compid);
    }

    T &at(size_t index, size_t compid) {
        return *const_cast<T *>(_M_at(index, compid));
    }
};
#endif

#include <variant>

using VariantArray = std::variant<Array<float, 1>, Array<float, 3>>;
