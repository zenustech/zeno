#pragma once

#include <vector>
#include <array>

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

    void resize(size_t size) const {
        m_data.resize((size + ChunkSize - 1) / ChunkSize);
        for (size_t i = m_size; i < size; i++) {
            for (int j = 0; j < N; j++) {
                at(i, j).T();
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
};

#include <variant>

using VariantArray = std::variant<Array<float, 1>, Array<float, 3>>;
