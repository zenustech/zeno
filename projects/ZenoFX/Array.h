#pragma once

#include <vector>
#include <array>

template <class T, size_t N>
struct Array {
    static constexpr size_t ChunkSize = 1024;

    struct Chunk {
        std::array<std::array<T, ChunkSize>, N> m;
    };

    std::vector<Chunk> m_data;
    size_t m_size = 0;

    size_t size() const {
        return m_size;
    }

    void resize(size_t size) const {
        m_size = size;
        m_data.resize((size + ChunkSize - 1) / ChunkSize);
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
