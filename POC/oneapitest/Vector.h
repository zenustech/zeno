#pragma once

#include <cstddef>
#include <utility>

template <class T, class Alloc>
struct Vector {
    T *m_base{nullptr};
    size_t m_cap{0};
    size_t m_size{0};
    Alloc m_alloc;

    Vector(size_t n = 0, Alloc alloc = {})
        : m_base(n ? (T *)alloc.allocate(n * sizeof(T)) : nullptr)
        , m_cap(n), m_size(n)
        , m_alloc(std::move(alloc))
    {}

    Vector(Vector const &) = delete;
    Vector &operator=(Vector const &) = delete;

    ~Vector() {
        if (m_base) {
            m_alloc.deallocate(m_base);
            m_base = nullptr;
            m_cap = 0;
            m_size = 0;
        }
    }

    size_t size() const {
        return m_size;
    }

    size_t capacity() const {
        return m_cap;
    }

    void __recapacity(size_t n) {
        if (!n) {
            m_alloc.deallocate(m_base);
            m_base = nullptr;
        } else {
            if (m_base) {
                m_base = (T *)m_alloc.reallocate(m_base, m_size * sizeof(T), n * sizeof(T));
            } else {
                m_base = (T *)m_alloc.allocate(n * sizeof(T));
            }
        }
        m_cap = n;
    }

    void reserve(size_t n) {
        if (m_cap < n) {
            __recapacity(n);
        }
    }

    void shrink_to_fit() {
        if (m_cap > m_size) {
            __recapacity(m_size);
        }
    }

    void resize(size_t n) {
        reserve(n);
        m_size = n;
    }

    void clear() {
        resize(0);
    }

    T &operator[](size_t i) {
        return m_base[i];
    }

    T const &operator[](size_t i) const {
        return m_base[i];
    }

    using iterator = T *;
    using const_iterator = T const *;

    iterator begin() {
        return m_base;
    }

    iterator end() {
        return m_base + m_size;
    }

    const_iterator begin() const {
        return m_base;
    }

    const_iterator end() const {
        return m_base + m_size;
    }

    iterator __push_back() {
        size_t idx = m_size;
        size_t new_size = m_size + 1;
        if (new_size >= m_cap) {
            reserve(new_size + (new_size >> 1) + 1);
        }
        m_size = new_size;
        return m_base + idx;
    }

    void push_back(T const &val) {
        *__push_back() = val;
    }
};
