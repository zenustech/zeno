#pragma once

#include <cstddef>
#include <utility>

template <class T, class Alloc>
struct Vector {
    T *m_base{nullptr};
    size_t m_cap{0};
    size_t m_size{0};
    Alloc m_alloc;

    explicit Vector(Alloc alloc, size_t n = 0)
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

    struct View {
        T *m_base;
        size_t m_size;

        View(Vector const &parent)
            : m_base(parent.m_base)
            , m_size(parent.m_size)
        {}

        using iterator = T *;

        iterator find(size_t i) const {
            return m_base + i;
        }

        T &operator[](size_t i) const {
            return *find(i);
        }

        iterator begin() const {
            return m_base;
        }

        iterator end() const {
            return m_base + m_size;
        }

        size_t size() const {
            return m_size;
        }
    };

    View view() const {
        return {*this};
    }

    struct ConstView {
        T const *m_base;
        size_t m_size;

        ConstView(Vector const &parent)
            : m_base(parent.m_base)
            , m_size(parent.m_size)
        {}

        T const &operator[](size_t i) const {
            return m_base[i];
        }

        using iterator = T const *;

        iterator begin() const {
            return m_base;
        }

        iterator end() const {
            return m_base + m_size;
        }

        size_t size() const {
            return m_size;
        }
    };

    ConstView const_view() const {
        return {*this};
    }

    void push_back(T const &val) {
        size_t idx = m_size;
        size_t new_size = m_size + 1;
        if (new_size >= m_cap) {
            reserve(new_size + (new_size >> 1) + 1);
        }
        m_size = new_size;
        m_base[idx] = val;
    }
};
