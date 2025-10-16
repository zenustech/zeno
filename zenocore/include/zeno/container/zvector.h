#pragma once

#ifndef __ZVECTOR_H__
#define __ZVECTOR_H__

#include <utility>

namespace zeno
{

template <class T>
class Vector
{
public:
    Vector() : m_size(0), m_data(0), m_capability(0) {}

    Vector(const Vector& rhs) noexcept {
        m_capability = rhs.m_capability;
        m_size = rhs.m_size;
        size_t nBytes = m_capability * sizeof(T);
        m_data = malloc(nBytes);
        memcpy(m_data, rhs.m_data, nBytes);
    }

    Vector(Vector&& rhs) noexcept {
        m_data = rhs.m_data;
        m_capability = rhs.m_capability;
        m_size = rhs.m_size;
        _init_reset(rhs);
    }

    Vector(size_t size, T&& val) noexcept
       : m_size(0), m_data(0), m_capability(0) {
        reallocate(size);
        m_size = size;
        for (size_t i = 0; i < m_size; i++) {
            new (locate(i)) T(std::move(val));
        }
    }

    Vector(size_t size, const T& val = T()) noexcept 
        : m_size(0), m_data(0), m_capability(0) {
        reallocate(size);
        m_size = size;
        for (size_t i = 0; i < m_size; i++) {
            new (locate(i)) T(val);
        }
    }

    ~Vector() noexcept {
        for (size_t i = 0; i < m_size; i++) {
            T* ptr = locate(i);
            ptr->~T();
        }
        free(m_data);
        m_capability = m_size = 0;
    }

    void resize(size_t sz) noexcept {
        if (sz > m_capability) {
            reallocate(sz);
        }
        if (sz > m_size) {
            //从m_size到sz，依次构造T
            for (size_t i = 0; i < sz - m_size; i++) {
                new (locate(i)) T();
            }
            m_size = sz;
        }
    }

    void reserve(size_t sz) noexcept {
        if (sz > m_capability) {
            reallocate(sz);
        }
    }

    T* data() const { return m_data; }
    size_t size() const noexcept { return m_size; }
    size_t capacity() const noexcept { return m_capability; }

    T* begin() { return static_cast<T*>(m_data); }
    T* end() { return  static_cast<T*>(m_data) + m_size; }
    const T* begin() const { return static_cast<T*>(m_data); }
    const T* end() const { return static_cast<T*>(m_data) + m_size; }

    void push_back(const T& val) noexcept {
        if (m_size == m_capability) {
            reallocate(-1);
        }
        new (locate(m_size)) T(val);
        m_size++;
    }

    void push_back(T&& val) noexcept {
        if (m_size == m_capability) {
            reallocate(-1);
        }
        new (locate(m_size)) T(std::move(val));
        m_size++;
    }

    const T operator[] (size_t idx) const noexcept {
        return *(static_cast<T*>(m_data) + idx);
    }

    T& operator[] (size_t idx) noexcept {
        return *(static_cast<T*>(m_data) + idx);
    }

    Vector& operator=(const Vector& rhs) noexcept {
        m_capability = rhs.m_capability;
        m_size = rhs.m_size;
        size_t nBytes = m_capability * sizeof(T);
        m_data = malloc(nBytes);
        memcpy(m_data, rhs.m_data, nBytes);
        return *this;
    }

private:
    inline T* locate(size_t idx) const noexcept {
        return static_cast<T*>(m_data) + idx;
    }

    void _init_reset(Vector& rhs) {
        rhs.m_data = nullptr;
        rhs.m_capability = rhs.m_size = 0;
    }

    void reallocate(size_t new_sz) noexcept {
        //当new_sz超过capablity时，重新分配一段新的内存块
        if (new_sz == -1) {
            //没有指定分配多少，就分配原来的两倍
            new_sz = m_capability * 2;
            if (new_sz == 0)
                new_sz = 1;
        }
        else if (new_sz <= m_capability) {
            return;
        }

        void* new_data = malloc(new_sz * sizeof(T));
        if (new_data) {
            memcpy(new_data, m_data, m_size * sizeof(T));
        }
        else {
            exit(-1);
        }
        free(m_data);
        m_data = new_data;
        m_capability = new_sz;
    }

    size_t m_size;
    size_t m_capability;
    void* m_data;
};

}

#endif