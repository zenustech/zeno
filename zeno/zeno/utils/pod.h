#pragma once

#include <new>
#include <utility>

namespace zeno {

template <class T>
struct pod {
private:
    T m_t;
public:
    pod() {}

    pod(pod &&p) : m_t(std::move(p.m_t)) {}

    pod(pod const &p) : m_t(p.m_t) {}

    pod &operator=(pod &&p) {
        m_t = std::move(p.m_t);
        return *this;
    }

    pod &operator=(pod const &p) {
        m_t = p.m_t;
        return *this;
    }

    pod(T &&t) : m_t(std::move(t)) {}

    pod(T const &t) : m_t(t) {}

    pod &operator=(T &&t) {
        m_t = std::move(t);
        return *this;
    }

    pod &operator=(T const &t) {
        m_t = t;
        return *this;
    }

    operator T const &() const {
        return m_t;
    }

    operator T &() {
        return m_t;
    }

    T const &get() const {
        return m_t;
    }

    T &get() {
        return m_t;
    }

    template <class ...Ts>
    pod &emplace(Ts &&...ts) {
        ::new (&m_t) T(std::forward<Ts>(ts)...);
        return *this;
    }

    void destroy() {
        m_t.~T();
    }
};

template <class AllocT>
struct pod_allocator : AllocT {
    template <class T, class ...Args>
    void construct(T *p, Args &&...args) const {
        if constexpr (!(sizeof...(Args) == 0 && std::is_pod_v<T>))
            ::new((void *)p) T(std::forward<Args>(args)...);
    }
};

}
