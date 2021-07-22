#pragma once

#include <memory>

namespace zeno {

template <class T>
struct shared_any_traits {
};

class shared_any {
    struct _AnyBase {
        virtual std::shared_ptr<_AnyBase> clone() const = 0;
        virtual bool assign(_AnyBase const *) = 0;
        virtual ~_AnyBase() = default;
    };

    template <class T>
    struct _AnyImpl : _AnyBase {
        T t;

        _AnyImpl(T const &t) : t(t) {}

        virtual std::shared_ptr<_AnyBase> clone() const {
            return std::make_shared<_AnyImpl<T>>(t);
        }

        virtual bool assign(_AnyBase const *b) {
            auto p = dynamic_cast<_AnyImpl<T> const *>(b);
            if (!p) return false;
            t = p->t;
            return true;
        }
    };

    /*template <class T, class = void>
    struct _AnyIface : _AnyBase {
        virtual T const &get() const = 0;

        virtual std::shared_ptr<_AnyBase> clone() const {
            return std::make_shared<_AnyImpl<T>>(get());
        }

        virtual bool assign(_AnyBase const *b) {
            auto p = dynamic_cast<_AnyImpl<T> const *>(b);
            if (!p) return false;
            const_cast<T &>(get()) = p->t;
            return true;
        }
    };

    template <class T, class = void>
    struct _AnyImpl : _AnyIface<T> {
        T t;

        _AnyImpl(T const &t) : t(t) {}

        virtual T const &get() const {
            return t;
        }
    };

    template <class T>
    struct _AnyImpl<T, std::void_t<
            typename shared_any_traits<T>::base
        >> : _AnyIface<T>
           , _AnyIface<typename shared_any_traits<T>::base>
    {
        T t;

        _AnyImpl(T const &t) : t(t) {}

        virtual T const &get() const {
            return t;
        }

        virtual std::shared_ptr<_AnyBase> clone() const {
            return std::make_shared<_AnyImpl<T>>(t);
        }

        virtual bool assign(_AnyBase const *b) {
            auto p = dynamic_cast<_AnyImpl<T> const *>(b);
            if (!p) return false;
            t = p->t;
            return true;
        }
    };*/

    std::shared_ptr<_AnyBase> m_ptr;

public:
    shared_any() : m_ptr(nullptr) {}
    ~shared_any() = default;
    shared_any(shared_any &&a) = default;
    shared_any(shared_any const &a)
        : m_ptr(a.m_ptr)
    {}

    template <class T>
    shared_any(T const &t)
        : m_ptr(std::make_shared<_AnyImpl<T>>(t))
    {}

    shared_any &operator=(std::nullptr_t) {
        m_ptr = nullptr;
        return *this;
    }

    shared_any &operator=(shared_any const &a) {
        m_ptr = a.m_ptr;
        return *this;
    }

    template <class T>
    T *cast() const {
        auto p = dynamic_cast<_AnyImpl<T> *>(m_ptr.get());
        if (!p) return nullptr;
        return &p->t;
    }

    template <class T>
    T &get() const {
        return *cast<T>();
    }

    template <class T>
    shared_any &operator=(T const &t) {
        emplace<T>(t);
        return *this;
    }

    void assign(shared_any const &a) const {
        m_ptr->assign(a.m_ptr.get());
    }

    template <class T, class ...Ts>
    void emplace(Ts &&...ts) {
        m_ptr = std::make_shared<_AnyImpl<T>>(std::forward<Ts>(ts)...);
    }

    template <class T, class ...Ts>
    static shared_any make(Ts &&...ts) {
        shared_any a;
        a.emplace<T>(std::forward<Ts>(ts)...);
        return a;
    }
};

}
