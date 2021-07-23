#include <memory>
#include <cstdio>

class unique_any {
    struct _AnyBase {
        virtual std::unique_ptr<_AnyBase> clone() const = 0;
        virtual bool assign(_AnyBase const *b) = 0;
        virtual ~_AnyBase() = default;
    };

    template <class T>
    struct _AnyImpl : _AnyBase {
        T t;

        _AnyImpl(T const &t) : t(t) {}

        virtual std::unique_ptr<_AnyBase> clone() const {
            return std::make_unique<_AnyImpl<T>>(t);
        }

        virtual bool assign(_AnyBase const *b) {
            auto p = dynamic_cast<_AnyImpl<T> const *>(b);
            if (!p) return false;
            t = p->t;
            return true;
        }
    };

    std::unique_ptr<_AnyBase> m_ptr;

public:
    unique_any() : m_ptr(nullptr) {}
    ~unique_any() = default;
    unique_any(unique_any &&a) = default;
    unique_any(unique_any const &a)
        : m_ptr(a.m_ptr->clone())
    {}

    template <class T>
    unique_any(T const &t)
        : m_ptr(std::make_unique<_AnyImpl<T>>(t))
    {}

    unique_any &operator=(std::nullptr_t) {
        m_ptr = nullptr;
        return *this;
    }

    unique_any &operator=(unique_any const &a) {
        m_ptr = a.m_ptr->clone();
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
    unique_any &operator=(T const &t) {
        emplace<T>(t);
        return *this;
    }

    void assign(unique_any const &a) const {
        m_ptr->assign(a.m_ptr.get());
    }

    template <class T, class ...Ts>
    void emplace(Ts &&...ts) {
        m_ptr = std::make_unique<_AnyImpl<T>>(std::forward<Ts>(ts)...);
    }

    template <class T, class ...Ts>
    static unique_any make(Ts &&...ts) {
        unique_any a;
        a.emplace<T>(std::forward<Ts>(ts)...);
        return a;
    }
};

class shared_any {
    struct _AnyBase {
        virtual std::shared_ptr<_AnyBase> clone() const = 0;
        virtual bool assign(_AnyBase const *b) = 0;
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

int main() {
    shared_any a = 32;
    printf("%d\n", a.get<int>());
    a = 32.1f;
    printf("%f\n", a.get<float>());
    return 0;
}
