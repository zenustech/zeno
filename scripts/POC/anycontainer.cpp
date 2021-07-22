#include <memory>
#include <cstdio>

class Any {
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
    Any() : m_ptr(nullptr) {}
    ~Any() = default;
    Any(Any &&a) = default;
    Any(Any const &a)
        : m_ptr(a.m_ptr->clone())
    {}

    template <class T>
    Any(T const &t)
        : m_ptr(std::make_unique<_AnyImpl<T>>(t))
    {}

    Any &operator=(std::nullptr_t) {
        m_ptr = nullptr;
        return *this;
    }

    Any &operator=(Any const &a) {
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
    Any &operator=(T const &t) {
        emplace<T>(t);
        return *this;
    }

    void assign(Any const &a) const {
        m_ptr->assign(a.m_ptr.get());
    }

    template <class T, class ...Ts>
    void emplace(Ts &&...ts) {
        m_ptr = std::make_unique<_AnyImpl<T>>(std::forward<Ts>(ts)...);
    }

    template <class T, class ...Ts>
    static Any make(Ts &&...ts) {
        Any a;
        a.emplace<T>(std::forward<Ts>(ts)...);
        return a;
    }
};

int main() {
    Any a = 32;
    printf("%d\n", a.get<int>());
    a = 32.1f;
    printf("%f\n", a.get<float>());
    return 0;
}
