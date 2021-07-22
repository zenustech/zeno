#include <memory>

struct _AnyBase {
    virtual std::unique_ptr<_AnyBase> clone() const = 0;
    virtual ~_AnyBase() = default;
};

template <class T>
struct _AnyImpl : _AnyBase {
    T t;

    _AnyImpl(T const &t) : t(t) {}

    virtual std::unique_ptr<_AnyBase> clone() const {
        return std::make_unique<_AnyImpl<T>>(t);
    }
};

class Any {
    std::unique_ptr<_AnyBase> m_ptr;

public:
    Any() : m_ptr(nullptr) {}
    ~Any() = default;

    Any(Any const &a) : m_ptr(a.m_ptr->clone()) {
    }

    Any &operator=(std::nullptr_t) {
        m_ptr = nullptr;
        return *this;
    }

    Any &operator=(Any const &a) {
        m_ptr = a.m_ptr->clone();
        return *this;
    }

    template <class T>
    Any &operator=(T const &t) {
        emplace<T>(t);
        return *this;
    }

    template <class T, class ...Ts>
    void emplace(Ts &&...ts) {
        m_ptr = std::make_unique<_AnyImpl<T>>(std::forward<Ts>(ts)...);
    }
};

int main() {
    Any a;
}
