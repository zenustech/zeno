#include <any>

struct IObject {
    virtual ~IObject() = default;
};

template <class Iface>
class any_implementation {
    mutable std::any m_storage;
    Iface &(*m_getter)(std::any &);

public:
    template <class T>
    any_implementation(T &&t)
        : m_storage(std::forward<T>(t))
        , m_getter([](std::any &storage) -> Iface & {
            return std::any_cast<T &>(storage);
        })
    {}

    Iface &operator*() {
        return m_getter(m_storage);
    }

    Iface const &operator*() const {
        return m_getter(m_storage);
    }

    Iface *operator->() { return &**this; }
    Iface const *operator->() const { return &**this; }
};
