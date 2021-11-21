#include <functional>

template <class T>
class propview;

template <class T>
class property {
    T val;
    std::vector<propview<T> *> children;

    friend propview<T>;

public:
    property() = default;
    property(property const &) = delete;
    property &operator=(property const &) = delete;
    property(property &&) = default;
    property &operator=(property &&) = default;

    inline T const &get() const {
        return val;
    }

    inline void set(T const &nxt) {
        val = nxt;
        for (auto ch: children) {
            ch.changed(val);
        }
    }
};

template <class T>
class propview {
    property<T> *prop;
    std::function<void(T const &)> changed;

public:
    inline propview(property<T> *prop, std::function<void(T const &)> const &changed)
        : prop(prop), changed(changed)
    {
        prop->children.push_back(this);
    }

    propview(propview const &) = default;
    propview &operator=(propview const &) = default;
    propview(propview &&) = default;
    propview &operator=(propview &&) = default;

    inline T const &get() const {
        return prop->get();
    }

    inline void set(T const &nxt) const {
        prop->val = nxt;
        for (auto ch: prop->children) {
            if (ch != this)
                ch.changed(prop->val);
        }
    }
};
