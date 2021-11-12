#pragma once

#include <zeno/common.h>
#include <functional>
#include <vector>

ZENO_NAMESPACE_BEGIN
namespace ztd {

template <class T>
class property final {
public:
    class propview final {
    private:
        property *prop;
        std::function<void(T const &)> changed;

        void do_changed() const {
            changed(std::as_const(prop->val));
        }

    public:
        inline propview(property *prop, std::function<void(T const &)> const &changed)
            : prop(prop), changed(changed)
        {
            prop->children.push_back(this);
            do_changed();
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
                    ch.do_changed();
            }
        }
    };

private:
    T val;
    std::vector<propview *> children;

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
            ch.do_changed();
        }
    }

    propview view(std::function<void(T const &)> const &changed) const {
        return {this, changed};
    }
};

}
ZENO_NAMESPACE_END
