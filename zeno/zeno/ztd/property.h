#pragma once

#include <zeno/common.h>
#include <functional>
#include <vector>

ZENO_NAMESPACE_BEGIN
namespace ztd {

template <class T>
class property final {
    class propview final {
        friend property;

        property *prop;
        std::function<void(T const &)> changed;

        inline void do_changed() const {
            changed(std::as_const(prop->val));
        }

    public:
        inline propview(property *prop, auto &&changed)
            : prop(prop), changed(std::forward<decltype(changed)>(changed))
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

        inline void set(T nxt) const {
            prop->val = std::move(nxt);
            for (auto ch: prop->children) {
                if (ch != this)
                    ch->do_changed();
            }
        }
    };

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

    inline void set(T nxt) {
        val = std::move(nxt);
        for (auto ch: children) {
            ch->do_changed();
        }
    }

    inline propview view(auto &&changed) const {
        return {const_cast<property *>(this), std::forward<decltype(changed)>(changed)};
    }
};

template <class T>
class prop_list final {
    class propview final {
        friend prop_list;

        prop_list *prop;
        std::function<void(T const &)> added;
        std::function<void(T const &)> changed;
        std::function<void(T const &)> removed;

        inline void do_changed(size_t i) const {
            changed(std::as_const(prop->val[i]));
        }

        inline void do_added(size_t i) const {
            added(std::as_const(prop->val[i]));
        }

        inline void do_removed(size_t i) const {
            removed(std::as_const(prop->val[i]));
        }

    public:
        inline propview( prop_list *prop
                       , auto &&added
                       , auto &&changed
                       , auto &&removed
                       )
            : prop(prop)
            , added(std::forward<decltype(added)>(added))
            , changed(std::forward<decltype(changed)>(changed))
            , removed(std::forward<decltype(removed)>(removed))
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

        inline void set(T &&nxt) const {
            prop->val = std::move(nxt);
            for (auto ch: prop->children) {
                if (ch != this)
                    ch->do_changed();
            }
        }
    };

    std::vector<T> val;
    std::vector<propview *> children;

public:
    prop_list() = default;
    prop_list(prop_list const &) = delete;
    prop_list &operator=(prop_list const &) = delete;
    prop_list(prop_list &&) = default;
    prop_list &operator=(prop_list &&) = default;

    inline std::vector<T> const &get() const {
        return val;
    }

    inline T const &get(size_t i) const {
        return val;
    }

    inline size_t size() const {
        return val.size();
    }

    inline size_t add(T nxt) {
        size_t i = val.size();
        val.push_back(std::move(nxt));
        for (auto ch: children) {
            ch->do_added(i);
        }
        return i;
    }

    inline void set(size_t i, T nxt) {
        val.at(i) = std::move(nxt);
        for (auto ch: children) {
            ch->do_changed(i);
        }
    }

    inline void remove(size_t i) {
        (void)val.at(i);
        for (auto ch: children) {
            ch->do_removed(i);
        }
        val.erase(val.begin() + i);
    }

    inline void pop(size_t i) {
        if (val.empty())
            return;
        size_t i = val.size() - 1;
        for (auto ch: children) {
            ch->do_removed(i);
        }
        val.pop_back();
    }

    inline propview view(auto &&changed) const {
        return {const_cast<prop_list *>(this), std::forward<decltype(changed)>(changed)};
    }
};

}
ZENO_NAMESPACE_END
