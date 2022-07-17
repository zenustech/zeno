#pragma once

#include <iterator>
#include <tuple>
#include <zeno/para/iterator_facade.h>

namespace zeno {

template <class It, class Func>
struct transform_iterator : iterator_adaptor<transform_iterator<It, Func>, It,
    std::invoke_result_t<Func, typename std::iterator_traits<It>::value_type>> {
    It it;
    Func func;

    transform_iterator() = default;

    template <class FuncT = Func>
    explicit transform_iterator(It it, FuncT &&func) : func(std::forward<FuncT>(func)) {}

    decltype(auto) dereference() const {
        return func(*it);
    }

    void increment() {
        ++it;
    }

    void decrement() {
        --it;
    }

    void advance(typename std::iterator_traits<It>::difference_type n) {
        it += n;
    }

    auto distance_to(transform_iterator const &that) const {
        return that.it - it;
    }

    bool equal_to(transform_iterator const &that) const {
        return it == that.it;
    }
};

template <class It, class Func>
transform_iterator(It, Func) -> transform_iterator<It, Func>;

}
