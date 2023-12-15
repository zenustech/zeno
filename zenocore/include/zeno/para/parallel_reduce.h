#pragma once

#include <zeno/para/execution.h>
#include <zeno/para/counter_iterator.h>
#include <zeno/utils/type_traits.h>
#include <zeno/utils/vec.h>
#include <numeric>
#include <limits>
#include <tuple>

namespace zeno {

template <class Index, class Value, class Reduce, class Transform>
Value parallel_reduce(Index first, Index last, Value initVal, Reduce reduceFn, Transform transformFn) {
    return std::transform_reduce(ZENO_PAR counter_iterator<Index>(first), counter_iterator<Index>(last),
            initVal, reduceFn, transformFn);
}

template <class It, class Transform = identity>
auto parallel_reduce_min(It first, It last, Transform transformFn = {}) {
    if (first == last) return std::decay_t<decltype(*first)>();
    return std::transform_reduce(ZENO_PAR_UNSEQ first, last, *first, [] (auto &&x, auto &&y) {
        return zeno::min(x, y);
    }, transformFn);
}

template <class It, class Transform = identity>
auto parallel_reduce_max(It first, It last, Transform transformFn = {}) {
    if (first == last) return std::decay_t<decltype(*first)>();
    return std::transform_reduce(ZENO_PAR_UNSEQ first, last, *first, [] (auto &&x, auto &&y) {
        return zeno::max(x, y);
    }, transformFn);
}

template <class It, class Transform = identity>
auto parallel_reduce_minmax(It first, It last, Transform transformFn = {}) {
    if (first == last) return std::make_pair(std::decay_t<decltype(*first)>(), std::decay_t<decltype(*first)>());
    return std::transform_reduce(ZENO_PAR_UNSEQ first, last, std::make_pair(*first, *first), [] (auto &&x, auto &&y) {
        return std::make_pair(zeno::min(x.first, y.first), zeno::max(x.second, y.second));
    }, [transformFn] (auto const &val) {
        return std::make_pair(val, val);
    });
}

template <class It, class Transform = identity>
auto parallel_reduce_sum(It first, It last, Transform transformFn = {}) {
    return std::transform_reduce(ZENO_PAR_UNSEQ first, last, std::decay_t<decltype(transformFn(*first))>(), [] (auto &&x, auto &&y) {
        return x + y;
    }, transformFn);
}

template <class It, class Transform = identity>
auto parallel_reduce_average(It first, It last, Transform transformFn = {}) {
    return parallel_reduce_sum(first, last, transformFn) / (last - first);
}

}
