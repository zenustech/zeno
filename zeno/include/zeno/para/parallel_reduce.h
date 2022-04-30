#pragma once

#include <zeno/para/execution.h>
#include <zeno/para/counter_iterator.h>
#include <zeno/utils/type_traits.h>
#include <zeno/utils/vec.h>
#include <numeric>
#include <limits>
#include <tuple>

namespace zeno {

template <class Pol, class Index, class Value, class Reduce, class Transform>
Value parallel_reduce(Pol pol, Index first, Index last, Value identity, Reduce reduceFn, Transform transformFn) {
    return std::transform_reduce(pol, counter_iterator<Index>(first), counter_iterator<Index>(last), identity, reduceFn, transformFn);
}

template <class Pol, class It, class Transform = identity>
auto parallel_reduce_min(Pol pol, It first, It last, Transform transformFn = {}) {
    return parallel_reduce(pol, first, last, *first, [] (auto &&x, auto &&y) {
        return zeno::min(x, y);
    }, transformFn);
}

template <class Pol, class It, class Transform = identity>
auto parallel_reduce_max(Pol pol, It first, It last, Transform transformFn = {}) {
    return parallel_reduce(pol, first, last, *first, [] (auto &&x, auto &&y) {
        return zeno::max(x, y);
    }, transformFn);
}

template <class Pol, class It, class Transform = identity>
auto parallel_reduce_minmax(Pol pol, It first, It last, Transform transformFn = {}) {
    return parallel_reduce(pol, first, last, std::make_pair(*first, *first), [] (auto &&x, auto &&y) {
        return std::make_pair(min(x.first, y.first), max(x.second, y.second));
    }, [transformFn] (auto const &val) {
        return std::make_pair(val, val);
    });
}

template <class Pol, class It, class Transform = identity>
auto parallel_reduce_sum(Pol pol, It first, It last, Transform transformFn = {}) {
    return parallel_reduce(pol, first, last, *first, [] (auto &&x, auto &&y) {
        return std::move(x) + std::move(y);
    }, transformFn);
}

template <class Pol, class It, class Transform = identity>
auto parallel_reduce_average(Pol pol, It first, It last, Transform transformFn = {}) {
    return parallel_reduce_sum(pol, first, last, transformFn) / (last - first);
}

}
