#pragma once

#include <zeno/parallel/execution.h>
#include <zeno/parallel/counter_iterator.h>
#include <zeno/utils/vec.h>
#include <numeric>
#include <limits>

namespace zeno {

template <class Pol, class Index, class Value, class Reduce, class Transform>
Value parallel_reduce(Pol pol, Index first, Index last, Value identity, Reduce reduceFn, Transform transformFn) {
    return std::transform_reduce(pol, counter_iterator<Index>(first), counter_iterator<Index>(last), identity, reduceFn, transformFn);
}

template <class Pol, class Index, class Value, class Reduce, class Transform>
Value parallel_reduce(Pol pol, Index first, Index last, Value identity, Reduce reduceFn) {
    return std::reduce(pol, first, last, identity, reduceFn);
}

template <class Pol, class It, class ...Extras>
auto parallel_reduce_min(Pol pol, It first, It last, Extras ...extras) {
    return parallel_reduce(pol, first, last, *first, [] (auto &&x, auto &&y) {
        return min(x, y);
    }, extras...);
}

template <class Pol, class It, class ...Extras>
auto parallel_reduce_max(Pol pol, It first, It last, Extras ...extras) {
    return parallel_reduce(pol, first, last, *first, [] (auto &&x, auto &&y) {
        return max(x, y);
    }, extras...);
}

template <class Pol, class It, class ...Extras>
auto parallel_reduce_minmax(Pol pol, It first, It last, Extras ...extras) {
    return parallel_reduce(pol, first, last, std::make_pair(*first, *first), [] (auto &&x, auto &&y) {
        return std::make_pair(min(x.first, y.first), max(x.second, y.second));
    }, extras...);
}

template <class Pol, class It, class ...Extras>
auto parallel_reduce_sum(Pol pol, It first, It last, Extras ...extras) {
    return parallel_reduce(pol, first, last, *first, [] (auto &&x, auto &&y) {
        return std::move(x) + std::move(y);
    }, extras...);
}

template <class Pol, class It, class ...Extras>
auto parallel_reduce_average(Pol pol, It first, It last, Extras ...extras) {
    return parallel_reduce_sum(pol, first, last, extras...) / (last - first);
}

}
