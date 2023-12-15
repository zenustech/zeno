#pragma once

#include <zeno/para/execution.h>
#include <zeno/para/counter_iterator.h>
#include <zeno/utils/type_traits.h>
#include <zeno/utils/vec.h>
#include <numeric>
#include <limits>
#include <tuple>

namespace zeno {

template <class Index, class OutputIt, class Value, class Reduce, class Transform>
OutputIt parallel_inclusive_scan(Index first, Index last, OutputIt dest,
                    Value initVal, Reduce reduceFn, Transform transformFn) {
    return std::transform_inclusive_scan(ZENO_PAR
            counter_iterator<Index>(first), counter_iterator<Index>(last),
            dest, reduceFn, transformFn, initVal);
}

template <class It, class OutputIt, class Transform = identity>
OutputIt parallel_inclusive_scan_sum(It first, It last, OutputIt dest, Transform transformFn = {}) {
    return std::transform_inclusive_scan(ZENO_PAR_UNSEQ first, last, dest, [] (auto &&x, auto &&y) {
        return x + y;
    }, transformFn, std::decay_t<decltype(transformFn(*first))>());
}

template <class Index, class OutputIt, class Value, class Reduce, class Transform>
Value parallel_exclusive_scan(Index first, Index last, OutputIt dest,
                    Value initVal, Reduce reduceFn, Transform transformFn) {
    auto endp = std::transform_exclusive_scan(ZENO_PAR
            counter_iterator<Index>(first), counter_iterator<Index>(last),
            dest, initVal, reduceFn, transformFn);
    if (first != last)
        return reduceFn(*std::prev(endp), transformFn(*std::prev(last)));
    else
        return initVal;
}

template <class It, class OutputIt, class Transform = identity>
auto parallel_exclusive_scan_sum(It first, It last, OutputIt dest, Transform transformFn = {}) {
    auto endp = std::transform_exclusive_scan(ZENO_PAR_UNSEQ first, last, dest, std::decay_t<decltype(transformFn(*first))>(), [] (auto &&x, auto &&y) {
        return x + y;
    }, transformFn);
    if (first != last)
        return *std::prev(endp) + transformFn(*std::prev(last));
    else
        return std::decay_t<decltype(transformFn(*first))>();
}

}
