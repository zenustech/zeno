#pragma once

#include <zeno/para/execution.h>
#include <zeno/para/counter_iterator.h>
#include <algorithm>

namespace zeno {

template <class Index, class Func>
void parallel_for(Index first, Index last, Func func) {
    std::for_each(ZENO_PAR counter_iterator<Index>(first), counter_iterator<Index>(last), func);
}

template <class Index, class Func>
void parallel_for(Index count, Func func) {
    std::for_each(ZENO_PAR counter_iterator<Index>(Index{}), counter_iterator<Index>(count), func);
}

template <class It, class Func>
void parallel_for_each(It first, It last, Func func) {
    std::for_each(ZENO_PAR_UNSEQ first, last, func);
}

}
