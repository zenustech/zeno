#pragma once

#include <zeno/para/execution.h>
#include <zeno/para/counter_iterator.h>
#include <algorithm>

namespace zeno {

template <class Pol, class Index, class Func>
void parallel_for(Pol pol, Index first, Index last, Func func) {
    std::for_each(pol, counter_iterator<Index>(first), counter_iterator<Index>(last), func);
}

template <class Pol, class It, class Func>
void parallel_for_each(Pol pol, It first, It last, Func func) {
    std::for_each(pol, first, last, func);
}

}
