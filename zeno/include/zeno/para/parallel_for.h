#pragma once

#include <zeno/para/execution.h>
#include <zeno/para/counter_iterator.h>
#include <algorithm>

namespace zeno {

template <ZENO_POL(class Pol,) class Index, class Func>
void parallel_for(ZENO_POL(Pol pol,) Index first, Index last, Func func) {
    std::for_each(ZENO_POL(pol,) counter_iterator<Index>(first), counter_iterator<Index>(last), func);
}

template <ZENO_POL(class Pol,) class It, class Func>
void parallel_for_each(ZENO_POL(Pol pol,) It first, It last, Func func) {
    std::for_each(ZENO_POL(pol,) first, last, func);
}

}
