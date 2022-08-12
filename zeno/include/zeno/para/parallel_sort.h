#pragma once

#include <zeno/para/execution.h>
#include <zeno/para/counter_iterator.h>
#include <algorithm>

namespace zeno {

template <class It, class Func>
void parallel_sort(It first, It last, Func func) {
    std::sort(ZENO_PAR_UNSEQ first, last, func);
}

}
