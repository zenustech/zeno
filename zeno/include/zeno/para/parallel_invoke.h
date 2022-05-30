#pragma once

#include <zeno/para/execution.h>
#include <initializer_list>
#include <functional>
#include <algorithm>

namespace zeno {

template <class ...Tasks>
void parallel_invoke(Tasks &&...tasks) {
    std::array<std::function<void()>, sizeof...(Tasks)> tmp{std::forward<Tasks>(tasks)...};
    std::for_each(ZENO_PAR tmp.begin(), tmp.end(), [] (auto &&f) { std::move(f)() });
}

//inline void parallel_invoke(std::initializer_list<std::function<void()> tasks) {
    //std::for_each(ZENO_PAR tasks.begin(), tasks.end(), [] (auto &&f) { std::move(f)() });
//}

}
