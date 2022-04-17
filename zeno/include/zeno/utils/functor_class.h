#pragma once

#include <utility>

#define ZENO_DEFINE_FUNCTOR(type, func) \
struct type { \
    template <class ...Ts> \
    auto operator()(Ts &&...ts) const { \
        return func(std::forward<Ts>(ts)...); \
    } \
}

#define ZENO_DEFINE_FUNCTOR_UNOP(type, op) \
struct type { \
    template <class T1> \
    auto operator()(T1 &&t1) const { \
        return op std::forward<T1>(t1); \
    } \
}

#define ZENO_DEFINE_FUNCTOR_BINOP(type, op) \
struct type { \
    template <class T1, class T2> \
    auto operator()(T1 &&t1, T2 &&t2) const { \
        return std::forward<T1>(t1) op std::forward<T2>(t2); \
    } \
}
