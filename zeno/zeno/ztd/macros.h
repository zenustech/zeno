#pragma once


#include <zeno/common.h>


#define ZENO_FWD(x) std::forward<decltype(x)>(x)
#define ZENO_TYPEOF(x) std::remove_cvref_t<decltype(x)>
#define ZENO_F0(x, ...) [&] { return (__VA_ARGS__); }
#define ZENO_F1(x, ...) [&] (auto &&x) { return (__VA_ARGS__); }
#define ZENO_F2(x, y, ...) [&] (auto &&x, auto &&y) { return (__VA_ARGS__); }
#define ZENO_F3(x, y, z, ...) [&] (auto &&x, auto &&y, auto &&z) { return (__VA_ARGS__); }
