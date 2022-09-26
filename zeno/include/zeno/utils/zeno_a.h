#pragma once

#include <zeno/zeno.h>

#define ZENO_A1(T_, name_, ...) auto name_ = get_input<T_>(#name_);
#define ZENO_B1(T_, name_) set_output(#name_, std::move(name_));
#define ZENO_A2(T_, name_, ...) auto name_ = get_input2<T_>(#name_);
#define ZENO_B2(T_, name_) set_output2(#name_, std::move(name_));

#define ZENO_A(node_) struct node_ : ::zeno::INode { virtual void apply() override {
#define ZENO_B(...) __VA_ARGS__
#define ZENO_C(...) } };

#define _ZENO_A1(T_, name_, ...) {#T_, #name_, __VA_ARGS__},
#define _ZENO_B1(T_, name_) {#T_, #name_},
#define _ZENO_A2(T_, name_, ...) {#T_, #name_, __VA_ARGS__},
#define _ZENO_B2(T_, name_) {#T_, #name_},

#define _ZENO_A(node_) ZENO_DEFNODE(node_)({{
#define _ZENO_B(...) }, {
#define _ZENO_C(cate_) }, {}, {#cate_}});

#define _ZENO_D1(x, ...) _##x __VA_OPT__(_ZENO_D1(__VA_ARGS__))
#define _ZENO_D0(x, ...) x __VA_OPT__(_ZENO_D0(__VA_ARGS__))
#define ZENO_D(...) _ZENO_D0(__VA_ARGS__) _ZENO_D1(__VA_ARGS__)
