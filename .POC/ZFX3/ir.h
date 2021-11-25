#pragma once

#include "statements.h"

struct IRBlock {
    std::vector<std::unique_ptr<Stmt>> stmts;

    template <class T, class ...Ts>
    T *emplace_back(Ts &&...ts) {
        auto ptr = std::make_unique<T>(std::forward<Ts>(ts)...);
        auto raw_ptr = ptr.get();
        stmts.push_back(std::move(ptr));
        return raw_ptr;
    }
};
