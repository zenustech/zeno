#pragma once

#include "Statement.h"

struct IR {
    std::vector<std::unique_ptr<Statement>> stmts;

    IR &operator=(IR const &ir) {
        stmts.clear();
        for (auto const &s: ir.stmts) {
            push_clone_back(s.get());
        }
        return *this;
    }

    template <class T, class ...Ts>
    T *emplace_back(Ts &&...ts) {
        auto id = stmts.size();
        auto stmt = std::make_unique<T>(id, std::forward<Ts>(ts)...);
        auto raw_ptr = stmt.get();
        stmts.push_back(std::move(stmt));
        return raw_ptr;
    }

    Statement *push_clone_back(Statement const *stmt) {
        auto id = stmts.size();
        auto new_stmt = stmt->clone(id);
        auto raw_ptr = new_stmt.get();
        stmts.push_back(std::move(new_stmt));
        return raw_ptr;
    }

    void print() const {
        for (auto const &s: stmts) {
            cout << s->print() << endl;
        }
    }
};
