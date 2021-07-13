#pragma once

#include "Statement.h"
#include <map>

namespace zfx {

struct IR {
    std::vector<std::unique_ptr<Statement>> stmts;
    std::map<Statement *, Statement *> cloned;

    IR() = default;
    IR(IR const &ir) { *this = ir; }

    void clear() {
        stmts.clear();
        cloned.clear();
    }

    size_t size() const {
        return stmts.size();
    }

    IR &operator=(IR const &ir) {
        clear();
        for (auto const &s: ir.stmts) {
            auto stmt = s.get();
            auto new_stmt = push_clone_back(stmt);
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

    struct Hole {
        int id;
        IR *ir;

        Hole() = default;
        Hole
            ( int id_
            , IR *ir_
            )
            : id(id_)
            , ir(ir_)
            {}

        template <class T, class ...Ts>
        T *place(Ts &&...ts) const {
            auto stmt = std::make_unique<T>(id, std::forward<Ts>(ts)...);
            auto raw_ptr = stmt.get();
            ir->stmts[id] = std::move(stmt);
            return raw_ptr;
        }
    };

    Hole make_hole_back() {
        int id = stmts.size();
        stmts.push_back(std::make_unique<EmptyStmt>(id));
        return Hole(id, this);
    }

    Statement *push_clone_back(Statement const *stmt_, bool find_only = false) {
        auto stmt = const_cast<Statement *>(stmt_);
        if (auto it = cloned.find(stmt); it != cloned.end()) {
            return it->second;
        }
        if (find_only)
            return emplace_back<EmptyStmt>();
        for (Statement *&field: stmt->fields()) {
            field = push_clone_back(field, true);
        }
        auto id = stmts.size();
        auto new_stmt = stmt->clone(id);
        auto raw_ptr = new_stmt.get();
        stmts.push_back(std::move(new_stmt));
        cloned[stmt] = raw_ptr;
        return raw_ptr;
    }

    void mark_replacement(Statement *old_stmt, Statement *new_stmt) {
        cloned[old_stmt] = new_stmt;
    }

    void print() const {
        for (auto const &s: stmts) {
            cout << s->print() << endl;
        }
    }
};

}
