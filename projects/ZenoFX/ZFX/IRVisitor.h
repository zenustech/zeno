#pragma once

#include "Statement.h"
#include "IR.h"

namespace zfx {

struct IRVisitor {
    virtual void apply(Statement *stmt) = 0;

    void apply(IR *ir) {
        for (auto const &s: ir->stmts) {
            apply(s.get());
        }
    }
};

template <class T>
struct
    Visitor : IRVisitor {
    using IRVisitor::apply;

    virtual void apply(Statement *stmt) override {
        if (dynamic_cast<EmptyStmt *>(stmt))
            return;
        using visit_stmt_types = typename T::visit_stmt_types;
        static_for<0, std::tuple_size_v<visit_stmt_types>>
        ([this, stmt] (auto i) {
            using S = std::tuple_element_t<i, visit_stmt_types>;
            auto p = dynamic_cast<S *>(stmt);
            if (!p) return false;
            reinterpret_cast<T *>(this)->visit(p);
            return true;
        });
    }
};

}
