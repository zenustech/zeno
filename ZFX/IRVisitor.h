#pragma once

#include "Statement.h"

struct IRVisitor {
    virtual void visit(Statement *stmt) = 0;

    void do_visit(IR *ir) {
        for (auto const &s: ir->stmts) {
            visit(s.get());
        }
    }
};

template <class T>
struct Visitor : IRVisitor {
    virtual void visit(Statement *stmt) override {
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

struct DemoVisitor : Visitor<DemoVisitor> {
    using visit_stmt_types = std::tuple
        < SymbolStmt
        , LiterialStmt
        >;

    void visit(SymbolStmt *stmt) {
        printf("DemoVisitor got symbol: [%s]\n", stmt->name.c_str());
    }

    void visit(LiterialStmt *stmt) {
        printf("DemoVisitor got literial: [%s]\n", stmt->name.c_str());
    }
};

