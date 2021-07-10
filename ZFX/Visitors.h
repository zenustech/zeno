#pragma once

#include "IRVisitor.h"
#include "Stmts.h"

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
