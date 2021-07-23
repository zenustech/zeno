#include "IRVisitor.h"
#include "Stmts.h"
#include <map>

namespace zfx {

struct GlobalMaxCounter : Visitor<GlobalMaxCounter> {
    using visit_stmt_types = std::tuple
        < AsmGlobalLoadStmt
        , AsmGlobalStoreStmt
        >;

    int nglobals = 0;

    void visit(AsmGlobalLoadStmt *stmt) {
        nglobals = std::max(nglobals, stmt->mem + 1);
    }

    void visit(AsmGlobalStoreStmt *stmt) {
        nglobals = std::max(nglobals, stmt->mem + 1);
    }
};

struct GlobalLocalize : Visitor<GlobalLocalize> {
    using visit_stmt_types = std::tuple
        < AsmLocalLoadStmt
        , AsmLocalStoreStmt
        , AsmGlobalLoadStmt
        , AsmGlobalStoreStmt
        , Statement
        >;

    std::unique_ptr<IR> ir = std::make_unique<IR>();

    int nglobals = 0;

    void visit(Statement *stmt) {
        ir->push_clone_back(stmt);
    }

    void visit(AsmLocalLoadStmt *stmt) {
        ir->emplace_back<AsmLocalLoadStmt>
            ( stmt->mem + nglobals
            , stmt->val
            );
    }

    void visit(AsmLocalStoreStmt *stmt) {
        ir->emplace_back<AsmLocalStoreStmt>
            ( stmt->mem + nglobals
            , stmt->val
            );
    }

    void visit(AsmGlobalLoadStmt *stmt) {
        ir->emplace_back<AsmLocalLoadStmt>
            ( stmt->mem
            , stmt->val
            );
    }

    void visit(AsmGlobalStoreStmt *stmt) {
        ir->emplace_back<AsmLocalStoreStmt>
            ( stmt->mem
            , stmt->val
            );
    }
};

void apply_global_localize(IR *ir) {
    GlobalMaxCounter counter;
    counter.apply(ir);
    GlobalLocalize visitor;
    visitor.nglobals = counter.nglobals;
    visitor.apply(ir);
    *ir = *visitor.ir;
}

}
