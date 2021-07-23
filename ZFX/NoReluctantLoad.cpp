#include "IRVisitor.h"
#include "Stmts.h"
#include <map>

namespace zfx {

struct NoReluctantLoad : Visitor<NoReluctantLoad> {
    using visit_stmt_types = std::tuple
        < AsmGlobalLoadStmt
        , AsmGlobalStoreStmt
        , AsmLocalLoadStmt
        , AsmLocalStoreStmt
        >;

    std::unique_ptr<IR> ir = std::make_unique<IR>();

    void local(int &mem) {
    }

    void global(int &mem) {
    }

    void visit(AsmLocalStoreStmt *stmt) {
        local(stmt->mem);
    }

    void visit(AsmLocalLoadStmt *stmt) {
        local(stmt->mem);
    }

    void visit(AsmGlobalStoreStmt *stmt) {
        global(stmt->mem);
    }

    void visit(AsmGlobalLoadStmt *stmt) {
        global(stmt->mem);
    }
};

std::unique_ptr<IR> apply_reassign_globals(IR *ir) {
    NoReluctantLoad visitor;
    visitor.apply(ir);
}

}
