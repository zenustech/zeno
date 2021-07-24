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

    std::map<int, std::vector<int>> writes;
    std::map<int, std::vector<int>> reads;

    void visit(Statement *stmt) {
        auto dst = stmt->dest_registers();
        auto src = stmt->source_registers();
        for (int r: dst) {
            writes[stmt->id].push_back(r);
        }
        for (int r: src) {
            reads[stmt->id].push_back(r);
        }
        ir->push_clone_back(stmt);
    }

    void visit(AsmLocalStoreStmt *stmt) {
        ir->push_clone_back(stmt);
    }

    void visit(AsmLocalLoadStmt *stmt) {
        ir->push_clone_back(stmt);
    }

    void visit(AsmGlobalStoreStmt *stmt) {
        ir->push_clone_back(stmt);
    }

    void visit(AsmGlobalLoadStmt *stmt) {
        ir->push_clone_back(stmt);
    }
};

std::unique_ptr<IR> apply_reassign_globals(IR *ir) {
    NoReluctantLoad visitor;
    visitor.apply(ir);
    return std::move(visitor.ir);
}

}
