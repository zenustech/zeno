#include "IRVisitor.h"
#include "Stmts.h"
#include <optional>

struct ResolveAssign : Visitor<ResolveAssign> {
    using visit_stmt_types = std::tuple
        < AssignStmt
        , Statement
        >;

    std::unique_ptr<IR> ir = std::make_unique<IR>();

    void visit(Statement *stmt) {
        ir->push_clone_back(stmt);
    }

    void visit(AssignStmt *stmt) {
        auto dstmem = dynamic_cast<SymbolStmt *>(stmt->dst);
        auto srcmem = dynamic_cast<SymbolStmt *>(stmt->src);
        if (dstmem) {
            ir->emplace_back<GlobalStoreStmt>(dstmem, stmt->src);
        } else if (srcmem) {
            ir->emplace_back<GlobalLoadStmt>(srcmem, stmt->dst);
        } else {
            ir->push_clone_back(stmt);
        }
    }
};

void apply_resolve_assign(IR *ir) {
    ResolveAssign visitor;
    visitor.apply(ir);
    *ir = *visitor.ir;
}
