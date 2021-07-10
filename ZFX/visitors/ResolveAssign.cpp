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
        if (auto dstmem = dynamic_cast<SymbolStmt *>(stmt->dst); dstmem) {
            //ir->emplace_after<GlobalStoreStmt>(stmt, dstmem, stmt->src);
        }
    }
};

void apply_resolve_assign(IR *ir) {
    ResolveAssign visitor;
    visitor.apply(ir);
    *ir = *visitor.ir;
}
