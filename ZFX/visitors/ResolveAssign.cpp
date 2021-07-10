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
        if (srcmem && dstmem) {
            auto tmp = ir->emplace_back<GlobalLoadStmt>(srcmem);
            ir->emplace_back<GlobalStoreStmt>(dstmem, tmp);
        } else if (dstmem) {
            ir->emplace_back<GlobalStoreStmt>(dstmem, stmt->src);
        } else if (srcmem) {
            auto tmp = ir->emplace_back<GlobalLoadStmt>(srcmem);
            ir->emplace_back<AssignStmt>(stmt->dst, tmp);
        } else {
            ir->emplace_back<AssignStmt>(stmt->dst, stmt->src);
        }
    }
};

void apply_resolve_assign(IR *ir) {
    ResolveAssign visitor;
    visitor.apply(ir);
    *ir = *visitor.ir;
}
