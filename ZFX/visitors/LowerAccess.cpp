#include "IRVisitor.h"
#include "Stmts.h"

struct LowerAccess : Visitor<LowerAccess> {
    using visit_stmt_types = std::tuple
        < AssignStmt
        , Statement
        >;

    std::unique_ptr<IR> ir = std::make_unique<IR>();

    void visit(Statement *stmt) {
        ir->push_clone_back(stmt);
    }

    void visit(AssignStmt *stmt) {
        ir->push_clone_back(stmt);
    }
};

std::unique_ptr<IR> apply_lower_access(IR *ir) {
    LowerAccess visitor;
    visitor.apply(ir);
    return std::move(visitor.ir);
}
