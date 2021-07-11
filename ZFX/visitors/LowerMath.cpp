#include "IRVisitor.h"
#include "Stmts.h"

struct LowerMath : Visitor<LowerMath> {
    using visit_stmt_types = std::tuple
        < Statement
        >;

    std::unique_ptr<IR> ir = std::make_unique<IR>();

    void visit(Statement *stmt) {
        ir->push_clone_back(stmt);
    }
};

std::unique_ptr<IR> apply_lower_math(IR *ir) {
    LowerMath visitor;
    visitor.apply(ir);
    return std::move(visitor.ir);
}
