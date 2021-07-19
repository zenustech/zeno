#include "IRVisitor.h"
#include "Stmts.h"
#include <functional>
#include <map>

namespace zfx {

struct LowerControl : Visitor<LowerControl> {
    using visit_stmt_types = std::tuple
        < FrontendIfStmt
        , FrontendElseIfStmt
        , FrontendElseStmt
        , FrontendEndIfStmt
        , Statement
        >;

    std::unique_ptr<IR> ir = std::make_unique<IR>();

    void visit(FrontendIfStmt *stmt) {
        ir->push_clone_back(stmt);
    }

    void visit(FrontendEndIfStmt *stmt) {
        ir->push_clone_back(stmt);
    }

    void visit(Statement *stmt) {
        ir->push_clone_back(stmt);
    }
};

std::unique_ptr<IR> apply_lower_control(IR *ir) {
    LowerControl visitor;
    visitor.apply(ir);
    return std::move(visitor.ir);
}

}
