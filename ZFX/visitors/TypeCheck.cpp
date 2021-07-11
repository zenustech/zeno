#include "IRVisitor.h"
#include "Stmts.h"

namespace zfx {

struct TypeCheck : Visitor<TypeCheck> {
    using visit_stmt_types = std::tuple
        < SymbolStmt
        , LiterialStmt
        , AssignStmt
        , BinaryOpStmt
        , UnaryOpStmt
        , Statement
        >;

    std::unique_ptr<IR> ir = std::make_unique<IR>();

    void visit(SymbolStmt *stmt) {
        ir->push_clone_back(stmt);
    }

    void visit(Statement *stmt) {
        ir->push_clone_back(stmt);
    }
};

std::unique_ptr<IR> apply_type_check(IR *ir) {
    TypeCheck visitor;
    visitor.apply(ir);
    return std::move(visitor.ir);
}

}
