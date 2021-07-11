#include "IRVisitor.h"
#include "Stmts.h"

namespace zfx {

struct LowerMath : Visitor<LowerMath> {
    using visit_stmt_types = std::tuple
        < SymbolStmt
        , Statement
        >;

    std::unique_ptr<IR> ir = std::make_unique<IR>();

    std::map<Statement *, std::vector<Statement *>> replaces;

    /*void visit(SymbolStmt *stmt) {
        if (stmt->is_temporary()) {
            auto &rep = replaces[stmt];
            for (int i = 0; i < stmt->dim; i++) {
                rep.push_back(ir->emplace_back<SymbolStmt>(
                    std::vector<int>{}));
            }
        } else {
            auto &rep = replaces[stmt];
            for (int i = 0; i < stmt->dim; i++) {
                auto symid = stmt->symids[i];
                rep.push_back(ir->emplace_back<SymbolStmt>(
                    std::vector<int>{symid}));
            }
        }
    }*/

    void visit(Statement *stmt) {
        ir->push_clone_back(stmt);
    }
};

std::unique_ptr<IR> apply_lower_math(IR *ir) {
    LowerMath visitor;
    visitor.apply(ir);
    return std::move(visitor.ir);
}

}
