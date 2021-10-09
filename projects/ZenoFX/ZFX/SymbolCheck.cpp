#include "IRVisitor.h"
#include "Stmts.h"

namespace zfx {

struct SymbolCheck : Visitor<SymbolCheck> {
    using visit_stmt_types = std::tuple
        < TempSymbolStmt
        , AssignStmt
        , Statement
        >;

    std::set<int> defined;

    void visit(TempSymbolStmt *stmt) {
    }

    void visit(AssignStmt *stmt) {
        check(stmt->src);
        if (auto dst = dynamic_cast<TempSymbolStmt *>(stmt->dst); dst) {
            defined.insert(dst->id);
        }
    }

    void check(Statement *stmt) {
        if (auto src = dynamic_cast<TempSymbolStmt *>(stmt); src) {
            if (defined.find(src->id) == defined.end()) {
                error("undefined symbol at $%d", src->id);
            }
        }
    }

    void visit(Statement *stmt) {
        for (auto const &field: stmt->fields()) {
            check(field.get());
        }
    }
};

void apply_symbol_check(IR *ir) {
    SymbolCheck visitor;
    visitor.apply(ir);
}

}
