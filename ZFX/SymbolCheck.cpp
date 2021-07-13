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
        if (auto dst = dynamic_cast<TempSymbolStmt *>(stmt->dst); dst) {
            defined.insert(dst->id);
        }
    }

    void visit(Statement *stmt) {
        for (auto const &field: stmt->fields()) {
            if (auto fie = dynamic_cast<TempSymbolStmt *>(field.get()); fie) {
                if (defined.find(fie->id) == defined.end()) {
                    error("undefined symbol $%d referenced by $%d",
                        fie->id, stmt->id);
                }
            }
        }
    }
};

void apply_symbol_check(IR *ir) {
    SymbolCheck visitor;
    visitor.apply(ir);
}

}
