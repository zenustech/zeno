#include "IRVisitor.h"
#include "Stmts.h"

namespace zfx {

struct LowerMath : Visitor<LowerMath> {
    using visit_stmt_types = std::tuple
        < TempSymbolStmt
        , SymbolStmt
        , Statement
        >;

    std::unique_ptr<IR> ir = std::make_unique<IR>();

    std::map<Statement *, std::vector<Statement *>> replaces;

    void visit(TempSymbolStmt *stmt) {
        auto &rep = replaces[stmt];
        rep.clear();
        for (int i = 0; i < stmt->dim; i++) {
            rep.push_back(ir->emplace_back<TempSymbolStmt>(
                stmt->tmpid, std::vector<int>{-1}));
        }
    }

    void visit(SymbolStmt *stmt) {
        auto &rep = replaces[stmt];
        rep.clear();
        for (int i = 0; i < stmt->dim; i++) {
            auto symid = stmt->symids[i];
            rep.push_back(ir->emplace_back<SymbolStmt>(
                std::vector<int>{symid}));
        }
    }

    void visit(Statement *stmt) {
        int repdim = -1;
        for (auto const &field: stmt->fields()) {
            if (auto it = replaces.find(stmt); it != replaces.end()) {
                auto newdim = it->second.size();
                if (repdim == -1) {
                    repdim = newdim;
                } else {
                    if (repdim == 1) {
                        repdim = newdim;
                    } else if (repdim != newdim) {
                        error("`%s`: vector dimension mismatch %d != %d",
                            typeid(stmt).name(), repdim, newdim);
                    }
                }
            }
        }
        if (repdim == -1) {
            ir->push_clone_back(stmt);
        } else {
            for (int i = 0; i < repdim; i++) {
                auto new_stmt = ir->push_clone_back(stmt);
                int f = 0;
                for (auto &field: stmt->fields()) {
                    if (auto it = replaces.find(stmt); it != replaces.end()) {
                        auto const &rep = replaces.at(stmt);
                        field.get() = rep[f % rep.size()];
                    }
                    f++;
                }
            }
        }
    }
};

std::unique_ptr<IR> apply_lower_math(IR *ir) {
    LowerMath visitor;
    visitor.apply(ir);
    return std::move(visitor.ir);
}

}
