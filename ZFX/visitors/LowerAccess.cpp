#include "IRVisitor.h"
#include "Stmts.h"

namespace zfx {

struct LowerAccess : Visitor<LowerAccess> {
    using visit_stmt_types = std::tuple
        < AssignStmt
        , UnaryOpStmt
        , BinaryOpStmt
        , LiterialStmt
        , SymbolStmt
        , TempSymbolStmt
        , Statement
        >;

    std::unique_ptr<IR> ir = std::make_unique<IR>();

    void visit(Statement *stmt) {
        error("unexpected statement type `%s`", typeid(*stmt).name());
    }

    struct RegInfo {
        int last_used = -1;
        int curr_stmtid = -1;
    };

    void visit(TempSymbolStmt *stmt) {
        if (stmt->symids.size() != 1) {
            error("scalar expected on temp load, got %d-D vector",
                stmt->symids.size());
        }
        stmt->symids[0] = stmt->id;
        ir->emplace_back<AsmLocalLoadStmt>
            ( stmt->symids[0]
            , stmt->id
            );
    }

    void visit(SymbolStmt *stmt) {
        if (stmt->symids.size() != 1) {
            error("scalar expected on load, got %d-D vector",
                stmt->symids.size());
        }
        ir->emplace_back<AsmGlobalLoadStmt>
            ( stmt->symids[0]
            , stmt->id
            );
    }

    void visit(LiterialStmt *stmt) {
        ir->emplace_back<AsmLoadConstStmt>
            ( stmt->id
            , stmt->value
            );
    }

    void visit(UnaryOpStmt *stmt) {
        ir->emplace_back<AsmUnaryOpStmt>
            ( stmt->op
            , stmt->id
            , stmt->src->id
            );
    }

    void visit(BinaryOpStmt *stmt) {
        ir->emplace_back<AsmBinaryOpStmt>
            ( stmt->op
            , stmt->id
            , stmt->lhs->id
            , stmt->rhs->id
            );
    }

    void visit(AssignStmt *stmt) {
        if (auto dst = dynamic_cast<TempSymbolStmt *>(stmt->dst); dst) {
            if (dst->symids.size() != 1) {
                error("scalar expected on temp store, got %d-D vector",
                    dst->symids.size());
            }
            dst->symids[0] = dst->id;
            ir->emplace_back<AsmLocalStoreStmt>
                ( dst->symids[0]
                , stmt->src->id
                );
            return;
        }
        if (auto dst = dynamic_cast<SymbolStmt *>(stmt->dst); dst) {
            if (dst->symids.size() != 1) {
                error("scalar expected on store, got %d-D vector",
                    dst->symids.size());
            }
            ir->emplace_back<AsmGlobalStoreStmt>
                ( dst->symids[0]
                , stmt->src->id
                );
            return;
        }
        ir->emplace_back<AsmAssignStmt>
            ( stmt->id
            , stmt->src->id
            );
    }
};

std::unique_ptr<IR> apply_lower_access(IR *ir) {
    LowerAccess visitor;
    visitor.apply(ir);
    return std::move(visitor.ir);
}

}
