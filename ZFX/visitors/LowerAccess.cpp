#include "IRVisitor.h"
#include "Stmts.h"
#include <map>
#include <functional>

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

    std::map<int, std::function<void(int)>> loaders;

    int lookup_temp(int stmtid) {
        return stmtid;
    }

    int lookup(int stmtid, bool is_store = false) {
        int regid = stmtid;
        if (is_store)
            return regid;
        if (auto it = loaders.find(stmtid); it != loaders.end()) {
            it->second(regid);
        }
        return regid;
    }

    void visit(TempSymbolStmt *stmt) {
        loaders[stmt->id] = [this, stmt](int regid) {
            if (stmt->symids.size() != 1) {
                error("scalar expected on temp load, got %d-D vector",
                    stmt->symids.size());
            }
            stmt->symids[0] = lookup_temp(stmt->id);
            ir->emplace_back<AsmLocalLoadStmt>
                ( stmt->symids[0]
                , regid
                );
        };
    }

    void visit(SymbolStmt *stmt) {
        loaders[stmt->id] = [this, stmt](int regid) {
            if (stmt->symids.size() != 1) {
                error("scalar expected on load, got %d-D vector",
                    stmt->symids.size());
            }
            ir->emplace_back<AsmGlobalLoadStmt>
                ( stmt->symids[0]
                , regid
                );
        };
    }

    void visit(LiterialStmt *stmt) {
        loaders[stmt->id] = [this, stmt](int regid) {
            ir->emplace_back<AsmLoadConstStmt>
                ( regid
                , stmt->value
                );
        };
    }

    void visit(UnaryOpStmt *stmt) {
        auto dstreg = lookup(stmt->id, true);
        ir->emplace_back<AsmUnaryOpStmt>
            ( stmt->op
            , dstreg
            , lookup(stmt->src->id)
            );
    }

    void visit(BinaryOpStmt *stmt) {
        auto dstreg = lookup(stmt->id, true);
        ir->emplace_back<AsmBinaryOpStmt>
            ( stmt->op
            , dstreg
            , lookup(stmt->lhs->id)
            , lookup(stmt->rhs->id)
            );
    }

    void visit(AssignStmt *stmt) {
        if (auto dst = dynamic_cast<TempSymbolStmt *>(stmt->dst); dst) {
            if (dst->symids.size() != 1) {
                error("scalar expected on temp store, got %d-D vector",
                    dst->symids.size());
            }
            dst->symids[0] = lookup_temp(dst->id);
            ir->emplace_back<AsmLocalStoreStmt>
                ( dst->symids[0]
                , lookup(stmt->src->id)
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
                , lookup(stmt->src->id)
                );
            return;
        }
        auto dstreg = lookup(stmt->id, true);
        ir->emplace_back<AsmAssignStmt>
            ( dstreg
            , lookup(stmt->src->id)
            );
    }
};

std::unique_ptr<IR> apply_lower_access(IR *ir) {
    LowerAccess visitor;
    visitor.apply(ir);
    return std::move(visitor.ir);
}

}
