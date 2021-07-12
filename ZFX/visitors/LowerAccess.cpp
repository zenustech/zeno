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
        , FunctionCallStmt
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

    int now() const {
        return ir->size();
    }

    std::vector<RegInfo> regs{8};  // seems need >= 3 to work
    std::vector<int> locals;
    std::map<int, int> locals_lut;
    std::map<int, std::function<void(int)>> loaders;

    int lookup_temp(int stmtid) {
        if (auto it = locals_lut.find(stmtid); it != locals_lut.end()) {
            return it->second;
        }
        int memid = locals.size();
        locals_lut[stmtid] = memid;
        locals.push_back(stmtid);
        return memid;
    }

    int temp_save_location(int regid, int stmtid) {
        for (int i = 0; i < locals.size(); i++) {
            if (locals[i] == -1) {
                locals_lut[stmtid] = i;
                locals[i] = stmtid;
                return i;
            }
        }
        int memid = locals.size();
        locals_lut[stmtid] = memid;
        locals.push_back(stmtid);
        return memid;
    }

    int alloc_register() {
        for (int i = 0; i < regs.size(); i++) {
            if (regs[i].curr_stmtid == -1)
                return i;
        }
        int regid = 0;
        for (int i = 1; i < regs.size(); i++) {
            if (regs[i].last_used < regs[regid].last_used)
                regid = i;
        }
        int old_stmtid = regs[regid].curr_stmtid;
        int memid = temp_save_location(regid, old_stmtid);
        ir->emplace_back<AsmLocalStoreStmt>(memid, regid);
        regs[regid].curr_stmtid = -1;
        return regid;
    }

    int lookup(int stmtid, bool is_store = false) {
        for (int i = 0; i < regs.size(); i++) {
            if (regs[i].curr_stmtid == stmtid) {
                regs[i].last_used = now();
                return i;
            }
        }
        auto regid = alloc_register();
        regs[regid].curr_stmtid = stmtid;
        regs[regid].last_used = now();
        if (is_store) {
            return regid;
        }

        if (auto it = locals_lut.find(stmtid); it != locals_lut.end()) {
            int memid = it->second;
            ir->emplace_back<AsmLocalLoadStmt>(memid, regid);
            return regid;
        }

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

    void visit(FunctionCallStmt *stmt) {
        if (stmt->args.size() == 1 && contains({"sqrt"}, stmt->name)) {
            auto dstreg = lookup(stmt->id, true);
            ir->emplace_back<AsmUnaryOpStmt>
                ( stmt->name
                , dstreg
                , lookup(stmt->args[0]->id)
                );
        } else {
            error("invalid function call to `%s` with %d argument(s)",
                stmt->name.c_str(), stmt->args.size());
        }
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
