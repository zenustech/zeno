#include "IRVisitor.h"
#include "Stmts.h"
#include <map>
#include <functional>

struct LowerAccess : Visitor<LowerAccess> {
    using visit_stmt_types = std::tuple
        < AssignStmt
        , UnaryOpStmt
        , BinaryOpStmt
        , LiterialStmt
        , SymbolStmt
        , Statement
        >;

    std::unique_ptr<IR> ir = std::make_unique<IR>();

    void visit(Statement *stmt) {
        ir->push_clone_back(stmt);
    }

    struct RegInfo {
        int last_used = -1;
        int curr_stmtid = -1;
    };

    int now() const {
        return ir->size();
    }

    std::vector<RegInfo> regs{2};
    std::vector<int> locals;
    std::map<int, int> locals_lut;
    std::map<int, std::function<void(int)>> loaders;
    std::map<int, std::function<void()>> savers;

    int temp_save(int regid, int stmtid) {
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
        auto hole = ir->make_hole_back();
        int old_stmtid = regs[regid].curr_stmtid;
        savers[old_stmtid] = [this, hole, old_stmtid, regid]() {
            int memid = temp_save(regid, old_stmtid);
            hole.place<AsmLocalStoreStmt>(memid, regid);
        };
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
        if (is_store) {
            return regid;
        }

        // think: what's the order for consts? locals_lut or loaders first?
        if (auto it = locals_lut.find(stmtid); it != locals_lut.end()) {
            int memid = it->second;
            ir->emplace_back<AsmLocalLoadStmt>(memid, regid);
            if (auto it = savers.find(stmtid); it != savers.end()) {
                it->second();
            }
            return regid;
        }

        if (auto it = loaders.find(stmtid); it != loaders.end()) {
            it->second(regid);
        }
        return regid;
    }

    std::map<std::string, int> symbols;
    int symid = 0;

    int lookup_symbol_id(std::string const &name) {
        auto it = symbols.find(name);
        if (it != symbols.end()) {
            return it->second;
        }
        auto id = symid++;
        symbols[name] = id;
        return id;
    }

    void visit(SymbolStmt *stmt) {
        loaders[stmt->id] = [this, stmt](int regid) {
            ir->emplace_back<AsmGlobalLoadStmt>
                ( stmt->symid
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
        ir->emplace_back<AsmUnaryOpStmt>
            ( stmt->op
            , lookup(stmt->id, true)
            , lookup(stmt->src->id)
            );
    }

    void visit(BinaryOpStmt *stmt) {
        ir->emplace_back<AsmBinaryOpStmt>
            ( stmt->op
            , lookup(stmt->id, true)
            , lookup(stmt->lhs->id)
            , lookup(stmt->rhs->id)
            );
    }

    void visit(AssignStmt *stmt) {
        if (auto dst = dynamic_cast<SymbolStmt *>(stmt->dst); dst) {
            ir->emplace_back<AsmGlobalStoreStmt>
                ( dst->symid
                , lookup(stmt->src->id)
                );
            return;
        }
        ir->emplace_back<AsmAssignStmt>
            ( lookup(stmt->dst->id, true)
            , lookup(stmt->src->id)
            );
    }
};

std::unique_ptr<IR> apply_lower_access(IR *ir) {
    LowerAccess visitor;
    visitor.apply(ir);
    return std::move(visitor.ir);
}
