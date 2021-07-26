#include "IRVisitor.h"
#include "Stmts.h"
#include <functional>
#include <map>

namespace zfx {

struct MergeIdentical : Visitor<MergeIdentical> {
    using visit_stmt_types = std::tuple
        < AsmLocalStoreStmt
        , AsmLocalLoadStmt
        , AsmGlobalStoreStmt
        , AsmGlobalLoadStmt
        , Statement
        >;

    std::unique_ptr<IR> ir = std::make_unique<IR>();

    std::map<int, int> regs;
    std::map<int, int> locals;
    std::map<int, std::set<int>> dets;
    std::map<std::pair<std::string, std::set<int>>, int> revdets;
    std::map<std::tuple<int, int>, int> revdet;

    struct call_on_dtor : std::function<void()> {
        using std::function<void()>::function;
        call_on_dtor(call_on_dtor const &) = delete;
        ~call_on_dtor() { (*this)(); }
    };

    call_on_dtor generic_visit(Statement *stmt) {
        auto dst = stmt->dest_registers();
        auto src = stmt->source_registers();
        auto &det = dets[stmt->id];
        for (int r: src) {
            if (auto it = regs.find(r); it != regs.end()) {
                det.insert(it->second);
            } else {
                error("use of uninitialized register %d at $%d",
                        r, stmt->id);
            }
        }
        for (int r: dst) {
            regs[r] = stmt->id;
        }
        return [this, stmt]() {
            std::string type = typeid(*stmt).name();
            auto detkey = std::make_pair(type, dets.at(stmt->id));
            revdets[detkey] = stmt->id;
        };
    }

    void visit(Statement *stmt) {
        generic_visit(stmt);
    }

    void visit(AsmLocalLoadStmt *stmt) {
        auto &det = dets[stmt->id];
        if (auto it = locals.find(stmt->mem); it != locals.end()) {
            det.insert(-1 - it->second);
        } else {
            error("use of uninitialized local memory %d at $%d",
                    stmt->mem, stmt->id);
        }
        generic_visit((Statement *)stmt);
    }

    void visit(AsmLocalStoreStmt *stmt) {
        auto _ = generic_visit((Statement *)stmt);
        locals[stmt->mem] = stmt->id;
    }

    void visit(AsmGlobalLoadStmt *stmt) {
        generic_visit((Statement *)stmt);
    }

    void visit(AsmGlobalStoreStmt *stmt) {
        generic_visit((Statement *)stmt);
    }
};

std::unique_ptr<IR> apply_merge_identical(IR *ir) {
    MergeIdentical visitor;
    visitor.apply(ir);
    return std::move(visitor.ir);
}

}
