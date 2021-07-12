#include "IRVisitor.h"
#include "Stmts.h"
#include <map>
#include <functional>

namespace zfx {

struct GatherLocalLoad : Visitor<GatherLocalLoad> {
    using visit_stmt_types = std::tuple
        < AsmLocalLoadStmt
        >;

    std::map<int, int> last_load;

    void visit(AsmLocalLoadStmt *stmt) {
        last_load[stmt->mem] = std::max(last_load[stmt->mem], stmt->id);
    }
};

struct KillLocalStore : Visitor<KillLocalStore> {
    using visit_stmt_types = std::tuple
        < AsmLocalLoadStmt
        , AsmLocalStoreStmt
        , Statement
        >;

    std::unique_ptr<IR> ir = std::make_unique<IR>();

    struct StoreRAII {
        AsmLocalStoreStmt *stmt = nullptr;
        IR::Hole hole;
        bool active = false;

        StoreRAII() = default;
        StoreRAII(StoreRAII &&) = default;
        StoreRAII(StoreRAII const &) = default;
        StoreRAII &operator=(StoreRAII const &) = default;

        StoreRAII(AsmLocalStoreStmt *stmt, IR::Hole const &hole)
            : stmt(stmt), hole(hole), active(true) {}

        ~StoreRAII() {
            if (active) {
                hole.place<AsmLocalStoreStmt>(stmt->mem, stmt->val);
            }
        }
    };

    std::map<int, int> last_load;
    std::unique_ptr<StoreRAII> storer;

    void visit(AsmLocalLoadStmt *stmt) {
        if (storer && storer->stmt->mem == stmt->mem) {
            if (last_load.at(stmt->mem) == stmt->id)
                storer->active = false;
            if (stmt->val != storer->stmt->val)
                ir->emplace_back<AsmAssignStmt>(
                    stmt->val, storer->stmt->val);
            storer = nullptr;

        } else {
            visit((Statement *)stmt);
        }
    }

    void visit(AsmLocalStoreStmt *stmt) {
        auto it = last_load.find(stmt->mem);
        if (it == last_load.end())
            return;
        if (stmt->id > it->second)
            return;
        auto hole = ir->make_hole_back();
        storer = std::make_unique<StoreRAII>(stmt, hole);
    }

    void visit(Statement *stmt) {
        storer = nullptr;
        ir->push_clone_back(stmt);
    }
};

std::unique_ptr<IR> apply_kill_local_store(IR *ir) {
    GatherLocalLoad gather;
    gather.apply(ir);
    KillLocalStore visitor;
    visitor.last_load = gather.last_load;
    visitor.apply(ir);
    visitor.storer = nullptr;
    return std::move(visitor.ir);
}

}
