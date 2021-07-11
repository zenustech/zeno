#include "IRVisitor.h"
#include "Stmts.h"
#include <map>
#include <functional>

namespace zfx {

struct GatherLocalLoad : Visitor<GatherLocalLoad> {
    using visit_stmt_types = std::tuple
        < AsmLocalLoadStmt
        >;

    std::map<int, int> usage;

    void visit(AsmLocalLoadStmt *stmt) {
        usage[stmt->mem] = std::max(usage[stmt->mem], stmt->id);
    }
};

struct KillLocalStore : Visitor<KillLocalStore> {
    using visit_stmt_types = std::tuple
        < AsmLocalStoreStmt
        , Statement
        >;

    std::unique_ptr<IR> ir = std::make_unique<IR>();

    std::map<int, int> usage;

    void visit(AsmLocalStoreStmt *stmt) {
        auto it = usage.find(stmt->mem);
        if (it == usage.end())
            return;
        if (stmt->id > it->second)
            return;
        ir->push_clone_back(stmt);
    }

    void visit(Statement *stmt) {
        ir->push_clone_back(stmt);
    }
};

std::unique_ptr<IR> apply_kill_local_store(IR *ir) {
    GatherLocalLoad gather;
    gather.apply(ir);
    KillLocalStore visitor;
    visitor.usage = gather.usage;
    visitor.apply(ir);
    return std::move(visitor.ir);
}

}
