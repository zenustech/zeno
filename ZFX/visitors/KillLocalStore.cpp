#include "IRVisitor.h"
#include "Stmts.h"
#include <map>
#include <functional>

namespace zfx {

struct MemInfo {
};

struct AnalysisLocal : Visitor<AnalysisLocal> {
    using visit_stmt_types = std::tuple
        < AsmLocalLoadStmt
        , AsmLocalStoreStmt
        >;

    std::map<int, MemInfo> usage;

    void visit(AsmLocalLoadStmt *stmt) {
    }

    void visit(AsmLocalStoreStmt *stmt) {
    }
};

struct KillLocalStore : Visitor<KillLocalStore> {
    using visit_stmt_types = std::tuple
        < AsmLocalLoadStmt
        , AsmLocalStoreStmt
        , Statement
        >;

    std::unique_ptr<IR> ir = std::make_unique<IR>();

    std::map<int, MemInfo> usage;

    void visit(Statement *stmt) {
        ir->push_clone_back(stmt);
    }
};

std::unique_ptr<IR> apply_kill_local_store(IR *ir) {
    AnalysisLocal analyser;
    analyser.apply(ir);
    KillLocalStore killer;
    killer.usage = analyser.usage;
    killer.apply(ir);
    return std::move(killer.ir);
}

}
