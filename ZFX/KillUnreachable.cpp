#include "IRVisitor.h"
#include "Stmts.h"
#include <stack>
#include <map>

namespace zfx {

struct GatherReachable : Visitor<GatherReachable> {
    using visit_stmt_types = std::tuple
        < AsmLocalStoreStmt
        , AsmLocalLoadStmt
        , AsmGlobalStoreStmt
        , AsmGlobalLoadStmt
        , Statement
        >;

    std::map<int, int> regs;
    std::map<int, int> locals;
    std::map<int, int> globals;
    std::map<int, std::set<int>> deps;
    std::set<int> reached;

    void visit(Statement *stmt) {
        auto dst = stmt->dest_registers();
        auto src = stmt->source_registers();
        for (int r: src) {
            auto stmtid = regs[r];
            deps[stmt->id].insert(stmtid);
        }
        for (int r: dst) {
            regs[r] = stmt->id;
        }
    }

    void visit(AsmLocalLoadStmt *stmt) {
        visit((Statement *)stmt);
        auto stmtid = locals[stmt->mem];
        deps[stmt->id].insert(stmtid);
    }

    void visit(AsmLocalStoreStmt *stmt) {
        visit((Statement *)stmt);
        locals[stmt->mem] = stmt->id;
    }

    void visit(AsmGlobalLoadStmt *stmt) {
        visit((Statement *)stmt);
        auto stmtid = globals[stmt->mem];
        deps[stmt->id].insert(stmtid);
    }

    void visit(AsmGlobalStoreStmt *stmt) {
        visit((Statement *)stmt);
        globals[stmt->mem] = stmt->id;
        reached.insert(stmt->id);
    }

    void finalize() {
        std::stack<int> stack;
        std::set<int> visited;
        for (auto stmtid: reached) {
            stack.push(stmtid);
        }
        while (stack.size() != 0) {
            auto id = stack.top(); stack.pop();
            visited.insert(id);
            for (auto srcid: deps[id]) {
                reached.insert(srcid);
                if (visited.find(srcid) == visited.end())
                    stack.push(srcid);
            }
        }
    }
};

struct KillUnreachable : Visitor<KillUnreachable> {
    using visit_stmt_types = std::tuple
        < Statement
        >;

    std::set<int> reached;

    std::unique_ptr<IR> ir = std::make_unique<IR>();

    void visit(Statement *stmt) {
        if (!stmt->is_control_stmt()) {
            if (reached.find(stmt->id) == reached.end()) {
                return;
            }
        }
        ir->push_clone_back(stmt);
    }
};

std::unique_ptr<IR> apply_kill_unreachable(IR *ir) {
    GatherReachable gather;
    gather.apply(ir);
    gather.finalize();
    KillUnreachable killer;
    killer.reached = gather.reached;
    killer.apply(ir);
    return std::move(killer.ir);
}

}
