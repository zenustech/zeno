#include "IRVisitor.h"
#include "Stmts.h"
#include <functional>
#include <stack>
#include <map>

namespace zfx {

struct LowerControl : Visitor<LowerControl> {
    using visit_stmt_types = std::tuple
        < FrontendIfStmt
        , FrontendElseIfStmt
        , FrontendElseStmt
        , FrontendEndIfStmt
        , Statement
        >;

    std::unique_ptr<IR> ir = std::make_unique<IR>();

    std::stack<Statement *> sources;

    void visit(FrontendIfStmt *stmt) {
        auto cond = ir->push_clone_back(stmt->cond);
        auto source = ir->emplace_back<GotoIfStmt>(cond);
        sources.push(source);
    }

    void visit(FrontendElseIfStmt *stmt) {
        if (!sources.size()) {
            error("`elseif` without matching `if` at $%d", stmt->id);
        }
        auto &source = sources.top();
        if (!dynamic_cast<GotoIfStmt *>(source)) {
            error("no `elseif` allowed after `else` at $%d", stmt->id);
        }
        ir->emplace_back<GofromStmt>(source);
        auto cond = ir->push_clone_back(stmt->cond);
        source = ir->emplace_back<GotoIfStmt>(cond);
    }

    void visit(FrontendElseStmt *stmt) {
        if (!sources.size()) {
            error("`else` without matching `if` at $%d", stmt->id);
        }
        auto &source = sources.top();
        ir->emplace_back<GofromStmt>(source);
        source = ir->emplace_back<GotoStmt>();
    }

    void visit(FrontendEndIfStmt *stmt) {
        if (!sources.size()) {
            error("`endif` without matching `if` at $%d", stmt->id);
        }
        auto source = sources.top(); sources.pop();
        ir->emplace_back<GofromStmt>(source);
    }

    void visit(Statement *stmt) {
        ir->push_clone_back(stmt);
    }

    void finish() {
        if (sources.size()) {
            error("not terminated `if` block (remain %d levels)",
                    sources.size());
        }
    }
};

std::unique_ptr<IR> apply_lower_control(IR *ir) {
    LowerControl visitor;
    visitor.apply(ir);
    visitor.finish();
    return std::move(visitor.ir);
}

}
