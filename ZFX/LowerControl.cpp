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

    std::stack<std::function<void(Statement *)>> if_callbacks;

    void visit(FrontendIfStmt *stmt) {
        auto hole = ir->make_hole_back();
        auto cond = ir->push_clone_back(stmt->cond);
        if_callbacks.push([=] (Statement *target) {
            hole.place<GotoIfStmt>(cond, target);
        });
    }

    void visit(FrontendElseIfStmt *stmt) {
        if (!if_callbacks.size()) {
            error("`elseif` without matching `if` at $%d", stmt->id);
        }
        auto &callback = if_callbacks.top();
        callback(stmt);
        auto hole = ir->make_hole_back();
        auto cond = ir->push_clone_back(stmt->cond);
        callback = [=] (Statement *target) {
            hole.place<GotoIfStmt>(cond, target);
        };
    }

    void visit(FrontendElseStmt *stmt) {
        if (!if_callbacks.size()) {
            error("`else` without matching `if` at $%d", stmt->id);
        }
        auto &callback = if_callbacks.top();
        callback(stmt);
        auto hole = ir->make_hole_back();
        callback = [=] (Statement *target) {
            hole.place<GotoStmt>(target);
        };
    }

    void visit(FrontendEndIfStmt *stmt) {
        if (!if_callbacks.size()) {
            error("`endif` without matching `if` at $%d", stmt->id);
        }
        auto callback = if_callbacks.top(); if_callbacks.pop();
        callback(stmt);
    }

    void visit(Statement *stmt) {
        ir->push_clone_back(stmt);
    }

    void finish() {
        if (if_callbacks.size()) {
            error("not terminated `if` block (remain %d levels)",
                    if_callbacks.size());
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
