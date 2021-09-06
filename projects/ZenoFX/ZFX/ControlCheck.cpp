#include "IRVisitor.h"
#include "Stmts.h"
#include <functional>
#include <stack>
#include <map>

namespace zfx {

struct ControlCheck : Visitor<ControlCheck> {
    using visit_stmt_types = std::tuple
        < FrontendIfStmt
        , FrontendElseIfStmt
        , FrontendElseStmt
        , FrontendEndIfStmt
        , Statement
        >;

    std::stack<bool> sources;

    void visit(FrontendIfStmt *stmt) {
        sources.push(true);
    }

    void visit(FrontendElseIfStmt *stmt) {
        if (!sources.size()) {
            error("`elseif` without matching `if` at $%d", stmt->id);
        }
        auto source = sources.top();
        if (!source) {
            error("no `elseif` allowed after `else` at $%d", stmt->id);
        }
    }

    void visit(FrontendElseStmt *stmt) {
        if (!sources.size()) {
            error("`else` without matching `if` at $%d", stmt->id);
        }
        auto &source = sources.top();
        if (!source) {
            error("got double `else` at $%d, please `endif`", stmt->id);
        }
        source = false;
    }

    void visit(FrontendEndIfStmt *stmt) {
        if (!sources.size()) {
            error("`endif` without matching `if` at $%d", stmt->id);
        }
        auto source = sources.top(); sources.pop();
    }

    void visit(Statement *stmt) {
    }

    void finish() {
        if (sources.size()) {
            error("not terminated `if` block (remain %d levels)",
                    sources.size());
        }
    }
};

void apply_control_check(IR *ir) {
    ControlCheck visitor;
    visitor.apply(ir);
    visitor.finish();
}

}
