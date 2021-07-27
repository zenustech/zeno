#include "IRVisitor.h"
#include "Stmts.h"
#include <sstream>

namespace zfx {

#define ERROR_IF(x) do { \
    if (x) { \
        error("%s:%d: `%s`", __FILE__, __LINE__, #x); \
    } \
} while (0)

struct LowerMath : Visitor<LowerMath> {
    using visit_stmt_types = std::tuple
        < SymbolStmt
        , TempSymbolStmt
        , ParamSymbolStmt
        , LiterialStmt
        , UnaryOpStmt
        , BinaryOpStmt
        , TernaryOpStmt
        , FunctionCallStmt
        , VectorSwizzleStmt
        , VectorComposeStmt
        , AssignStmt
        , Statement
        >;

    std::unique_ptr<IR> ir = std::make_unique<IR>();

    std::map<Statement *, std::vector<Statement *>> replaces;

    Statement *replace(Statement *stmt, int i) {
        auto const &rep = replaces.at(stmt);
        return rep[i % rep.size()];
    }

    void visit(TempSymbolStmt *stmt) {
        auto &rep = replaces[stmt];
        rep.clear();
        ERROR_IF(stmt->dim == 0);
        for (int i = 0; i < stmt->dim; i++) {
            rep.push_back(ir->emplace_back<TempSymbolStmt>(
                stmt->tmpid, std::vector<int>{-1}));
        }
    }

    void visit(ParamSymbolStmt *stmt) {
        auto &rep = replaces[stmt];
        rep.clear();
        ERROR_IF(stmt->dim == 0);
        for (int i = 0; i < stmt->dim; i++) {
            auto symid = stmt->symids[i];
            rep.push_back(ir->emplace_back<ParamSymbolStmt>(
                std::vector<int>{symid}));
        }
    }

    void visit(SymbolStmt *stmt) {
        auto &rep = replaces[stmt];
        rep.clear();
        ERROR_IF(stmt->dim == 0);
        for (int i = 0; i < stmt->dim; i++) {
            auto symid = stmt->symids[i];
            rep.push_back(ir->emplace_back<SymbolStmt>(
                std::vector<int>{symid}));
        }
    }

    void visit(UnaryOpStmt *stmt) {
        auto &rep = replaces[stmt];
        rep.clear();
        ERROR_IF(stmt->dim == 0);
        for (int i = 0; i < stmt->dim; i++) {
            rep.push_back(ir->emplace_back<UnaryOpStmt>(
                stmt->op, replace(stmt->src, i)));
        }
    }

    void visit(BinaryOpStmt *stmt) {
        auto &rep = replaces[stmt];
        rep.clear();
        ERROR_IF(stmt->dim == 0);
        for (int i = 0; i < stmt->dim; i++) {
            rep.push_back(ir->emplace_back<BinaryOpStmt>(
                stmt->op, replace(stmt->lhs, i), replace(stmt->rhs, i)));
        }
    }

    void visit(TernaryOpStmt *stmt) {
        auto &rep = replaces[stmt];
        rep.clear();
        ERROR_IF(stmt->dim == 0);
        for (int i = 0; i < stmt->dim; i++) {
            rep.push_back(ir->emplace_back<TernaryOpStmt>(
                replace(stmt->cond, i), replace(stmt->lhs, i),
                replace(stmt->rhs, i)));
        }
    }

    void visit(FunctionCallStmt *stmt) {
        auto &rep = replaces[stmt];
        rep.clear();
        ERROR_IF(stmt->dim == 0);
        for (int i = 0; i < stmt->dim; i++) {
            std::vector<Statement *> args;
            for (auto const &arg: stmt->args) {
                args.push_back(replace(arg, i));
            }
            rep.push_back(ir->emplace_back<FunctionCallStmt>(
                stmt->name, args));
        }
    }

    void visit(VectorComposeStmt *stmt) {
        auto &rep = replaces[stmt];
        rep.clear();
        ERROR_IF(stmt->dim == 0);
        ERROR_IF(stmt->args.size() == 0);
        for (int i = 0; i < stmt->dim; i++) {
            auto arg = stmt->args[i % stmt->args.size()];
            ERROR_IF(arg->dim != 1);
            rep.push_back(replace(arg, 0));
        }
    }

    void visit(VectorSwizzleStmt *stmt) {
        auto &rep = replaces[stmt];
        rep.clear();
        ERROR_IF(stmt->dim == 0);
        ERROR_IF(stmt->dim != stmt->swizzles.size());
        for (int i = 0; i < stmt->dim; i++) {
            auto s = stmt->swizzles[i];
            rep.push_back(replace(stmt->src, s));
        }
    }

    void visit(AssignStmt *stmt) {
        auto &rep = replaces[stmt];
        rep.clear();
        ERROR_IF(stmt->dim == 0);
        for (int i = 0; i < stmt->dim; i++) {
            rep.push_back(ir->emplace_back<AssignStmt>(
                replace(stmt->dst, i), replace(stmt->src, i)));
        }
    }

    void visit(LiterialStmt *stmt) {
        auto &rep = replaces[stmt];
        rep.clear();
        rep.push_back(ir->push_clone_back(stmt));
    }

    void visit(Statement *stmt) {
        if (stmt->is_control_stmt()) {
            ir->push_clone_back(stmt);
            return;
        }
        error("unexpected statement type to LowerMath: `%s`",
            typeid(*stmt).name());
    }
};

std::unique_ptr<IR> apply_lower_math(IR *ir) {
    LowerMath visitor;
    visitor.apply(ir);
    return std::move(visitor.ir);
}

}
