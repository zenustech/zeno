#include "IRVisitor.h"
#include "Stmts.h"
#include <sstream>
#include "StmHelper.h"

namespace zfx {

#define ERROR_IF(x) do { \
    if (x) { \
        error("`%s`", #x); \
    } \
} while (0)

struct MathFunctions : Visitor<MathFunctions> {
    using visit_stmt_types = std::tuple
        < FunctionCallStmt
        , Statement
        >;

    std::unique_ptr<IR> ir = std::make_unique<IR>();

    /*Stm emit_stm(std::string const &name, std::vector<Stm> const &args) {
        ERROR_IF(args.size() == 0);
        std::vector<Statement *> argptrs;
        for (auto const &arg: args) {
            argptrs.push_back(arg);
        }
        return {args[0].ir, emit_op(name, argptrs)};
    }*/

    Stm make_stm(Statement *stmt) {
        return {ir.get(), ir->push_clone_back(stmt)};
    }

    Stm stm_const(float x) {
        std::stringstream ss; ss << x;
        return {ir.get(), ir->emplace_back<LiterialStmt>(ss.str())};
    }

    Statement *emit_op(std::string const &name, std::vector<Statement *> const &args) {
        if (0) {

        } else if (name == "sin") {
            ERROR_IF(args.size() != 1);
            auto x = make_stm(args[0]);
            auto z = x;
            auto z2 = z * z;
            auto r = stm_const(1);
            auto t = z2 * stm_const(1.f / 6);
            r = r - t;
            t = z2 * stm_const(1.f / 20) * t;
            r = r + t;
            t = z2 * stm_const(1.f / 48) * t;
            r = r - t;
            t = z2 * stm_const(1.f / 72) * t;
            r = r + t;
            r = r * z;
            return r;

        } else {
            error("invalid function name `%s` (with %d args)", name.c_str(), args.size());
        }
    }

    void visit(FunctionCallStmt *stmt) {
        auto new_stmt = emit_op(stmt->name, stmt->args);
        ir->mark_replacement(stmt, new_stmt);
    }

    void visit(Statement *stmt) {
        ir->push_clone_back(stmt);
    }
};

std::unique_ptr<IR> apply_math_functions(IR *ir) {
    MathFunctions visitor;
    visitor.apply(ir);
    return std::move(visitor.ir);
}

}
