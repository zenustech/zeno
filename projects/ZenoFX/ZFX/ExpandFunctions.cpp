#include "IRVisitor.h"
#include "Stmts.h"
#include <sstream>
#include "StmHelper.h"
#include "iostream"
namespace zfx {

#define ERROR_IF(x) do { \
    if (x) { \
        error("`%s`", #x); \
    } \
} while (0)

struct ExpandFunctions : Visitor<ExpandFunctions> {
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

    Statement *emit_op(std::string const &name, std::vector<Statement *> const &args) {

        if (name.substr(0, 3) == "vec" && name.size() == 4 && isdigit(name[3])) {
            int dim = name[3] - '0';
            int argdim = 0;
            for (auto const &arg: args) {
                std::cout<<arg->dim<<std::endl;
                argdim += arg->dim;
            }
            if (argdim != 1 && argdim != dim) {
                error("`%s` expect 1 or %d components in arguments, got %d",
                    name.c_str(), dim, argdim);
            }
            std::vector<Statement *> retargs;
            for (auto const &arg: args) {
                auto retarg = ir->push_clone_back(arg);
                for (int d = 0; d < arg->dim; d++) {
                    retargs.push_back(ir->emplace_back<VectorSwizzleStmt>(
                        std::vector<int>{d}, retarg));
                }
            }
            return ir->emplace_back<VectorComposeStmt>(dim, retargs);

        } else if (name == "mix") {
            ERROR_IF(args.size() != 3);
            auto x = make_stm(args[0]);
            auto y = make_stm(args[1]);
            auto a = make_stm(args[2]);
            return (y - x) * a + x;

        } else if (name == "clamp") {
            ERROR_IF(args.size() != 3);
            auto x = make_stm(args[0]);
            auto a = make_stm(args[1]);
            auto b = make_stm(args[2]);
            return stm("min", b, stm("max", a, x));

        } else if (name == "length") {
            ERROR_IF(args.size() != 1);
            auto x = make_stm(args[0]);
            return stm("sqrt", stm_sqrlength(x));

        } else if (name == "normalize") {
            ERROR_IF(args.size() != 1);
            auto x = make_stm(args[0]);
            return x / stm("sqrt", stm_sqrlength(x));

        } else if (name == "distance") {
            ERROR_IF(args.size() != 2);
            auto a = make_stm(args[0]);
            auto b = make_stm(args[1]);
            if (a->dim != b->dim) {
                error("dimension mismatch for function `%s`: %d != %d",
                    name.c_str(), a->dim, b->dim);
            }
            return stm("sqrt", stm_sqrlength(a - b));

        } else if (name == "dot") {
            ERROR_IF(args.size() != 2);
            auto x = make_stm(args[0]);
            auto y = make_stm(args[1]);
            ERROR_IF((!x->dim || !y->dim) && "dot");
            if (x->dim != y->dim) {
                error("dimension mismatch for function `%s`: %d != %d",
                    name.c_str(), x->dim, y->dim);
            }
            return stm_dot(x, y);

        } else if (name == "cross") {
            ERROR_IF(args.size() != 2);
            auto x = make_stm(args[0]);
            auto y = make_stm(args[1]);
            if (x->dim != 3 || y->dim != 3) {
                error("`cross` requires two 3-D vectors, got %d-D and %d-D",
                    x->dim, y->dim);
            }
            //return stm_cross(x, y);
            std::vector<Statement *> retargs;
            retargs.push_back(ir->emplace_back<VectorSwizzleStmt>(
                std::vector<int>{0}, stm_crossx(x,y)));
            retargs.push_back(ir->emplace_back<VectorSwizzleStmt>(
                std::vector<int>{0}, stm_crossy(x,y)));
            retargs.push_back(ir->emplace_back<VectorSwizzleStmt>(
                std::vector<int>{0}, stm_crossz(x,y)));
            

            return ir->emplace_back<VectorComposeStmt>(3, retargs);

        } else if (name == "all") {
            ERROR_IF(args.size() != 1);
            auto x = make_stm(args[0]);
            ERROR_IF(!x->dim && "all");
            auto ret = x[0];
            for (int i = 1; i < x->dim; i++) {
                ret = ret & x[i];
            }
            return ret;

        } else if (name == "any") {
            ERROR_IF(args.size() != 1);
            auto x = make_stm(args[0]);
            ERROR_IF(!x->dim && "all");
            auto ret = x[0];
            for (int i = 1; i < x->dim; i++) {
                ret = ret | x[i];
            }
            return ret;

        } else {
            std::vector<Stm> argstms;
            for (auto const &x: args) {
                argstms.emplace_back(make_stm(x));
            }
            return stm_func(name, argstms);
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

std::unique_ptr<IR> apply_expand_functions(IR *ir) {
    ExpandFunctions visitor;
    visitor.apply(ir);
    return std::move(visitor.ir);
}

}
