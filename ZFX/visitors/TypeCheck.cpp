#include "IRVisitor.h"
#include "Stmts.h"

namespace zfx {

#define ERROR_IF(x) do { \
    if (x) { \
        error("`%s`", #x); \
    } \
} while (0)

struct TypeCheck : Visitor<TypeCheck> {
    using visit_stmt_types = std::tuple
        < AssignStmt
        , LiterialStmt
        , SymbolStmt
        , VectorSwizzleStmt
        , FunctionCallStmt
        , BinaryOpStmt
        , UnaryOpStmt
        , Statement
        >;

    std::unique_ptr<IR> ir = std::make_unique<IR>();

    void visit(Statement *stmt) {
        if (stmt->dim == 0) {
            if (dynamic_cast<TempSymbolStmt *>(stmt))
                error("undefined variable $%d (used before assignment)", stmt->id);
            else
                error("statement $%d has unclear type", stmt->id);
        }
    }

    void visit(SymbolStmt *stmt) {
        stmt->dim = stmt->symids.size();
        visit((Statement *)stmt);
    }

    void visit(LiterialStmt *stmt) {
        stmt->dim = 1;
        visit((Statement *)stmt);
    }

    void visit(UnaryOpStmt *stmt) {
        stmt->dim = stmt->src->dim;
        visit((Statement *)stmt);
    }

    void visit(BinaryOpStmt *stmt) {
        if (stmt->lhs->dim > 1 && stmt->rhs->dim > 1
            && stmt->lhs->dim != stmt->rhs->dim) {
            error("dimension mismatch in binary `%s`: %d != %d",
                stmt->op.c_str(), stmt->lhs->dim, stmt->rhs->dim);
        }
        ERROR_IF(stmt->lhs->dim == 0);
        ERROR_IF(stmt->rhs->dim == 0);
        stmt->dim = std::max(stmt->lhs->dim, stmt->rhs->dim);
        visit((Statement *)stmt);
    }

    void visit(FunctionCallStmt *stmt) {
        stmt->dim = 0;

        if (0) {

        } else if (contains({"sqrt", "sin", "cos", "tan", "asin", "acos",
            "atan", "exp", "log", "rsqrt", "floor", "ceil", "normalize"}, stmt->name)) {
            if (stmt->args.size() != 3) {
                error("function `%s` takes exactly 3 arguments", stmt->name.c_str());
            }

        } else if (contains({"min", "max", "pow", "atan2"}, stmt->name)) {
            if (stmt->args.size() != 2) {
                error("function `%s` takes exactly 2 arguments", stmt->name.c_str());
            }

        } else if (contains({"clamp", "mix"}, stmt->name)) {
            if (stmt->args.size() != 3) {
                error("function `%s` takes exactly 3 arguments", stmt->name.c_str());
            }

        } else if (contains({"dot", "distance"}, stmt->name)) {
            if (stmt->args.size() != 2) {
                error("function `%s` takes exactly 2 arguments", stmt->name.c_str());
            }
            stmt->dim = 1;

        } else if (contains({"cross"}, stmt->name)) {
            if (stmt->args.size() != 2) {
                error("function `%s` takes exactly 2 arguments", stmt->name.c_str());
            }
            stmt->dim = 3;

        } else {
            error("invalid function name `%s` (with %d args)",
                stmt->name.c_str(), stmt->args.size());
        }

        if (stmt->dim == 0) {
            for (auto const &arg: stmt->args) {
                ERROR_IF(arg->dim == 0);
                if (stmt->dim == 0) {
                    stmt->dim = arg->dim;
                    continue;
                }
                if (stmt->dim != 1 && arg->dim != 1 && arg->dim != stmt->dim) {
                    error("dimension mismatch in element-wise function `%s`: %d != %d",
                        stmt->name.c_str(), stmt->dim, arg->dim);
                } else if (arg->dim > stmt->dim)
                    stmt->dim = arg->dim;
            }
        }
        visit((Statement *)stmt);
    }

    void visit(VectorSwizzleStmt *stmt) {
        for (int s: stmt->swizzles) {
            if (s >= stmt->src->dim) {
                error("swizzle dimension out of range: %d >= %d",
                    s, stmt->src->dim);
            }
        }
        stmt->dim = stmt->swizzles.size();
        visit((Statement *)stmt);
    }

    void visit(AssignStmt *stmt) {
        if (stmt->dst->dim == 0 && stmt->src->dim != 0) {
            stmt->dst->dim = stmt->src->dim;
            if (auto dst = dynamic_cast<TempSymbolStmt *>(stmt->dst); dst) {
                dst->symids.clear();
                dst->symids.resize(dst->dim, -1);
            }
        } else if (stmt->src->dim > 1 && stmt->dst->dim != stmt->src->dim) {
            error("dimension mismatch in assign: %d != %d",
                stmt->dst->dim, stmt->src->dim);
        }
        stmt->dim = stmt->dst->dim;
        visit((Statement *)stmt);
    }
};

void apply_type_check(IR *ir) {
    TypeCheck visitor;
    visitor.apply(ir);
}

}
