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
        , TempSymbolStmt
        , ParamSymbolStmt
        , SymbolStmt
        , VectorSwizzleStmt
        , VectorComposeStmt
        , FunctionCallStmt
        , TernaryOpStmt
        , BinaryOpStmt
        , UnaryOpStmt
        , Statement
        >;

    void visit(Statement *stmt) {
        if (!stmt->is_control_stmt() && stmt->dim == 0) {
            error("statement $%d has unclear type", stmt->id);
        }
    }

    void visit(TempSymbolStmt *stmt) {
    }

    void visit(ParamSymbolStmt *stmt) {
        stmt->dim = stmt->symids.size();
        visit((Statement *)stmt);
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

    void visit(TernaryOpStmt *stmt) {
        if (stmt->cond->dim > 1 && stmt->lhs->dim > 1 && stmt->rhs->dim > 1
            && (stmt->cond->dim != stmt->lhs->dim || stmt->lhs->dim != stmt->rhs->dim)) {
            error("dimension mismatch in ternary `?`: %d != %d != %d",
                stmt->cond->dim, stmt->lhs->dim, stmt->rhs->dim);
        }
        ERROR_IF(stmt->cond->dim == 0);
        ERROR_IF(stmt->lhs->dim == 0);
        ERROR_IF(stmt->rhs->dim == 0);
        stmt->dim = std::max(stmt->cond->dim, std::max(stmt->lhs->dim, stmt->rhs->dim));
        visit((Statement *)stmt);
    }

    void visit(FunctionCallStmt *stmt) {
        stmt->dim = 0;
        auto const &name = stmt->name;

        if (name.substr(0, 3) == "vec" && name.size() == 4 && isdigit(name[3])) {
            int dim = name[3] - '0';
            stmt->dim = dim;

        } else if (contains({
                    "sqrt",
                    "sin",
                    "cos",
                    "tan",
                    "asin",
                    "acos",
                    "atan",
                    "exp",
                    "log",
                    "rsqrt",
                    "floor",
                    "ceil",
                    "round",
                    "abs",
                    "all",
                    "any",
            }, name)) {
            if (stmt->args.size() != 1) {
                error("function `%s` takes exactly 1 argument", name.c_str());
            }

        } else if (contains({
                    "min",
                    "max",
                    "pow",
                    "atan2",
            }, name)) {
            if (stmt->args.size() != 2) {
                error("function `%s` takes exactly 2 arguments", name.c_str());
            }

        } else if (contains({"clamp", "mix"}, name)) {
            if (stmt->args.size() != 3) {
                error("function `%s` takes exactly 3 arguments", name.c_str());
            }

        } else if (contains({"normalize"}, name)) {
            if (stmt->args.size() != 1) {
                error("function `%s` takes exactly 1 argument", name.c_str());
            }

        } else if (contains({"length"}, name)) {
            if (stmt->args.size() != 1) {
                error("function `%s` takes exactly 1 argument", name.c_str());
            }
            stmt->dim = 1;

        } else if (contains({"dot", "distance"}, name)) {
            if (stmt->args.size() != 2) {
                error("function `%s` takes exactly 2 arguments", name.c_str());
            }
            stmt->dim = 1;

        } else if (contains({"cross"}, name)) {
            if (stmt->args.size() != 2) {
                error("function `%s` takes exactly 2 arguments", name.c_str());
            }
            stmt->dim = 3;

        } else {
            error("invalid function name `%s` (with %d args)",
                name.c_str(), stmt->args.size());
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

    void visit(VectorComposeStmt *stmt) {
        stmt->dim = stmt->dimension;
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
