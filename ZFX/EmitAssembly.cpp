#include "IRVisitor.h"
#include "Stmts.h"
#include <sstream>

namespace zfx {

struct EmitAssembly : Visitor<EmitAssembly> {
    using visit_stmt_types = std::tuple
        < AssignStmt
        , AsmUnaryOpStmt
        , AsmBinaryOpStmt
        , AsmTernaryOpStmt
        , AsmFuncCallStmt
        , AsmAssignStmt
        , AsmLoadConstStmt
        , AsmParamLoadStmt
        , AsmLocalStoreStmt
        , AsmLocalLoadStmt
        , AsmGlobalStoreStmt
        , AsmGlobalLoadStmt
        , AsmIfStmt
        , AsmElseIfStmt
        , AsmElseStmt
        , AsmEndIfStmt
        , Statement
        >;

    std::stringstream oss;

    template <class ...Ts>
    void emit(Ts &&...ts) {
        oss << format(std::forward<Ts>(ts)...) << endl;
    }

    void visit(Statement *stmt) {
        error("unexpected statement type `%s`", typeid(*stmt).name());
    }

    static inline std::set<std::string> maths =
        { "sqrt"
        , "sin"
        , "cos"
        , "tan"
        , "asin"
        , "acos"
        , "atan"
        , "exp"
        , "log"
        , "floor"
        , "ceil"
        , "abs"
        , "rsqrt"
        , "min"
        , "max"
        , "pow"
        , "atan2"
        };

    void visit(AsmFuncCallStmt *stmt) {
        emit("%s %d %s", stmt->name.c_str(), stmt->dst,
            format_join(" ", "%d", stmt->args).c_str());
    }

    void visit(AsmUnaryOpStmt *stmt) {
        const char *opcode = [](auto const &op) {
            if (0) {
            } else if (op == "+") { return "mov";
            } else if (op == "-") { return "neg";
            } else if (op == "!") { return "not";
            } else { return op.c_str();
            }
        }(stmt->op);
        emit("%s %d %d", opcode,
            stmt->dst, stmt->src);
    }

    void visit(AsmBinaryOpStmt *stmt) {
        const char *opcode = [](auto const &op) {
            if (0) {
            } else if (op == "+") { return "add";
            } else if (op == "-") { return "sub";
            } else if (op == "*") { return "mul";
            } else if (op == "/") { return "div";
            } else if (op == "%") { return "mod";
            } else if (op == "&") { return "and";
            } else if (op == "|") { return "or";
            } else if (op == "^") { return "xor";
            } else if (op == "&!") { return "andnot";
            } else if (op == "==") { return "cmpeq";
            } else if (op == "!=") { return "cmpne";
            } else if (op == "<") { return "cmplt";
            } else if (op == "<=") { return "cmple";
            } else if (op == ">") { return "cmpgt";
            } else if (op == ">=") { return "cmpge";
            } else { return op.c_str();
            }
        }(stmt->op);
        emit("%s %d %d %d", opcode,
            stmt->dst, stmt->lhs, stmt->rhs);
    }

    void visit(AsmTernaryOpStmt *stmt) {
        emit("blend %d %d %d %d", stmt->dst,
            stmt->cond, stmt->lhs, stmt->rhs);
    }

    void visit(AsmGlobalStoreStmt *stmt) {
        emit("stg %d %d", stmt->val, stmt->mem);
    }

    void visit(AsmGlobalLoadStmt *stmt) {
        emit("ldg %d %d", stmt->val, stmt->mem);
    }

    void visit(AsmLocalStoreStmt *stmt) {
        emit("stl %d %d", stmt->val, stmt->mem);
    }

    void visit(AsmLocalLoadStmt *stmt) {
        emit("ldl %d %d", stmt->val, stmt->mem);
    }

    void visit(AsmParamLoadStmt *stmt) {
        emit("ldp %d %d", stmt->val, stmt->mem);
    }

    void visit(AsmLoadConstStmt *stmt) {
        emit("ldi %d %f", stmt->dst, stmt->value);
    }

    void visit(AsmAssignStmt *stmt) {
        emit("mov %d %d", stmt->dst, stmt->src);
    }

    void visit(AsmIfStmt *stmt) {
        emit(".if %d", stmt->cond);
    }

    void visit(AsmElseIfStmt *stmt) {
        emit(".elseif %d", stmt->cond);
    }

    void visit(AsmElseStmt *stmt) {
        emit(".else");
    }

    void visit(AsmEndIfStmt *stmt) {
        emit(".endif");
    }
};

std::string apply_emit_assembly(IR *ir) {
    EmitAssembly visitor;
    visitor.apply(ir);
    return visitor.oss.str();
}

}
