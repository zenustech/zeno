#include "IRVisitor.h"
#include "Stmts.h"
#include <sstream>

struct EmitAssembly : Visitor<EmitAssembly> {
    using visit_stmt_types = std::tuple
        < AssignStmt
        , AsmBinaryOpStmt
        , AsmAssignStmt
        , AsmLoadConstStmt
        , AsmLoadSymbolStmt
        , AsmMemoryStoreStmt
        , AsmMemoryLoadStmt
        , Statement
        >;

    std::stringstream oss;

    template <class ...Ts>
    void emit(Ts &&...ts) {
        oss << format(std::forward<Ts>(ts)...) << endl;
    }

    void visit(Statement *stmt) {
        emit("; unexpected `%s`", typeid(*stmt).name());
    }

    void visit(AsmUnaryOpStmt *stmt) {
        const char *opcode = [](auto const &op) {
            if (0) {
            } else if (op == "+") { return "mov";
            } else if (op == "-") { return "neg";
            } else { error("invalid unary op `%s`", op.c_str());
            }
        }(stmt->op);
        emit("%s r%d r%d", opcode,
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
            } else { error("invalid binary op `%s`", op.c_str());
            }
        }(stmt->op);
        emit("%s r%d r%d r%d", opcode,
            stmt->dst, stmt->lhs, stmt->rhs);
    }

    void visit(AsmMemoryStoreStmt *stmt) {
        emit("st r%d [%d]", stmt->val, stmt->mem);
    }

    void visit(AsmMemoryLoadStmt *stmt) {
        emit("ld r%d [%d]", stmt->val, stmt->mem);
    }

    void visit(AsmLoadSymbolStmt *stmt) {
        emit("lds r%d [%s]", stmt->dst, stmt->name.c_str());
    }

    void visit(AsmLoadConstStmt *stmt) {
        emit("ldi r%d #%s", stmt->dst, stmt->name.c_str());
    }

    void visit(AsmAssignStmt *stmt) {
        emit("mov r%d r%d", stmt->dst, stmt->src);
    }
};

std::string apply_emit_assembly(IR *ir) {
    EmitAssembly visitor;
    visitor.apply(ir);
    return visitor.oss.str();
}
