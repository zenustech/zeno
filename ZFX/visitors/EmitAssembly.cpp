#include "IRVisitor.h"
#include "Stmts.h"
#include <sstream>

struct EmitAssembly : Visitor<EmitAssembly> {
    using visit_stmt_types = std::tuple
        < AssignStmt
        , BinaryOpStmt
        , LiterialStmt
        , SymbolStmt
        , Statement
        >;

    std::stringstream oss;

    template <class ...Ts>
    void emit(Ts &&...ts) {
        oss << format(std::forward<Ts>(ts)...) << endl;
    }

    void visit(Statement *stmt) {
        emit("; unexpected %s", typeid(*stmt).name());
    }

    void visit(SymbolStmt *stmt) {
        emit("symbol %d %s", stmt->id, stmt->name.c_str());
    }

    void visit(LiterialStmt *stmt) {
        emit("const %d %s", stmt->id, stmt->value.c_str());
    }

    void visit(BinaryOpStmt *stmt) {
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
        emit("%s %d %d %d", opcode,
            stmt->id, stmt->lhs->id, stmt->rhs->id);
    }

    void visit(AssignStmt *stmt) {
        emit("mov %d %d", stmt->dst->id, stmt->src->id);
    }
};

std::string apply_emit_assembly(IR *ir) {
    EmitAssembly visitor;
    visitor.apply(ir);
    return visitor.oss.str();
}
