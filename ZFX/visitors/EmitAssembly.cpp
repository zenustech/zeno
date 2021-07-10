#include "IRVisitor.h"
#include "Stmts.h"
#include <sstream>

struct EmitAssembly : Visitor<EmitAssembly> {
    using visit_stmt_types = std::tuple
        < SymbolStmt
        , LiterialStmt
        , BinaryOpStmt
        , AssignStmt
        >;

    std::stringstream oss;

    decltype(auto) getResult() const {
        return oss.str();
    }

    template <class ...Ts>
    void emit(Ts &&...ts) {
        oss << format(std::forward<Ts>(ts)...) << '\n';
    }

    std::string express(int id) {
        return format("%d", id);
    }

    void visit(AssignStmt *stmt) {
        emit("assign %s %s",
            express(stmt->dst->id).c_str(),
            express(stmt->src->id).c_str());
    }

    void visit(BinaryOpStmt *stmt) {
        const char *opcode = [](auto const &op) {
            if (0) {
            } else if (op == "+") { return "add";
            } else if (op == "-") { return "sub";
            } else if (op == "*") { return "mul";
            } else if (op == "/") { return "div";
            } else if (op == "%") { return "mod";
            } else { error("invalid binary op: `%s`", op.c_str());
            }
        }(stmt->op);
        emit("%s %s %s", opcode,
            express(stmt->lhs->id).c_str(),
            express(stmt->rhs->id).c_str());
    }

    void visit(SymbolStmt *stmt) {
        emit(".symbol %d %s", stmt->id, stmt->name.c_str());
    }

    void visit(LiterialStmt *stmt) {
        emit(".literial %d %s", stmt->id, stmt->value.c_str());
    }
};

std::string apply_emit_assembly(IR *ir) {
    EmitAssembly emitter;
    emitter.apply(ir);
    return emitter.getResult();
}
