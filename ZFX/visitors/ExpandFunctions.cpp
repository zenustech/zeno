#include "IRVisitor.h"
#include "Stmts.h"

namespace zfx {

#define ERROR_IF(x) do { \
    if (x) { \
        error("`%s`", #x); \
    } \
} while (0)

struct Stm {
    IR *ir = nullptr;
    Statement *stmt = nullptr;

    Stm() = default;

    Stm(IR *ir_, Statement *stmt_)
        : ir(ir_), stmt(stmt_) {}

    operator Statement *&() {
        return stmt;
    }

    operator Statement * const &() const {
        return stmt;
    }
};

Stm operator+(Stm const &lhs, Stm const &rhs) {
    return {lhs.ir, lhs.ir->emplace_back<BinaryOpStmt>("+", lhs.stmt, rhs.stmt)};
}

Stm operator-(Stm const &lhs, Stm const &rhs) {
    return {lhs.ir, lhs.ir->emplace_back<BinaryOpStmt>("-", lhs.stmt, rhs.stmt)};
}

Stm operator*(Stm const &lhs, Stm const &rhs) {
    return {lhs.ir, lhs.ir->emplace_back<BinaryOpStmt>("*", lhs.stmt, rhs.stmt)};
}

Stm operator/(Stm const &lhs, Stm const &rhs) {
    return {lhs.ir, lhs.ir->emplace_back<BinaryOpStmt>("/", lhs.stmt, rhs.stmt)};
}

Stm operator+(Stm const &src) {
    return {src.ir, src.ir->emplace_back<UnaryOpStmt>("+", src.stmt)};
}

Stm operator-(Stm const &src) {
    return {src.ir, src.ir->emplace_back<UnaryOpStmt>("-", src.stmt)};
}

struct ExpandFunctions : Visitor<ExpandFunctions> {
    using visit_stmt_types = std::tuple
        < FunctionCallStmt
        , Statement
        >;

    std::unique_ptr<IR> ir = std::make_unique<IR>();

    Statement *emit_op(std::string const &name, std::vector<Statement *> const &args) {
        if (0) {

        } else if (name == "mix") {
            if (args.size() != 3)
                error("function `%s` takes exactly 3 arguments", name.c_str());
            Stm x(ir.get(), ir->push_clone_back(args[0]));
            Stm y(ir.get(), ir->push_clone_back(args[1]));
            Stm a(ir.get(), ir->push_clone_back(args[2]));
            return (y - x) * a + x;

        } else if (name == "sqrt") {
            if (args.size() != 1)
                error("function `%s` takes exactly 1 argument", name.c_str());
            return ir->emplace_back<UnaryOpStmt>("sqrt", args[0]);

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

std::unique_ptr<IR> apply_expand_functions(IR *ir) {
    ExpandFunctions visitor;
    visitor.apply(ir);
    return std::move(visitor.ir);
}

}
