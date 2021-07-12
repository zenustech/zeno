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

    Statement *operator->() const {
        return stmt;
    }

    Stm operator[](std::vector<int> const &indices) const {
        auto const &src = *this;
        return {src.ir, src.ir->emplace_back<VectorSwizzleStmt>(indices, src.stmt)};
    }

    Stm operator[](int index) const {
        auto const &src = *this;
        return {src.ir, src.ir->emplace_back<VectorSwizzleStmt>(std::vector<int>{index}, src.stmt)};
    }
};

template <class ExpandFuncs>
Stm stm(ExpandFuncs *that, std::string const &name, std::vector<Stm> const &args) {
    ERROR_IF(args.size() == 0);
    return {args[0].ir, that->emit_op(name, args)};
}

Stm stm(std::string const &op_name, Stm const &lhs, Stm const &rhs) {
    return {lhs.ir, lhs.ir->emplace_back<BinaryOpStmt>(op_name, lhs.stmt, rhs.stmt)};
}

Stm stm(std::string const &op_name, Stm const &src) {
    return {src.ir, src.ir->emplace_back<UnaryOpStmt>(op_name, src.stmt)};
}

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
            ERROR_IF(args.size() != 3);
            Stm x(ir.get(), ir->push_clone_back(args[0]));
            Stm y(ir.get(), ir->push_clone_back(args[1]));
            Stm a(ir.get(), ir->push_clone_back(args[2]));
            return (y - x) * a + x;

        } else if (name == "clamp") {
            ERROR_IF(args.size() != 3);
            Stm x(ir.get(), ir->push_clone_back(args[0]));
            Stm a(ir.get(), ir->push_clone_back(args[1]));
            Stm b(ir.get(), ir->push_clone_back(args[2]));
            return stm("min", b, stm("max", a, x));

        } else if (name == "dot") {
            ERROR_IF(args.size() != 2);
            Stm a(ir.get(), ir->push_clone_back(args[0]));
            Stm b(ir.get(), ir->push_clone_back(args[1]));
            ERROR_IF(!a->dim || !b->dim);
            if (a->dim != b->dim) {
                error("dimension mismatch for function `%s`: %d != %d",
                    name.c_str(), a->dim, b->dim);
            }
            Stm ret = a[0] * b[0];
            for (int i = 1; i < a->dim; i++) {
                ret = ret + a[i] * b[i];
            }
            return ret;

        } else if (contains({"sqrt", "sin", "cos", "tan", "asin", "acos",
            "atan", "exp", "log", "rsqrt", "floor", "ceil"}, name)) {
            ERROR_IF(args.size() != 1);
            return ir->emplace_back<UnaryOpStmt>(name, args[0]);

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
