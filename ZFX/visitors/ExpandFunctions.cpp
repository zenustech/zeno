#include "IRVisitor.h"
#include "Stmts.h"

namespace zfx {

#define ERROR_IF(x) do { \
    if (x) { \
        error("`%s`", #x); \
    } \
} while (0)

struct ExpandFunctions : Visitor<ExpandFunctions> {
    using visit_stmt_types = std::tuple
        < FunctionCallStmt
        >;

    std::unique_ptr<IR> ir = std::make_unique<IR>();

    Statement *emit_op(std::string const &name, std::vector<Statement *> const &args) {
        if (0) {

        } else if (name == "mix") {
            if (args.size() != 3)
                error("function `%s` takes exactly 3 arguments", name.c_str());
            auto x = ir->push_clone_back(args[0]);
            auto y = ir->push_clone_back(args[1]);
            auto f = ir->push_clone_back(args[2]);
            // mix(x, y, a) = (y - x) * a + x
            return ir->emplace_back<BinaryOpStmt>("+",
                ir->emplace_back<BinaryOpStmt>("*",
                    ir->emplace_back<BinaryOpStmt>("-", y, x),
                    f), x);

        } else if (name == "sqrt") {
            if (args.size() != 1)
                error("function `%s` takes exactly 1 argument", name.c_str());
            return ir->emplace_back<UnaryOpStmt>("sqrt", args[0]);

        } else {
            error("invalid function name `%s` (with %d args)", name.c_str(), args.size());
        }
    }

    void visit(FunctionCallStmt *stmt) {
        emit_op(stmt->name, stmt->args);
    }
};

std::unique_ptr<IR> apply_expand_functions(IR *ir) {
    ExpandFunctions visitor;
    visitor.apply(ir);
    return std::move(visitor.ir);
}

}
