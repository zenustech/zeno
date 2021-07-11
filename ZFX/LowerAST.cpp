#include "IR.h"
#include "AST.h"
#include "Stmts.h"
#include "Lexical.h"
#include <map>

struct LowerAST {
    std::unique_ptr<IR> ir = std::make_unique<IR>();

    std::map<std::string, int> symbols;
    std::map<std::string, int> constants;
    int symid = 0, constid = 0;

    int resolve_symbol(std::string const &sym) {
        auto it = symbols.find(sym);
        if (it != symbols.end()) {
            return it->second;
        }
        auto id = symid++;
        symbols[sym] = id;
        return id;
    }

    int resolve_literial(std::string const &expr) {
        auto it = constants.find(expr);
        if (it != constants.end()) {
            return it->second;
        }
        auto id = constid++;
        constants[expr] = id;
        return id;
    }

    Statement *serialize(AST *ast) {
        if (0) {

        } else if (contains({"+", "-", "*", "/", "%"}, ast->token) && ast->args.size() == 2) {
            auto lhs = serialize(ast->args[0].get());
            auto rhs = serialize(ast->args[1].get());
            return ir->emplace_back<BinaryOpStmt>(ast->token, lhs, rhs);

        } else if (contains({"+", "-"}, ast->token) && ast->args.size() == 1) {
            auto src = serialize(ast->args[0].get());
            return ir->emplace_back<UnaryOpStmt>(ast->token, src);

        } else if (contains({"="}, ast->token) && ast->args.size() == 2) {
            auto dst = serialize(ast->args[0].get());
            auto src = serialize(ast->args[1].get());
            return ir->emplace_back<AssignStmt>(dst, src);

        } else if (is_symbolic_atom(ast->token) && ast->args.size() == 0) {
            return ir->emplace_back<SymbolStmt>(resolve_symbol(ast->token));

        } else if (is_literial_atom(ast->token) && ast->args.size() == 0) {
            return ir->emplace_back<LiterialStmt>(resolve_literial(ast->token));

        } else {
            error("cannot lower AST at token: `%s` (%d args)\n",
                ast->token.c_str(), ast->args.size());
            return nullptr;
        }
    }
};

std::tuple
    < std::unique_ptr<IR>
    > lower_ast
    ( std::vector<AST::Ptr> asts
    ) {
    LowerAST lower;
    for (auto const &ast: asts) {
        lower.serialize(ast.get());
    }
    return
        { std::move(lower.ir)
        };
}
