#include "IR.h"
#include "AST.h"
#include "Stmts.h"
#include "Lexical.h"
#include <map>

namespace zfx {

struct LowerAST {
    std::unique_ptr<IR> ir = std::make_unique<IR>();

    std::map<std::string, int> symdims;
    std::map<std::string, std::vector<int>> symbols;
    int symid = 0;

    std::vector<int> resolve_symbol(std::string const &sym) {
        if (auto it = symbols.find(sym); it != symbols.end()) {
            return it->second;
        }
        if (auto it = symdims.find(sym); it != symdims.end()) {
            auto dim = it->second;
            auto &res = symbols[sym];
            for (int i = 0; i < dim; i++) {
                res.push_back(symid++);
            }
            return res;
        }
        error("undefined symbol `%s`", sym.c_str());
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
            return ir->emplace_back<LiterialStmt>(ast->token);

        } else {
            error("cannot lower AST at token: `%s` (%d args)\n",
                ast->token.c_str(), ast->args.size());
            return nullptr;
        }
    }

    auto getSymbols() const {
        std::vector<std::pair<std::string, int>> ret(symid);
        for (auto const &[key, ids]: symbols) {
            for (int i = 0; i < ids.size(); i++) {
                ret[ids[i]] = std::make_pair(key, i);
            }
        }
        return ret;
    }
};

std::tuple
    < std::unique_ptr<IR>
    , std::vector<std::pair<std::string, int>>
    > lower_ast
    ( std::vector<AST::Ptr> asts
    , std::map<std::string, int> const &symdims
    ) {
    LowerAST lower;
    lower.symdims = symdims;
    for (auto const &ast: asts) {
        lower.serialize(ast.get());
    }
    auto symbols = lower.getSymbols();
    return
        { std::move(lower.ir)
        , symbols
        };
}

}
