#include "IR.h"
#include "AST.h"
#include "Stmts.h"
#include "Lexical.h"
#include <sstream>
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
            res.clear();
            for (int i = 0; i < dim; i++) {
                res.push_back(symid++);
            }
            return res;
        }
        //error("undefined symbol `%s`", sym.c_str());
        return {};  // undefined for now, will be further defined in TypeCheck
    }

    std::map<std::string, int> pardims;
    std::map<std::string, std::vector<int>> params;
    int parid = 0;

    std::vector<int> resolve_param(std::string const &par) {
        if (auto it = params.find(par); it != params.end()) {
            return it->second;
        }
        if (auto it = pardims.find(par); it != pardims.end()) {
            auto dim = it->second;
            auto &res = params[par];
            res.clear();
            for (int i = 0; i < dim; i++) {
                res.push_back(parid++);
            }
            return res;
        }
        return {};
    }

    std::map<std::string, SymbolStmt *> globsyms;

    SymbolStmt *emplace_global_symbol(std::string const &name) {
        auto symids = resolve_symbol(name);
        if (symids.size() == 0)
            return nullptr;
        if (auto it = globsyms.find(name); it != globsyms.end()) {
            return it->second;
        }
        auto ret = ir->emplace_back<SymbolStmt>(symids);
        globsyms[name] = ret;
        return ret;
    }

    std::map<std::string, ParamSymbolStmt *> parasyms;

    ParamSymbolStmt *emplace_param_symbol(std::string const &name) {
        auto symids = resolve_param(name);
        if (symids.size() == 0)
            return nullptr;
        if (auto it = parasyms.find(name); it != parasyms.end()) {
            return it->second;
        }
        auto ret = ir->emplace_back<ParamSymbolStmt>(symids);
        parasyms[name] = ret;
        return ret;
    }

    std::map<std::string, TempSymbolStmt *> tempsyms;
    int tmpid = 0;

    std::map<int, std::string> temporaries;

    TempSymbolStmt *emplace_temporary_symbol(std::string const &name) {
        if (auto it = tempsyms.find(name); it != tempsyms.end()) {
            return it->second;
        }
        auto id = tmpid++;
        auto ret = ir->emplace_back<TempSymbolStmt>(id, std::vector<int>{});
        tempsyms[name] = ret;
        temporaries[id] = name;
        return ret;
    }

    Statement *serialize(AST *ast) {
        if (0) {

        } else if (contains({"+", "-", "*", "/", "%",
                    "==", "!=", "<", "<=", ">", ">=",
                    "&", "&!", "|", "^",
                    }, ast->token) && ast->args.size() == 2) {
            auto lhs = serialize(ast->args[0].get());
            auto rhs = serialize(ast->args[1].get());
            return ir->emplace_back<BinaryOpStmt>(ast->token, lhs, rhs);

        } else if (contains({"+", "-", "!"}, ast->token) && ast->args.size() == 1) {
            auto src = serialize(ast->args[0].get());
            return ir->emplace_back<UnaryOpStmt>(ast->token, src);

        } else if (contains({"="}, ast->token) && ast->args.size() == 2) {
            auto dst = serialize(ast->args[0].get());
            auto src = serialize(ast->args[1].get());
            return ir->emplace_back<AssignStmt>(dst, src);

        } else if (contains({"?"}, ast->token) && ast->args.size() == 3) {
            auto cond = serialize(ast->args[0].get());
            auto lhs = serialize(ast->args[1].get());
            auto rhs = serialize(ast->args[2].get());
            return ir->emplace_back<TernaryOpStmt>(cond, lhs, rhs);

        } else if (contains({"+=", "-=", "*=", "/=", "%="},
            ast->token) && ast->args.size() == 2) {
            auto dst = serialize(ast->args[0].get());
            auto src = serialize(ast->args[1].get());
            auto val = ir->emplace_back<BinaryOpStmt>(ast->token.substr(
                0, ast->token.size() - 1), dst, src);
            return ir->emplace_back<AssignStmt>(dst, val);

        } else if (contains({"if"}, ast->token) && ast->args.size() == 1) {
            auto cond = serialize(ast->args[0].get());
            return ir->emplace_back<FrontendIfStmt>(cond);

        } else if (contains({"elseif"}, ast->token) && ast->args.size() == 1) {
            auto cond = serialize(ast->args[0].get());
            return ir->emplace_back<FrontendElseIfStmt>(cond);

        } else if (contains({"else"}, ast->token) && ast->args.size() == 0) {
            return ir->emplace_back<FrontendElseStmt>();

        } else if (contains({"endif"}, ast->token) && ast->args.size() == 0) {
            return ir->emplace_back<FrontendEndIfStmt>();

        } else if (is_symbolic_atom(ast->token) && ast->args.size() == 0) {
            if (auto ret = emplace_global_symbol(ast->token); ret) {
                return ret;
            }
            if (auto ret = emplace_param_symbol(ast->token); ret) {
                return ret;
            }
            return emplace_temporary_symbol(ast->token);

        } else if (is_literial_atom(ast->token) && ast->args.size() == 0) {
            float value = 0.0f;
            if (!(std::stringstream(ast->token) >> value))
                error("failed to parse float literial: %s\n", ast->token.c_str());
            return ir->emplace_back<LiterialStmt>(value);

        } else if (contains({"()"}, ast->token) && ast->args.size() >= 1) {
            auto func_name = ast->args[0]->token;
            std::vector<Statement *> func_args;
            for (int i = 1; i < ast->args.size(); i++) {
                func_args.push_back(serialize(ast->args[i].get()));
            }
            return ir->emplace_back<FunctionCallStmt>(func_name, func_args);

        } else if (contains({"."}, ast->token) && ast->args.size() == 2) {
            auto expr = serialize(ast->args[0].get());
            auto swiz_expr = ast->args[1]->token;
            std::vector<int> swizzles;
            for (auto const &c: swiz_expr) {
                int axis = swizzle_from_char(c);
                if (axis == -1)
                    error("invalid swizzle character: `%c`", c);
                swizzles.push_back(axis);
            }
            return ir->emplace_back<VectorSwizzleStmt>(swizzles, expr);

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

    auto getParams() const {
        std::vector<std::pair<std::string, int>> ret(parid);
        for (auto const &[key, ids]: params) {
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
    , std::vector<std::pair<std::string, int>>
    , std::map<int, std::string>
    > lower_ast
    ( std::vector<AST::Ptr> asts
    , std::map<std::string, int> const &symdims
    , std::map<std::string, int> const &pardims
    ) {
    LowerAST lower;
    lower.symdims = symdims;
    lower.pardims = pardims;
    for (auto const &ast: asts) {
        lower.serialize(ast.get());
    }
    auto symbols = lower.getSymbols();
    auto params = lower.getParams();
    auto temporaries = lower.temporaries;
    return
        { std::move(lower.ir)
        , symbols
        , params
        , temporaries
        };
}

}
