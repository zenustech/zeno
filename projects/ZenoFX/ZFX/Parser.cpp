#include "AST.h"
#include "Lexical.h"

namespace zfx {

struct Parser {
    std::vector<std::string> tokens;

    explicit Parser
        ( std::vector<std::string> const &tokens_
        )
        : tokens(tokens_)
    {
    }

    AST::Ptr parse_atom(AST::Iter iter) {
        if (auto s = *iter; is_atom(s)) {
            return make_ast(s, iter + 1);
        }
        return nullptr;
    }

    AST::Ptr parse_operator(AST::Iter iter, std::set<std::string> const &allows) {
        if (auto s = *iter; contains(allows, s)) {
            return make_ast(s, iter + 1);
        }
        return nullptr;
    }

    AST::Ptr parse_compound(AST::Iter iter) {
        if (auto atm = parse_atom(iter); atm) {
            if (auto bra = parse_operator(atm->iter, {"("}); bra) {
                auto last_iter = bra->iter;
                std::vector<AST::Ptr> args;
                args.push_back(std::move(atm));
                while (1) if (auto arg = parse_expr(last_iter); arg) {
                        args.push_back(std::move(arg));
                        last_iter = arg->iter;
                        if (auto sep = parse_operator(last_iter, {","}); sep) {
                            last_iter = sep->iter;
                        } else break;
                } else break;
                if (auto ket = parse_operator(last_iter, {")"}); ket) {
                    return make_ast("()", ket->iter, args);
                } else {
                    error("`)` expected at the end of argument list, got `%s`",
                        last_iter->c_str());
                }
            }
            return atm;
        }
        if (auto ope = parse_operator(iter, {"+", "-", "!"}); ope) {
            if (auto rhs = parse_compound(ope->iter); rhs) {
                return make_ast(ope->token, rhs->iter, {std::move(rhs)});
            } else {
                error("expression expected after unary `%s`, got `%s`",
                    iter->c_str(), ope->iter->c_str());
            }
        }
        if (auto ope = parse_operator(iter, {"("}); ope) {
            if (auto rhs = parse_expr(ope->iter); rhs) {
                if (auto ket = parse_operator(rhs->iter, {")"}); ket) {
                    rhs->iter = ket->iter;
                    return rhs;
                } else {
                    error("`)` expected to match the `(`, got `%s`",
                        rhs->iter->c_str());
                }
            } else {
                error("expression expected after `(`, got `%s`",
                    ope->iter->c_str());
            }
        }
        return nullptr;
    }

    AST::Ptr parse_factor(AST::Iter iter) {
        if (auto comp = parse_compound(iter); comp) {
            if (auto ope = parse_operator(comp->iter, {"."}); ope) {
                if (auto atm = parse_atom(ope->iter); atm) {
                    return make_ast(ope->token, atm->iter, {std::move(comp), std::move(atm)});
                } else error("`%s` expecting swizzle expression, got `%s`",
                    comp->iter->c_str(), ope->iter->c_str());
            }
            return comp;
        }
        return nullptr;
    }

    AST::Ptr parse_term(AST::Iter iter) {
        if (auto lhs = parse_factor(iter); lhs) {
            while (1) if (auto ope = parse_operator(lhs->iter, {"*", "/", "%"}); ope) {
                if (auto rhs = parse_factor(ope->iter); rhs) {
                    lhs = make_ast(ope->token, rhs->iter, {std::move(lhs), std::move(rhs)});
                    } else error("`%s` expecting rhs, got `%s`",
                        lhs->iter->c_str(), ope->iter->c_str());
            } else break;
            return lhs;
        }
        return nullptr;
    }

    AST::Ptr parse_side(AST::Iter iter) {
        if (auto lhs = parse_term(iter); lhs) {
            while (1) if (auto ope = parse_operator(lhs->iter, {"+", "-"}); ope) {
                    if (auto rhs = parse_term(ope->iter); rhs) {
                        lhs = make_ast(ope->token, rhs->iter, {std::move(lhs), std::move(rhs)});
                    } else error("`%s` expecting rhs, got `%s`",
                        lhs->iter->c_str(), ope->iter->c_str());
            } else break;
            return lhs;
        }
        return nullptr;
    }

    AST::Ptr parse_cond(AST::Iter iter) {
        if (auto lhs = parse_side(iter); lhs) {
            while (1) if (auto ope = parse_operator(lhs->iter, {
                        "==", "!=", "<", "<=", ">", ">="}); ope) {
                    if (auto rhs = parse_side(ope->iter); rhs) {
                        lhs = make_ast(ope->token, rhs->iter, {std::move(lhs), std::move(rhs)});
                    } else error("`%s` expecting rhs, got `%s`",
                        lhs->iter->c_str(), ope->iter->c_str());
            } else break;
            return lhs;
        }
        return nullptr;
    }

    AST::Ptr parse_andexpr(AST::Iter iter) {
        if (auto lhs = parse_cond(iter); lhs) {
            while (1) if (auto ope = parse_operator(lhs->iter, {"&", "&!"}); ope) {
                    if (auto rhs = parse_cond(ope->iter); rhs) {
                        lhs = make_ast(ope->token, rhs->iter, {std::move(lhs), std::move(rhs)});
                    } else error("`%s` expecting rhs, got `%s`",
                        lhs->iter->c_str(), ope->iter->c_str());
            } else break;
            return lhs;
        }
        return nullptr;
    }

    AST::Ptr parse_orexpr(AST::Iter iter) {
        if (auto lhs = parse_andexpr(iter); lhs) {
            while (1) if (auto ope = parse_operator(lhs->iter, {"|", "^"}); ope) {
                    if (auto rhs = parse_andexpr(ope->iter); rhs) {
                        lhs = make_ast(ope->token, rhs->iter, {std::move(lhs), std::move(rhs)});
                    } else error("`%s` expecting rhs, got `%s`",
                        lhs->iter->c_str(), ope->iter->c_str());
            } else break;
            return lhs;
        }
        return nullptr;
    }

    AST::Ptr parse_expr(AST::Iter iter) {
        if (auto cond = parse_orexpr(iter); cond) {
            if (auto ope = parse_operator(cond->iter, {"?"}); ope) {
                if (auto lhs = parse_orexpr(ope->iter); lhs) {
                    if (auto ope2 = parse_operator(lhs->iter, {":"}); ope2) {
                        if (auto rhs = parse_expr(ope2->iter); rhs) {
                            return make_ast(ope->token, rhs->iter, {
                                    std::move(cond), std::move(lhs), std::move(rhs)});
                        } else error("`%s` expecting rhs, got `%s`",
                            lhs->iter->c_str(), ope2->iter->c_str());
                    } else error("`%s` expecting `:`, got `%s`",
                        ope->iter->c_str(), lhs->iter->c_str());
                } else error("`%s` expecting lhs, got `%s`",
                    lhs->iter->c_str(), ope->iter->c_str());
            } else {
                return cond;
            }
        }
        return nullptr;
    }

    AST::Ptr parse_stmt(AST::Iter iter) {
        if (auto ope = parse_operator(iter, {"if", "elseif"}); ope) {
            if (auto cond = parse_expr(ope->iter)) {
                return make_ast(ope->token, cond->iter, {std::move(cond)});
            } else {
                error("`%s` is expecting condition, got `%s`",
                        iter->c_str(), ope->iter->c_str());
            }
        } else if (auto ope = parse_operator(iter, {"else", "endif"}); ope) {
            return make_ast(ope->token, ope->iter);
        } else if (auto lhs = parse_factor(iter); lhs) {
            if (auto ope = parse_operator(lhs->iter, {"=",
                "+=", "-=", "*=", "/=", "%="}); ope) {
                if (auto rhs = parse_expr(ope->iter); rhs) {
                    return make_ast(ope->token, rhs->iter, {std::move(lhs), std::move(rhs)});
                } else {
                    error("rhs expected in assign statement, got `%s`",
                        ope->iter->c_str());
                }
            } else {
                error("`=` or `+=` series expected in statement, got `%s`",
                    lhs->iter->c_str());
            }
        }
        return nullptr;
    }

    auto parse() {
        std::vector<AST::Ptr> asts;
        AST::Iter iter = tokens.begin();
        while (iter != tokens.end()) {
            auto p = parse_stmt(iter);
            if (!p) break;
            iter = p->iter;
            asts.push_back(std::move(p));
        }
        return asts;
    }
};

std::vector<AST::Ptr> parse(std::string const &code) {
    auto tokens = tokenize(code.c_str());
    Parser parser(tokens);
    return parser.parse();
}

}
