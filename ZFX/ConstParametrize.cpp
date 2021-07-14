#include "IRVisitor.h"
#include "Stmts.h"
#include <functional>
#include <map>

namespace zfx {

struct ReassignParameters : Visitor<ReassignParameters> {
    using visit_stmt_types = std::tuple
        < AsmParamLoadStmt
        >;

    std::map<int, int> uniforms;

    int replace(int id) {
        if (auto it = uniforms.find(id); it != uniforms.end()) {
            return it->second;
        }
        int ret = uniforms.size();
        uniforms[id] = ret;
        return ret;
    }

    void visit(AsmParamLoadStmt *stmt) {
        stmt->mem = replace(stmt->mem);
    }
};

struct ConstParametrize : Visitor<ConstParametrize> {
    using visit_stmt_types = std::tuple
        < AsmLoadConstStmt
        , Statement
        >;

    std::unique_ptr<IR> ir = std::make_unique<IR>();

    int nuniforms = 0;
    std::map<std::string, int> constants;

    int lookup(std::string const &expr) {
        if (auto it = constants.find(expr); it != constants.end()) {
            return it->second;
        }
        int constid = nuniforms + constants.size();
        constants[expr] = constid;
        return constid;
    }

    void visit(Statement *stmt) {
        ir->push_clone_back(stmt);
    }

    void visit(AsmLoadConstStmt *stmt) {
        int constid = lookup(stmt->value);
        ir->emplace_back<AsmParamLoadStmt>(constid, stmt->dst);
    }

    auto getConstants() const {
        std::map<int, std::string> res;
        for (auto const &[name, idx]: constants) {
            res[idx] = name;
        }
        return res;
    }
};

std::pair
    < std::map<int, int>
    , std::map<int, std::string>
    > apply_const_parametrize(IR *ir) {
    ReassignParameters reassign;
    reassign.apply(ir);
    ConstParametrize visitor;
    visitor.nuniforms = reassign.uniforms.size();
    visitor.apply(ir);
    *ir = *visitor.ir;
    return
        { reassign.uniforms
        , visitor.getConstants()
        };
}

}
