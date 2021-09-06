#include "IRVisitor.h"
#include "Stmts.h"
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

std::map<int, int> apply_reassign_parameters(IR *ir) {
    ReassignParameters reassign;
    reassign.apply(ir);
    return reassign.uniforms;
}

}
