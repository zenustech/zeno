#include "IRVisitor.h"
#include "Stmts.h"
#include <map>
#include <functional>
#include <set>
#include <map>
#include <array>
#include <cstdio>
#include <cassert>

#define NREGS 8

namespace zfx {

struct InspectRegisters : Visitor<InspectRegisters> {
    using visit_stmt_types = std::tuple
        < AsmAssignStmt
        , AsmUnaryOpStmt
        , AsmBinaryOpStmt
        , AsmLoadConstStmt
        , AsmLocalLoadStmt
        , AsmLocalStoreStmt
        , AsmGlobalLoadStmt
        , AsmGlobalStoreStmt
        >;

    void touch(int stmtid, int regid) {
        // TODO
    }

    void visit(AsmAssignStmt *stmt) {
        touch(stmt->id, stmt->dst);
        touch(stmt->id, stmt->src);
    }

    void visit(AsmUnaryOpStmt *stmt) {
        touch(stmt->id, stmt->dst);
        touch(stmt->id, stmt->src);
    }

    void visit(AsmBinaryOpStmt *stmt) {
        touch(stmt->id, stmt->dst);
        touch(stmt->id, stmt->lhs);
        touch(stmt->id, stmt->rhs);
    }

    void visit(AsmLoadConstStmt *stmt) {
        touch(stmt->id, stmt->dst);
    }

    void visit(AsmLocalLoadStmt *stmt) {
        touch(stmt->id, stmt->val);
    }

    void visit(AsmLocalStoreStmt *stmt) {
        touch(stmt->id, stmt->val);
    }

    void visit(AsmGlobalLoadStmt *stmt) {
        touch(stmt->id, stmt->val);
    }

    void visit(AsmGlobalStoreStmt *stmt) {
        touch(stmt->id, stmt->val);
    }
};

void apply_register_allocation(IR *ir) {
    InspectRegisters inspect;
    inspect.apply(ir);
}

}
