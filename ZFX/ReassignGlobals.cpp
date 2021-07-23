#include "IRVisitor.h"
#include "Stmts.h"
#include <map>

namespace zfx {

struct ReassignGlobals : Visitor<ReassignGlobals> {
    using visit_stmt_types = std::tuple
        < AsmGlobalLoadStmt
        , AsmGlobalStoreStmt
        , AsmLocalLoadStmt
        , AsmLocalStoreStmt
        >;

    std::map<int, int> globals;
    std::map<int, int> locals;

    void local(int &mem) {
        if (auto it = locals.find(mem); it != locals.end()) {
            mem = it->second;
            return;
        }
        int newmem = locals.size();
        locals[mem] = newmem;
        mem = newmem;
    }

    void global(int &mem) {
        if (auto it = globals.find(mem); it != globals.end()) {
            mem = it->second;
            return;
        }
        int newmem = globals.size();
        globals[mem] = newmem;
        mem = newmem;
    }

    void visit(AsmLocalStoreStmt *stmt) {
        local(stmt->mem);
    }

    void visit(AsmLocalLoadStmt *stmt) {
        local(stmt->mem);
    }

    void visit(AsmGlobalStoreStmt *stmt) {
        global(stmt->mem);
    }

    void visit(AsmGlobalLoadStmt *stmt) {
        global(stmt->mem);
    }
};

std::map<int, int> apply_reassign_globals(IR *ir) {
    ReassignGlobals visitor;
    visitor.apply(ir);
    return visitor.globals;
}

}
