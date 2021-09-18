#include "IRVisitor.h"
#include "Stmts.h"
#include <functional>
#include <stack>
#include <map>

namespace zfx {

struct SaveMathRegisters : Visitor<SaveMathRegisters> {
    using visit_stmt_types = std::tuple
        < AsmFuncCallStmt
        , Statement
        >;

    std::unique_ptr<IR> ir = std::make_unique<IR>();

    size_t minaddr = 0;
    int nregs = 0;

    void visit(AsmFuncCallStmt *stmt) {
        std::stack<std::function<void()>> callbacks;
        for (int regid = 0; regid < nregs; regid++) {
            if (regid == stmt->dst)
                continue;
            int addr = minaddr + regid;
            ir->emplace_back<AsmLocalStoreStmt>(addr, regid);
            callbacks.push([=]() {
                ir->emplace_back<AsmLocalLoadStmt>(addr, regid);
            });
        }
        visit((Statement *)stmt);
        while (callbacks.size()) {
            callbacks.top()();
            callbacks.pop();
        }
    }

    void visit(Statement *stmt) {
        ir->push_clone_back(stmt);
    }
};

std::unique_ptr<IR> apply_save_math_registers(IR *ir, int nregs, int memsize) {
    SaveMathRegisters visitor;
    visitor.nregs = nregs;
    visitor.minaddr = memsize;
    visitor.apply(ir);
    return std::move(visitor.ir);
}

}
