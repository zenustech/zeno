#include "IRVisitor.h"
#include "Stmts.h"
#include <functional>
#include <stack>
#include <map>

namespace zfx {

struct SaveMathRegisters : Visitor<SaveMathRegisters> {
    using visit_stmt_types = std::tuple
        < AsmUnaryOpStmt
        , AsmBinaryOpStmt
        , Statement
        >;

    std::unique_ptr<IR> ir = std::make_unique<IR>();

    size_t minaddr = 0;
    std::map<int, std::vector<std::pair<int, int>>> regusage;

    void visit(AsmUnaryOpStmt *stmt) {
        bool need_save = contains({"sqrt", "sin", "cos", "tan", "asin", "acos",
            "atan", "exp", "log", "rsqrt", "floor", "ceil", "abs"}, stmt->op);
        std::stack<std::function<void()>> callbacks;
        if (need_save) {
            for (auto const &[regid, reginfo]: regusage) {
                bool found = false;
                for (auto const &[beg, end]: reginfo) {
                    if (beg < stmt->id && end > stmt->id) {
                        found = true;
                        break;
                    }
                }
                if (found) {
                    int addr = minaddr + regid;
                    ir->emplace_back<AsmLocalStoreStmt>(addr, regid);
                    callbacks.push([ir_ = ir.get(), regid_ = regid, addr_ = addr] () {
                        ir_->emplace_back<AsmLocalLoadStmt>(addr_, regid_);
                    });
                }
            }
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

std::unique_ptr<IR> apply_save_math_registers
    ( IR *ir
    , std::map<int, std::vector<std::pair<int, int>>> const &usage
    ) {
    SaveMathRegisters visitor;
    visitor.regusage = usage;
    visitor.minaddr = ir->size() * 50;
    visitor.apply(ir);
    return std::move(visitor.ir);
}

}
