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
    std::map<int, std::pair<int, int>> regusage;

    void visit(AsmUnaryOpStmt *stmt) {
        bool need_save = contains({"sqrt", "sin", "cos", "tan", "asin", "acos",
            "atan", "exp", "log", "rsqrt", "floor", "ceil", "abs"}, stmt->op);
        std::stack<std::function<void()>> callbacks;
        if (need_save) {
            for (auto const &[regid, begend]: regusage) {
                auto const &[beg, end] = begend;
                if (beg < stmt->id && end > stmt->id) {
                    ir->emplace_back<AsmLocalStoreStmt>(minaddr, regid);
                    callbacks.push([ir_ = ir.get(), regid_ = regid, addr = minaddr] () {
                        ir_->emplace_back<AsmLocalLoadStmt>(addr, regid_);
                    });
                    minaddr++;
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
        , std::map<int, std::pair<int, int>> const &regusage
        ) {
    SaveMathRegisters visitor;
    visitor.regusage = regusage;
    visitor.minaddr = ir->size();
    visitor.apply(ir);
    return std::move(visitor.ir);
}

}
