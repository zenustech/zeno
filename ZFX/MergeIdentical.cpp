#include "IRVisitor.h"
#include "Stmts.h"
#include <functional>
#include <map>

namespace zfx {

struct MergeIdentical : Visitor<MergeIdentical> {
    using visit_stmt_types = std::tuple
        < AsmLocalStoreStmt
        , AsmLocalLoadStmt
        , AsmGlobalStoreStmt
        , AsmGlobalLoadStmt
        , Statement
        >;

    std::unique_ptr<IR> ir = std::make_unique<IR>();

    std::map<std::string, int> revstmts;
    std::map<int, int> regs;

    void visit(Statement *stmt) {
        if (stmt->is_control_stmt()) {
            revstmts.clear();
            regs.clear();
            ir->push_clone_back(stmt);
            return;
        }

        std::stringstream ss;
        ss << stmt->serialize_identity();
        for (auto r: stmt->source_registers()) {
            auto s = regs.at(r);
            ss << '|' << s;
        }
        auto key = ss.str();

        bool found = false;
        if (auto it = revstmts.find(key); it != revstmts.end()) {
            for (auto [r_d, r_s]: regs) {
                if (r_s == it->second) {
                    for (auto d: stmt->dest_registers()) {
                        if (d != r_d)
                            ir->emplace_back<AsmAssignStmt>(d, r_d);
                    }
                    found = true;
                    break;
                }
            }
        }

        if (!found) {
            ir->push_clone_back(stmt);
        }

        revstmts[key] = stmt->id;
        for (auto d: stmt->dest_registers()) {
            regs[d] = stmt->id;
        }
    }

    void visit(AsmLocalLoadStmt *stmt) {
        ir->push_clone_back(stmt);
        regs[stmt->val] = stmt->id;
    }

    void visit(AsmLocalStoreStmt *stmt) {
        ir->push_clone_back(stmt);
    }

    void visit(AsmGlobalLoadStmt *stmt) {
        ir->push_clone_back(stmt);
        regs[stmt->val] = stmt->id;
    }

    void visit(AsmGlobalStoreStmt *stmt) {
        ir->push_clone_back(stmt);
    }
};

std::unique_ptr<IR> apply_merge_identical(IR *ir) {
    MergeIdentical visitor;
    visitor.apply(ir);
    return std::move(visitor.ir);
    //return std::make_unique<IR>(*ir);
}

}
