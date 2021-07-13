#include "IRVisitor.h"
#include "Stmts.h"
#include <set>
#include <map>

// let's left two regs for load/store from spilled memory:
inline constexpr int NREGS = 8 - 2;

namespace zfx {

// http://web.cs.ucla.edu/~palsberg/course/cs132/linearscan.pdf
struct UCLAScanner {
    struct Stmt {
        std::set<int> regs;
    };

    std::map<int, Stmt> stmts;

    struct Reg {
        int regid;
        std::set<int> stmts;

        int startpoint() const {
            return *stmts.begin();
        }

        int endpoint() const {
            return *stmts.rbegin();
        }
    };

    std::map<int, Reg> regs;

    struct inc_by_start {
        bool operator()(Reg *l, Reg *r) const {
            return l->startpoint() < r->startpoint();
        }
    };

    struct inc_by_end {
        bool operator()(Reg *l, Reg *r) const {
            return l->endpoint() < r->endpoint();
        }
    };

    std::set<Reg *, inc_by_start> interval;
    std::set<Reg *, inc_by_end> active;
    std::map<Reg *, int> usage;
    std::set<int> freed_pool;
    std::set<int> used_pool;
    std::map<int, int> result;
    int memsize = 0;

    void free_register(Reg *i) {
        int newid = usage.at(i);
        used_pool.erase(newid);
        freed_pool.insert(newid);
    }

    void alloc_register(Reg *i) {
        int newid = *freed_pool.begin();
        used_pool.insert(newid);
        freed_pool.erase(newid);
        usage[i] = newid;
        result[i->regid] = newid;
    }

    void transit_register(Reg *i, Reg *spill) {
        int newid = usage.at(spill);
        usage[i] = newid;
        usage.erase(spill);
        result[i->regid] = newid;
    }

    void alloc_stack(Reg *i) {
        result[i->regid] = NREGS + memsize++;
    }

    void do_scan() {
        active.clear();
        for (auto const &i: interval) {
            for (auto const &j: std::set(active)) {
                if (j->endpoint() >= i->startpoint()) {
                    break;
                }
                active.erase(j);
                free_register(j);
            }
            if (active.size() == NREGS) {
                auto spill = *active.begin();
                if (spill->endpoint() > i->endpoint()) {
                    transit_register(i, spill);
                    alloc_stack(spill);
                    active.erase(spill);
                    active.insert(i);
                } else {
                    alloc_stack(i);
                }
            } else {
                alloc_register(i);
                active.insert(i);
            }
        }
    }

    void add_usage(int stmtid, int regid) {
        stmts[stmtid].regs.insert(regid);
    }

    void scan() {
        for (int i = 0; i < NREGS; i++) {
            freed_pool.insert(i);
        }

        for (auto const &[stmtid, stmt]: stmts) {
            for (auto const &regid: stmt.regs) {
                regs[regid].regid = regid;
                regs[regid].stmts.insert(stmtid);
            }
        }

        for (auto &[regid, reg]: regs) {
            interval.insert(&reg);
        }

        do_scan();
    }

    int lookup(int regid) {
        return result.at(regid);
    }
};

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

    UCLAScanner *scanner;

    void touch(int stmtid, int regid) {
        scanner->add_usage(stmtid, regid);
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

struct ReassignRegisters : Visitor<ReassignRegisters> {
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

    UCLAScanner *scanner;

    void reassign(int &regid) {
        regid = scanner->lookup(regid);
    }

    void visit(AsmAssignStmt *stmt) {
        reassign(stmt->dst);
        reassign(stmt->src);
    }

    void visit(AsmUnaryOpStmt *stmt) {
        reassign(stmt->dst);
        reassign(stmt->src);
    }

    void visit(AsmBinaryOpStmt *stmt) {
        reassign(stmt->dst);
        reassign(stmt->lhs);
        reassign(stmt->rhs);
    }

    void visit(AsmLoadConstStmt *stmt) {
        reassign(stmt->dst);
    }

    void visit(AsmLocalLoadStmt *stmt) {
        reassign(stmt->val);
    }

    void visit(AsmLocalStoreStmt *stmt) {
        reassign(stmt->val);
    }

    void visit(AsmGlobalLoadStmt *stmt) {
        reassign(stmt->val);
    }

    void visit(AsmGlobalStoreStmt *stmt) {
        reassign(stmt->val);
    }
};

struct FixupMemorySpill : Visitor<FixupMemorySpill> {
    using visit_stmt_types = std::tuple
        < AsmAssignStmt
        , AsmUnaryOpStmt
        , AsmBinaryOpStmt
        , AsmLoadConstStmt
        , AsmLocalLoadStmt
        , AsmLocalStoreStmt
        , AsmGlobalLoadStmt
        , AsmGlobalStoreStmt
        , Statement
        >;

    std::unique_ptr<IR> ir = std::make_unique<IR>();

    void touch(int operandid, int &regid) {
        if (regid >= NREGS) {
            printf("register spilled at %d\n", regid);
            int memid = regid - NREGS;
            if (!operandid) {
                int tmpid = NREGS;
                ir->emplace_back<AsmLocalStoreStmt>(
                    memid, tmpid);
                regid = tmpid;
            } else {
                int tmpid = NREGS + (operandid - 1);
                ir->emplace_back<AsmLocalLoadStmt>(
                    memid, tmpid);
                regid = tmpid;
            }
        }
    }

    void visit(Statement *stmt) {
        ir->push_clone_back(stmt);
    }

    void visit(AsmAssignStmt *stmt) {
        touch(1, stmt->src);
        visit((Statement *)stmt);
        touch(0, stmt->dst);
    }

    void visit(AsmUnaryOpStmt *stmt) {
        touch(1, stmt->src);
        visit((Statement *)stmt);
        touch(0, stmt->dst);
    }

    void visit(AsmBinaryOpStmt *stmt) {
        touch(1, stmt->lhs);
        touch(2, stmt->rhs);
        visit((Statement *)stmt);
        touch(0, stmt->dst);
    }

    void visit(AsmLoadConstStmt *stmt) {
        visit((Statement *)stmt);
        touch(0, stmt->dst);
    }

    void visit(AsmLocalLoadStmt *stmt) {
        visit((Statement *)stmt);
        touch(0, stmt->val);
    }

    void visit(AsmLocalStoreStmt *stmt) {
        touch(1, stmt->val);
        visit((Statement *)stmt);
    }

    void visit(AsmGlobalLoadStmt *stmt) {
        visit((Statement *)stmt);
        touch(0, stmt->val);
    }

    void visit(AsmGlobalStoreStmt *stmt) {
        touch(1, stmt->val);
        visit((Statement *)stmt);
    }
};

void apply_register_allocation(IR *ir) {
    UCLAScanner scanner;
    InspectRegisters inspect;
    inspect.scanner = &scanner;
    inspect.apply(ir);
    scanner.scan();
    ReassignRegisters reassign;
    reassign.scanner = &scanner;
    reassign.apply(ir);
    FixupMemorySpill fixspill;
    fixspill.apply(ir);
    *ir = *fixspill.ir;
}

}
