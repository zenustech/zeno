#include "IRVisitor.h"
#include "Stmts.h"
#include <set>
#include <map>

#define NREGS 8

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
        result[i->regid] = -1;
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

    void touch(int stmtid, int &regid) {
        regid = scanner->lookup(regid);
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
    UCLAScanner scanner;
    InspectRegisters inspect;
    inspect.scanner = &scanner;
    inspect.apply(ir);
    scanner.scan();
    ReassignRegisters reassign;
    reassign.scanner = &scanner;
    reassign.apply(ir);
}

}
