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

// http://web.cs.ucla.edu/~palsberg/course/cs132/linearscan.pdf, page 899 (??)
namespace ucla {
    int n;
    std::vector<int> start;
    std::vector<int> end;

    struct inc_start {
        bool operator()(int i, int j) const {
            return start[i] < start[j];
        }
    };

    struct inc_end {
        bool operator()(int i, int j) const {
            return end[i] < end[j];
        }
    };

    std::set<int, inc_start> interval;
    std::set<int, inc_end> active;

    std::map<int, int> usage;
    std::array<bool, NREGS> used;
    std::map<int, int> storage;

    void initialize() {
        interval.clear();
        active.clear();
        usage.clear();
        for (int i = 0; i < NREGS; i++) {
            used[i] = false;
        }
        storage.clear();
        start.clear();
        end.clear();
        n = 0;
    }

    void expire_old(int i) {
        for (int j = 0; j < n; j++) {
            if (end[j] >= start[i]) {
                return;
            }
            active.erase(j);
            printf("free register %d\n", j);
        }
    }

    int alloc_reg(int i) {
        int r;
        for (r = 0; r < NREGS; r++)
            if (!used[r])
                break;
        assert(r < NREGS);
        used[r] = true;
        usage[i] = r;
        return r;
    }

    void free_reg(int spill) {
        assert(used[spill]);
        used[spill] = false;
    }

    void spill_at(int i) {
        int spill = *active.rbegin();
        if (end[spill] > end[i]) {
            free_reg(spill);
            int regid = alloc_reg(i);
            printf("need store %d\n", spill);
            storage[spill] = storage.size();
            printf("$%d -> r%d\n", i, regid);
            active.erase(spill);
            active.insert(i);
        } else {
            printf("need store %d\n", i);
            storage[i] = storage.size();
        }
    }

    void linear_scan() {
        active.clear();
        for (int i = 0; i < n; i++) {
            expire_old(i);
            if (active.size() == NREGS) {
                spill_at(i);
            } else {
                int regid = alloc_reg(i);
                printf("$%d -> r%d\n", i, regid);
                active.insert(i);
            }
        }
    }
};

struct CalcLiveIntervals : Visitor<CalcLiveIntervals> {
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

    struct Range {
        int beg = -1, end = -1;

        void hit(int i) {
            if (beg == -1) {
                beg = end = i;
            } else {
                beg = std::min(beg, i);
                end = std::max(end, i);
            }
        }
    };

    std::vector<Range> live_ranges;

    void touch(int stmtid, int regid) {
        if (live_ranges.size() < regid + 1)
            live_ranges.resize(regid + 1);
        live_ranges[regid].hit(stmtid);
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

struct RegisterAllocation : Visitor<RegisterAllocation> {
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

    std::unique_ptr<IR> ir;

    std::map<int, int> usage;
    std::array<int, NREGS> used;
    std::map<int, int> storage;

    RegisterAllocation() {
        for (int i = 0; i < NREGS; i++) {
            used[i] = -1;
        }
    }

    void load(int stmtid, int &regid) {
        stmtid = regid;
        regid = usage.at(stmtid);
        if (used[regid] != -1 && used[regid] != stmtid) {
            printf("load %d %d\n", regid, used[regid]);
        }
        used[regid] = stmtid;
    }

    void store(int stmtid, int &regid) {
        stmtid = regid;
        regid = usage.at(stmtid);
        if (used[regid] != -1 && used[regid] != stmtid) {
            printf("store %d %d\n", regid, used[regid]);
        }
        used[regid] = stmtid;
    }

    void visit(AsmAssignStmt *stmt) {
        store(stmt->id, stmt->dst);
        load(stmt->id, stmt->src);
    }

    void visit(AsmUnaryOpStmt *stmt) {
        store(stmt->id, stmt->dst);
        load(stmt->id, stmt->src);
    }

    void visit(AsmBinaryOpStmt *stmt) {
        store(stmt->id, stmt->dst);
        load(stmt->id, stmt->lhs);
        load(stmt->id, stmt->rhs);
    }

    void visit(AsmLoadConstStmt *stmt) {
        store(stmt->id, stmt->dst);
    }

    void visit(AsmLocalLoadStmt *stmt) {
        store(stmt->id, stmt->val);
    }

    void visit(AsmLocalStoreStmt *stmt) {
        load(stmt->id, stmt->val);
    }

    void visit(AsmGlobalLoadStmt *stmt) {
        store(stmt->id, stmt->val);
    }

    void visit(AsmGlobalStoreStmt *stmt) {
        load(stmt->id, stmt->val);
    }

    void visit(Statement *stmt) {
        ir->push_clone_back(stmt);
    }
};

void apply_register_allocation(IR *ir) {
    CalcLiveIntervals calc;
    calc.apply(ir);

    ucla::initialize();
    for (auto const &range: calc.live_ranges) {
        ucla::start.push_back(range.beg);
        ucla::end.push_back(range.end);
    }
    ucla::n = ucla::start.size();
    ucla::linear_scan();

    //return std::make_unique<IR>(*ir);
    RegisterAllocation alloc;
    alloc.storage = ucla::storage;
    alloc.usage = ucla::usage;
    alloc.apply(ir);
}

}
