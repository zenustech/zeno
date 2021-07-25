#include "IRVisitor.h"
#include "Stmts.h"
#include <functional>
#include <optional>
#include <cassert>
#include <set>
#include <map>

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
            auto lsp = l->startpoint();
            auto rsp = r->startpoint();
            return lsp == rsp ? l < r : lsp < rsp;
        }
    };

    struct inc_by_end {
        bool operator()(Reg *l, Reg *r) const {
            auto lep = l->endpoint();
            auto rep = r->endpoint();
            return lep == rep ? l < r : lep < rep;
        }
    };

    std::set<Reg *, inc_by_start> interval;
    std::set<Reg *, inc_by_end> active;
    std::map<Reg *, int> usage;
    std::set<int> freed_pool;
    std::set<int> used_pool;
    std::map<int, int> result;
    int memsize = 0;
    int maxregs = 0;

    void free_register(Reg *i) {
        int newid = usage.at(i);
        used_pool.erase(newid);
        freed_pool.insert(newid);
        usage.erase(i);
    }

    void alloc_register(Reg *i) {
        if (!freed_pool.size()) {
            int newreg = used_pool.size();
            maxregs = std::max(maxregs, newreg + 1);
            freed_pool.insert(newreg);
        }
        int newid = *freed_pool.begin();
        used_pool.insert(newid);
        freed_pool.erase(newid);
        usage[i] = newid;
        result[i->regid] = newid;
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
            alloc_register(i);
            active.insert(i);
        }
    }

    void add_usage(int stmtid, int regid) {
        stmts[stmtid].regs.insert(regid);
    }

    void scan() {
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
        if (auto it = result.find(regid); it != result.end()) {
            return it->second;
        }
        error("r%d not found when allocating registers", regid);
        return -1;
    }
};

struct InspectRegisters : Visitor<InspectRegisters> {
    using visit_stmt_types = std::tuple
        < AsmAssignStmt
        , AsmUnaryOpStmt
        , AsmBinaryOpStmt
        , AsmTernaryOpStmt
        , AsmFuncCallStmt
        , AsmLoadConstStmt
        , AsmParamLoadStmt
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

    void visit(AsmTernaryOpStmt *stmt) {
        touch(stmt->id, stmt->dst);
        touch(stmt->id, stmt->cond);
        touch(stmt->id, stmt->lhs);
        touch(stmt->id, stmt->rhs);
    }

    void visit(AsmFuncCallStmt *stmt) {
        touch(stmt->id, stmt->dst);
        for (auto const &arg: stmt->args) {
            touch(stmt->id, arg);
        }
    }

    void visit(AsmLoadConstStmt *stmt) {
        touch(stmt->id, stmt->dst);
    }

    void visit(AsmParamLoadStmt *stmt) {
        touch(stmt->id, stmt->val);
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
        , AsmTernaryOpStmt
        , AsmFuncCallStmt
        , AsmLoadConstStmt
        , AsmParamLoadStmt
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

    void visit(AsmTernaryOpStmt *stmt) {
        reassign(stmt->dst);
        reassign(stmt->cond);
        reassign(stmt->lhs);
        reassign(stmt->rhs);
    }

    void visit(AsmFuncCallStmt *stmt) {
        reassign(stmt->dst);
        for (auto &arg: stmt->args) {
            reassign(arg);
        }
    }

    void visit(AsmLoadConstStmt *stmt) {
        reassign(stmt->dst);
    }

    void visit(AsmParamLoadStmt *stmt) {
        reassign(stmt->val);
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
        , AsmTernaryOpStmt
        , AsmFuncCallStmt
        , AsmLoadConstStmt
        , AsmParamLoadStmt
        , AsmLocalLoadStmt
        , AsmLocalStoreStmt
        , AsmGlobalLoadStmt
        , AsmGlobalStoreStmt
        , Statement
        >;

    const int NREGS;

    explicit FixupMemorySpill(int nregs) : NREGS(nregs) {}

    std::unique_ptr<IR> ir = std::make_unique<IR>();

    struct call_on_dtor : std::function<void()> {
        using std::function<void()>::function;
        call_on_dtor(call_on_dtor const &) = delete;
        ~call_on_dtor() { (*this)(); }
    };

    int memsize = 0;

    std::optional<call_on_dtor> touch(int operandid, int &regid) {
        if (regid >= NREGS) {
            printf("register spilled at %d\n", regid);
            int memid = regid - NREGS;
            memsize = std::max(memsize, memid + 1);
            if (!operandid) {
                int tmpid = NREGS;
                regid = tmpid;
                //ir->emplace_back<AsmLocalStoreStmt>(memid2, tmpid);
                return [=] () {
                    ir->emplace_back<AsmLocalStoreStmt>(memid, tmpid);
                    //ir->emplace_back<AsmLocalLoadStmt>(memid2, tmpid);
                };
            } else {
                int tmpid = NREGS + (operandid - 1);
                regid = tmpid;
                //ir->emplace_back<AsmLocalStoreStmt>(memid2, tmpid);
                ir->emplace_back<AsmLocalLoadStmt>(
                    memid, tmpid);
                //return [=] () {
                    //ir->emplace_back<AsmLocalLoadStmt>(memid2, tmpid);
                //};
            }
        }
        return std::nullopt;
    }

    void visit(Statement *stmt) {
        ir->push_clone_back(stmt);
    }

    void visit(AsmAssignStmt *stmt) {
        touch(1, stmt->src);
        auto _ = touch(0, stmt->dst);
        visit((Statement *)stmt);
    }

    void visit(AsmUnaryOpStmt *stmt) {
        touch(1, stmt->src);
        auto _ = touch(0, stmt->dst);
        visit((Statement *)stmt);
    }

    void visit(AsmBinaryOpStmt *stmt) {
        touch(1, stmt->lhs);
        touch(2, stmt->rhs);
        auto _ = touch(0, stmt->dst);
        visit((Statement *)stmt);
    }

    void visit(AsmTernaryOpStmt *stmt) {
        touch(1, stmt->cond);
        touch(2, stmt->lhs);
        touch(3, stmt->rhs);
        auto _ = touch(0, stmt->dst);
        visit((Statement *)stmt);
    }

    void visit(AsmFuncCallStmt *stmt) {
        for (int i = 0; i < stmt->args.size(); i++) {
            touch(i + 1, stmt->args[i]);
        }
        auto _ = touch(0, stmt->dst);
        visit((Statement *)stmt);
    }

    void visit(AsmLoadConstStmt *stmt) {
        auto _ = touch(0, stmt->dst);
        visit((Statement *)stmt);
    }

    void visit(AsmParamLoadStmt *stmt) {
        auto _ = touch(0, stmt->val);
        visit((Statement *)stmt);
    }

    void visit(AsmLocalLoadStmt *stmt) {
        auto _ = touch(0, stmt->val);
        visit((Statement *)stmt);
    }

    void visit(AsmLocalStoreStmt *stmt) {
        touch(1, stmt->val);
        visit((Statement *)stmt);
    }

    void visit(AsmGlobalLoadStmt *stmt) {
        auto _ = touch(0, stmt->val);
        visit((Statement *)stmt);
    }

    void visit(AsmGlobalStoreStmt *stmt) {
        touch(1, stmt->val);
        visit((Statement *)stmt);
    }
};

int apply_register_allocation(IR *ir, int nregs) {
    if (nregs <= 3) {
        error("no enough registers!\n");
    }
    InspectRegisters inspect;
    UCLAScanner scanner;
    inspect.scanner = &scanner;
    inspect.apply(ir);
    scanner.scan();
    ReassignRegisters reassign;
    reassign.scanner = &scanner;
    reassign.apply(ir);
    int memsize = 0;
    if (scanner.maxregs >= nregs) {
        // left 3 regs for load/store from spilled memory (ternaryop)
        FixupMemorySpill fixspill(nregs - 3);
        fixspill.apply(ir);
        *ir = *fixspill.ir;
        memsize = fixspill.memsize;
    }
    return memsize;
}

}
