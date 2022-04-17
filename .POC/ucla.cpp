#include <cstdio>
#include <set>
#include <map>
#include <array>
#include <vector>

#define NREGS 2

struct UCLALinearScan {
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

    void linear_scan() {
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

    auto const &apply() {
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

        linear_scan();
        return result;
    }
};

int main() {
    UCLALinearScan scanner;

    auto &stmts = scanner.stmts;
    stmts[0].regs = {0};
    stmts[1].regs = {1};
    stmts[2].regs = {2};
    stmts[3].regs = {3};
    stmts[4].regs = {4, 0};
    stmts[5].regs = {5};
    stmts[6].regs = {5};
    stmts[6].regs = {6, 1};
    stmts[7].regs = {7, 2};

    auto const &result = scanner.apply();

    for (auto &[stmtid, stmt]: stmts) {
        printf("$%d :", stmtid);
        for (auto &regid: stmt.regs) {
            int newid = result.at(regid);
            if (newid != -1) {
                printf(" r%d", newid);
            } else {
                printf(" [%d]", regid);
            }
        }
        printf("\n");
    }

    return 0;
}
