#include <cstdio>
#include <set>
#include <map>
#include <array>
#include <vector>

#define NREGS 4

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

void free_register(Reg *i) {
    printf("free %d\n", i->regid);
}

void alloc_register(Reg *i) {
    printf("alloc %d\n", i->regid);
}

void transit_register(Reg *i, Reg *spill) {
    printf("transit %d <- %d\n", i->regid, spill->regid);
}

void alloc_stack(Reg *i) {
    printf("alloc_stack %d\n", i->regid);
}

void linscan() {
    active.clear();
    for (auto const &i: interval) {
        for (auto const &j: active) {
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

int main() {
    stmts[0].regs = {0};
    stmts[1].regs = {1};
    stmts[2].regs = {2, 0, 1};

    for (auto const &[stmtid, stmt]: stmts) {
        for (auto const &regid: stmt.regs) {
            regs[regid].regid = regid;
            regs[regid].stmts.insert(stmtid);
        }
    }

    for (auto const &[regid, reg]: regs) {
        printf("r%d", regid);
        for (auto const &stmtid: reg.stmts) {
            printf(" s%d", stmtid);
        }
        printf("\n");
    }

    for (auto &[regid, reg]: regs) {
        interval.insert(&reg);
    }

    linscan();
    return 0;
}
