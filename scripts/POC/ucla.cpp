#include <cstdio>
#include <set>
#include <map>
#include <array>
#include <vector>

#define NREGS 2

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

void free_register(Reg *i) {
    int newid = usage.at(i);
    used_pool.erase(newid);
    freed_pool.insert(newid);
    printf("free %d: %d\n", i->regid, newid);
}

void alloc_register(Reg *i) {
    int newid = *freed_pool.begin();
    used_pool.insert(newid);
    freed_pool.erase(newid);
    usage[i] = newid;
    printf("alloc %d: %d\n", i->regid, newid);
}

void transit_register(Reg *i, Reg *spill) {
    int newid = usage.at(spill);
    usage[i] = newid;
    usage.erase(spill);
    printf("transit %d <- %d: %d\n", i->regid, spill->regid, newid);
}

void alloc_stack(Reg *i) {
    printf("alloc_stack %d\n", i->regid);
}

void linscan() {
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
            //if (spill->endpoint() > i->endpoint()) {
            transit_register(i, spill);
            alloc_stack(spill);
            active.erase(spill);
            active.insert(i);
            //} else {
                //printf("???\n");
                //alloc_stack(i);
            //}
        } else {
            alloc_register(i);
            active.insert(i);
        }
    }
}

int main() {
    stmts[0].regs = {0};
    stmts[1].regs = {1};
    stmts[2].regs = {2};
    stmts[3].regs = {3};
    stmts[4].regs = {4, 0};
    stmts[5].regs = {5};
    stmts[6].regs = {5};
    stmts[6].regs = {6, 1};
    stmts[7].regs = {7, 2};

    for (int i = 0; i < NREGS; i++) {
        freed_pool.insert(i);
    }

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
