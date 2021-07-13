#include <cstdio>
#include <set>
#include <map>
#include <array>
#include <vector>

struct Stmt {
    std::set<int> regs;
};

std::map<int, Stmt> stmts;

struct Reg {
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

int main() {
    stmts[0].regs = {0};
    stmts[1].regs = {1};
    stmts[2].regs = {2, 0, 1};

    for (auto const &[stmtid, stmt]: stmts) {
        for (auto const &regid: stmt.regs) {
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
}
