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
};

std::map<int, Reg> regs;

int main() {
    stmts[0].regs = {0, 1, 2};

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
