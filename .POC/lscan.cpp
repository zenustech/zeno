#include <set>
#include <map>
#include <array>
#include <cstdio>
#include <cassert>

#define NREGS 2

int n;
int start[233];
int end[233];

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
std::set<int> stored;

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
    used[r] = 1;
    usage[i] = r;
    return r;
}

void free_reg(int spill) {
    assert(used[spill]);
    used[spill] = 0;
}

void spill_at(int i) {
    int spill = *active.rbegin();
    if (end[spill] > end[i]) {
        free_reg(spill);
        int regid = alloc_reg(i);
        printf("need store %d\n", spill);
        stored.insert(spill);
        printf("$%d -> r%d\n", i, regid);
        active.erase(spill);
        active.insert(i);
    } else {
        printf("need store %d\n", i);
        stored.insert(i);
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

int main(void) {
    n = 4;
    start[0] = 0; end[0] = 4;
    start[1] = 3; end[1] = 8;
    start[2] = 1; end[2] = 5;
    start[3] = 3; end[3] = 6;
    linear_scan();
    return 0;
}
