#include <fdb/types.h>
#include <fdb/policy.h>
#include <fdb/VDBGrid.h>
#include <fdb/Stencil.h>
#include <cstdio>

using namespace fdb;

int main() {
    VDBGrid grid;

    grid.add({3, 1, 5});

    fdb::foreach(policy::Serial{}, grid, [&] (auto coor, auto &value) {
        printf("%d %d %d: %f\n", coor[0], coor[1], coor[2], value);
    });
}
