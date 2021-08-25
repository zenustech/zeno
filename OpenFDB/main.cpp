#include <fdb/types.h>
#include <fdb/VDBGrid.h>
#include <cstdio>

using namespace fdb;

int main() {
    VDBGrid grid;

    grid.add({3, 1, 5});

    grid.foreach(policy::Serial{}, [&] (auto coor, auto &leaf) {
        printf("%d %d %d\n", coor[0], coor[1], coor[2]);
    });
}
