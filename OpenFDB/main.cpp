#include <cstdint>
#include <cstddef>
#include <fdb/types.h>
#include <fdb/VDBGrid.h>

using namespace fdb;

int main() {
    VDBGrid grid;

    grid.foreach(policy::Serial{}, [&] (auto coor, auto &leaf) {
        grid.get(coor + 1);
    });
}
