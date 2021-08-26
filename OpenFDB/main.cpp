#include <cstdio>
#include <fdb/schedule.h>
#include <fdb/VDBGrid.h>
#include <fdb/openvdb.h>

using namespace fdb;

int main() {
    vdbgrid::VDBGrid<float> g_pre;

    ndrange_for(Serial{}, vec3i(-64), vec3i(64), [&] (auto idx) {
        float value = max(0.f, 40.f - length(tofloat(idx)));
        g_pre.set(idx, value);
    });

    write_dense_vdb("/tmp/a.vdb", [&] (auto idx) {
        return abs(g_pre.get(idx));
    }, vec3i(-64), vec3i(64));

    return 0;
}
