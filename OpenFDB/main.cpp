#include <cstdio>
#include <fdb/schedule.h>
#include <fdb/SPGrid.h>
#include <fdb/openvdb.h>

using namespace fdb;

struct AdaptiveGrid {
    spgrid::SPBoolGrid<12> m_lv1;
    spgrid::SPBoolGrid<12> m_lv1_ghost;
    spgrid::SPBoolGrid<11> m_lv2;
    spgrid::SPFloatGrid<12> m_pre1;
    spgrid::SPFloatGrid<11> m_pre2;

    void populate_ghost_cell() {
        m_lv2.foreach(gSerial{}, [&] (auto idx2) {
            ndrange_for(gSerial{}, idx2 * 2, idx2 * 2 + 2, [&] (auto idx1) {
                if (!m_lv1.is_active(idx1)
                    && (m_lv1.is_active(idx1 + vec3i(1, 0, 0))
                    || m_lv1.is_active(idx1 - vec3i(1, 0, 0))
                    || m_lv1.is_active(idx1 + vec3i(0, 1, 0))
                    || m_lv1.is_active(idx1 - vec3i(0, 1, 0))
                    || m_lv1.is_active(idx1 + vec3i(0, 0, 1))
                    || m_lv1.is_active(idx1 - vec3i(0, 0, 1)))
                    ) {
                    m_pre1.at(idx1) = m_pre2.at(idx2);
                    m_lv1_ghost.activate(idx1);
                }
            });
        });
    }

    void accumate_ghost_cell() {
        m_lv2.foreach(gSerial{}, [&] (auto idx2) {
            ndrange_for(gSerial{}, idx2 * 2, idx2 * 2 + 2, [&] (auto idx1) {
                if (!m_lv1.is_active(idx1)
                    && (m_lv1.is_active(idx1 + vec3i(1, 0, 0))
                    || m_lv1.is_active(idx1 - vec3i(1, 0, 0))
                    || m_lv1.is_active(idx1 + vec3i(0, 1, 0))
                    || m_lv1.is_active(idx1 - vec3i(0, 1, 0))
                    || m_lv1.is_active(idx1 + vec3i(0, 0, 1))
                    || m_lv1.is_active(idx1 - vec3i(0, 0, 1)))
                    ) {
                    m_pre1.at(idx1) = m_pre2.at(idx2);
                }
            });
        });
    }
};

int main() {
    spgrid::SPFloatGrid<12> g_pre;
    spgrid::SPFloatGrid<11> g_pre2;

    ndrange_for(gSerial{}, vec3i(-64), vec3i(64), [&] (auto idx) {
        float value = max(0.f, 40.f - length(tofloat(idx)));
        g_pre.set(idx, value);
    });

    ndrange_for(gSerial{}, vec3i(1), vec3i(127), [&] (auto idx) {
        float c = g_pre.get(idx);
    });

    write_dense_vdb("/tmp/a.vdb", [&] (auto idx) {
        return abs(g_pre.get(idx));
    }, vec3i(-64), vec3i(64));

    return 0;
}
