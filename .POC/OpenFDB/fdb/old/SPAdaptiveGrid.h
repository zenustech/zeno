#pragma once

#include "SPGrid.h"

namespace fdb::spgrid {

struct SPAdaptiveGrid {
    SPBoolGrid<12> m_lv1;
    SPBoolGrid<12> m_lv1_ghost;
    SPBoolGrid<11> m_lv2;
    SPFloatGrid<12> m_pre1;
    SPFloatGrid<11> m_pre2;

    void populate_ghost_cell() {
        m_lv2.foreach(Serial{}, [&] (auto idx2) {
            ndrange_for(Serial{}, idx2 * 2, idx2 * 2 + 2, [&] (auto idx1) {
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
        m_lv2.foreach(Serial{}, [&] (auto idx2) {
            ndrange_for(Serial{}, idx2 * 2, idx2 * 2 + 2, [&] (auto idx1) {
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

}
