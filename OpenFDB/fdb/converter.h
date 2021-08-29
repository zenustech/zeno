#pragma once

#include <openvdb/openvdb.h>
#include <openvdb/tree/LeafManager.h>
#include "schedule.h"


namespace fdb::converter {

template <class OurGridT, class VdbGridT>
void from_vdb_grid(OurGridT &ourGrid, VdbGridT &vdbGrid) {
    auto wrangler = [&](auto &leaf, openvdb::Index leafpos) {
        for (auto iter = leaf.beginValueOn(); iter != leaf.endValueOn(); ++iter) {
            auto valpos = iter.getCoord();
            auto value = iter.getValue();
            ourGrid.set(vec3i(valpos[0], valpos[1], valpos[2]), value);
        }
    };
    openvdb::tree::LeafManager<std::decay_t<decltype(vdbGrid.tree())>>
        leafman(vdbGrid.tree());
    leafman.foreach(wrangler);
}

template <class OurGridT, class VdbGridT>
void to_vdb_grid(OurGridT const &ourGrid, VdbGridT &vdbGrid) {
    auto vdbAxr = vdbGrid.getAccessor();
    auto wrangler = [&](auto ijk, auto const &value) {
        vdbAxr.setValue(openvdb::math::Coord(ijk[0], ijk[1], ijk[2]), value);
    };
    ourGrid.foreach(Serial{}, wrangler);
}

}
