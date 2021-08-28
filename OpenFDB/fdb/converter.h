#pragma once

#include <openvdb/openvdb.h>
#include "schedule.h"


namespace fdb::converter {

template <class OurGridT, class VdbGridT>
void from_vdb_grid(OurGridT &ourGrid, VdbGridT const &vdbGrid) {
    auto wrangler = [&](auto &leaf, openvdb::Index leafpos) {
        for (auto iter = leaf.beginValueOn(); iter != leaf.endValueOn(); ++iter) {
            auto valpos = iter.getCoord();
            auto value = iter.getValue();
            ourGrid.set(ijk, value);
        }
    };
    auto leafman = openvdb::tree::LeafManager<std::decay_t<decltype(vdbgrid.tree())>>
        (vdbgrid.tree());
    leafman.foreach(wrangler);
}

template <class OurGridT, class VdbGridT>
void to_vdb_grid(OurGridT const &ourGrid, VdbGridT &vdbGrid) {
    auto vdbAxr = vdbGrid.getAccessor();
    auto wrangler = [&](auto ijk, auto const &value) {
        vdbAxr.setValue(ijk, value);
    };
    ourGrid.foreach(Serial{}, wrangler);
}

}
