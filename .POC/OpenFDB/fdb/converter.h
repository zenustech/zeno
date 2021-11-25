#pragma once

#include <openvdb/openvdb.h>
#include <openvdb/tree/LeafManager.h>
#include "schedule.h"
#include <vector>


namespace fdb::converter {

template <class OurGridT, class VdbGridT>
void from_vdb_grid(OurGridT &ourGrid, VdbGridT &vdbGrid) {

    //std::mutex mtx;
    auto wrangler = [&](auto &leaf, openvdb::Index leafpos) {
        //std::lock_guard _(mtx);
        for (auto iter = leaf.beginValueOn(); iter != leaf.endValueOn(); ++iter) {
            auto coord = iter.getCoord();
            auto value = iter.getValue();
            ourGrid.set(vec3i(coord[0], coord[1], coord[2]), value);
        }
        for (auto iter = leaf.beginValueOff(); iter != leaf.endValueOff(); ++iter) {
            auto coord = iter.getCoord();
            auto value = iter.getValue();
            ourGrid.set(vec3i(coord[0], coord[1], coord[2]), value);
        }
    };
    using TreeType = std::decay_t<decltype(vdbGrid.tree())>;
    using LeafNodeType = typename TreeType::LeafNodeType;
    openvdb::tree::LeafManager<TreeType> leafman(vdbGrid.tree());
    leafman.foreach(wrangler);

    /*std::vector<LeafNodeType *> nodes;
    vdbGrid.tree().getNodes(nodes);
    for (auto const &node: nodes) {
        node.background();
    }*/
}

template <class VdbGridT>
void clear_vdb_grid(VdbGridT &vdbGrid) {
    auto wrangler = [&](auto &leaf, openvdb::Index leafpos) {
        for (auto iter = leaf.beginValueOn(); iter != leaf.endValueOn(); ++iter) {
            auto valpos = iter.getCoord();
            iter.modifyValue([&] (auto &value) {
                value = {0};
            });
        }
        for (auto iter = leaf.beginValueOff(); iter != leaf.endValueOff(); ++iter) {
            auto valpos = iter.getCoord();
            iter.modifyValue([&] (auto &value) {
                value = {0};
            });
        }
    };
    openvdb::tree::LeafManager<std::decay_t<decltype(vdbGrid.tree())>>
        leafman(vdbGrid.tree());
    leafman.foreach(wrangler);
}

template <class OurGridT, class VdbGridT>
void to_vdb_grid(OurGridT &ourGrid, VdbGridT &vdbGrid) {
    auto vdbAxr = vdbGrid.getAccessor();
    auto wrangler = [&](auto ijk, auto const &value) {
        openvdb::math::Coord coord(ijk[0], ijk[1], ijk[2]);
        //auto vdbValue = vdbAxr.getValue(coord);
        //if (vdbValue != value)
            //printf("%f vs %f\n", vdbValue, value);
        vdbAxr.setValue(coord, value);
    };
    ourGrid.foreach(Serial{}, wrangler); // todo: TBBParallel{}
}

}
