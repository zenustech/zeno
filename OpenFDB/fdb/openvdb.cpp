#include <openvdb/openvdb.h>
#include "openvdb.h"
#include "VDBGrid.h"

template <class VdbGridT, class OurGridT>
void vdb_to_our(VdbGridT &vdbGrid, OurGridT &ourGrid) {
    vdbGrid.foreach([&] (auto leaf, auto coor));

    auto wrangler = [&](auto &leaf, openvdb::Index leafpos) {
        for (auto iter = leaf.beginValueOn(); iter != leaf.endValueOn(); ++iter) {
            auto valpos = iter.getCoord();
            auto value = iter.getValue();
            ourGrid.set(ijk, value);
        }
    };
    auto leafman = openvdb::tree::LeafManager<std::decay_t<decltype(vdbgrid->tree())>>
        (vdbgrid->tree());
    leafman.foreach(wrangler);
}
