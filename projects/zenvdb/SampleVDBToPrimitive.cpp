#if 0
#include <zeno/zeno.h>
#include <zeno/VDBGrid.h>
#include <zeno/NumericObject.h>
#include <zeno/PrimitiveObject.h>
#include <openvdb/openvdb.h>
#include <openvdb/tree/LeafManager.h>

namespace zeno {

template <class T, class S>
void sampleVDBAttribute(std::vector<zen::vec3f> const &pos, std::vector<T> &arr,
    VDBGrid *ggrid, std::string const &attr) {
    //dynamic_cast<VDBFloatGrid>(ggrid);
}

struct SampleVDBToPrimitive : INode {
    virtual void apply() override {
        auto prim = get_input<PrimitiveObject>("prim");
        auto grid = get_input<VDBGrid>("grid");
        auto attrPrim = get_param<std::string>("attrPrim");
        auto attrGrid = get_param<std::string>("attrGrid");
        auto &pos = prim->attr<zen::vec3f>("pos");

        std::visit([&] (auto &vel) {
            sampleVDBAttribute(pos, vel, grid.get(), attrGrid);
        }, prim->attr(attrPrim));

        set_output("prim", std::move(prim));
    }
};

ZENDEFNODE(SampleVDBToPrimitive, {
    {"prim", "vdbGrid"},
    {"prim"},
    {{"string", "attrPrim", "vel"}, {"string", "attrGrid", "v"}},
    {"openvdb"},
});


}

#endif
