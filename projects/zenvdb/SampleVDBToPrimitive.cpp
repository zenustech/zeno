#include <zeno/zeno.h>
#include <zeno/VDBGrid.h>
#include <zeno/NumericObject.h>
#include <zeno/PrimitiveObject.h>

namespace zeno {

template <class T>
struct attr_to_vdb_type {
};

template <>
struct attr_to_vdb_type<float> {
    using type = VDBFloatGrid;
};

template <>
struct attr_to_vdb_type<vec3f> {
    using type = VDBFloat3Grid;
};

template <>
struct attr_to_vdb_type<int> {
    using type = VDBIntGrid;
};

template <>
struct attr_to_vdb_type<vec3i> {
    using type = VDBInt3Grid;
};

template <class T>
using attr_to_vdb_type_t = typename attr_to_vdb_type<T>::type;

template <class T>
void sampleVDBAttribute(std::vector<vec3f> const &pos, std::vector<T> &arr,
    VDBGrid *ggrid, std::string const &attr) {
    using VDBType = attr_to_vdb_type_t<T>;
    auto ptr = dynamic_cast<VDBType *>(ggrid);
    if (!ptr) {
        printf("ERROR: vdb attribute type mismatch!\n");
        return;
    }
    auto grid = ptr->m_grid;
}

struct SampleVDBToPrimitive : INode {
    virtual void apply() override {
        auto prim = get_input<PrimitiveObject>("prim");
        auto grid = get_input<VDBGrid>("grid");
        auto attrPrim = get_param<std::string>("attrPrim");
        auto attrGrid = get_param<std::string>("attrGrid");
        auto &pos = prim->attr<vec3f>("pos");

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
