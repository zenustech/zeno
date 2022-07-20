#include <zeno/zeno.h>
#include <zeno/VDBGrid.h>
#include <zeno/types/NumericObject.h>
#include <openvdb/openvdb.h>
#include <openvdb/tree/LeafManager.h>
#include <openvdb/points/PointCount.h>
#include <openvdb/points/PointAdvect.h>
#include <openvdb/tools/Interpolation.h>

namespace {

using namespace zeno;


template <class T>
struct fill_voxels_op {
    T value;
    fill_voxels_op(T const &val) : value(val) {}

    template <class LeafT>
    void operator()(LeafT &leaf, openvdb::Index leafpos) const {
        for (auto iter = leaf.beginValueOn(); iter != leaf.endValueOn(); ++iter) {
            iter.setValue(value);
        }
    }
};
struct VDBFillActiveVoxels : INode {
  virtual void apply() override {
    auto grid = get_input<VDBGrid>("grid");
    auto value = get_input<NumericObject>("fillValue")->value;
    if (auto p = std::dynamic_pointer_cast<VDBFloatGrid>(grid); p) {
        auto velman = openvdb::tree::LeafManager
            <std::decay_t<decltype(p->m_grid->tree())>>(p->m_grid->tree());
        velman.foreach(fill_voxels_op(std::get<float>(value)));
    } else if (auto p = std::dynamic_pointer_cast<VDBFloat3Grid>(grid); p) {
        auto velman = openvdb::tree::LeafManager
            <std::decay_t<decltype(p->m_grid->tree())>>(p->m_grid->tree());
        velman.foreach(fill_voxels_op(vec_to_other<openvdb::Vec3f>(std::get<vec3f>(value))));
    }

    set_output("grid", get_input("grid"));
  }
};

ZENO_DEFNODE(VDBFillActiveVoxels)(
     { /* inputs: */ {
     "grid",
     {"NumericObject", "fillValue", "0.0"},
     }, /* outputs: */ {
       "grid",
     }, /* params: */ {
     }, /* category: */ {
     "openvdb",
     }});


#if 0
struct multiply_voxels_op {
    T value;
    fill_voxels_op(T const &val) : value(val) {}

    template <class LeafT>
    void operator()(LeafT &leaf, openvdb::Index leafpos) const {
        for (auto iter = leaf.beginValueOn(); iter != leaf.endValueOn(); ++iter) {
            iter.setValue(value);
        }
    }
};

struct VDBMultiplyOperation : INode {
  virtual void apply() override {
    auto grid = get_input<VDBGrid>("grid");
    auto value = get_input<NumericObject>("fillValue")->value;
    if (auto p = std::dynamic_pointer_cast<VDBFloatGrid>(grid); p) {
        auto velman = openvdb::tree::LeafManager
            <std::decay_t<decltype(p->m_grid->tree())>>(p->m_grid->tree());
        velman.foreach(fill_voxels_op(std::get<float>(value)));
    } else if (auto p = std::dynamic_pointer_cast<VDBFloat3Grid>(grid); p) {
        auto velman = openvdb::tree::LeafManager
            <std::decay_t<decltype(p->m_grid->tree())>>(p->m_grid->tree());
        velman.foreach(fill_voxels_op(vec_to_other<openvdb::Vec3f>(std::get<vec3f>(value))));
    }

    set_output("grid", get_input("grid"));
  }
};

ZENO_DEFNODE(VDBMultiplyOperation)(
     { /* inputs: */ {
     "grid",
     {"NumericObject", "fillValue", "0.0"},
     }, /* outputs: */ {
       "grid",
     }, /* params: */ {
     }, /* category: */ {
     "openvdb",
     }});
#endif


template <class GridPtr>
void touch_aabb_region(GridPtr const &grid, vec3f const &bmin, vec3f const &bmax) {
    auto cmin = grid->transform().worldToIndex(openvdb::Vec3R(bmin[0], bmin[1], bmin[2]));
    auto cmax = grid->transform().worldToIndex(openvdb::Vec3R(bmax[0], bmax[1], bmax[2]));
    using size_type = std::decay_t<decltype(std::declval<openvdb::Coord>()[0])>;

    //std::mutex mtx;
    //tbb::parallel_for(tbb::blocked_range<size_type>(cmin[2], cmax[2]), [&] (auto const &r) {
        //std::lock_guard _(mtx);
        //for (size_type z = r.begin(); z < r.end(); z++) {
        auto axr = grid->getAccessor();
        for (size_type z = cmin[2]; z < cmax[2]; z++) {
            for (size_type y = cmin[1]; y < cmax[1]; y++) {
                for (size_type x = cmin[0]; x < cmax[0]; x++) {
                    using value_type = std::decay_t<decltype(axr.getValue({x, y, z}))>;
                    axr.setValue({x, y, z}, value_type(0));
                }
            }
        }
    //});
}

struct VDBTouchAABBRegion : INode {
  virtual void apply() override {
    auto grid = get_input<VDBGrid>("grid");
    auto bmin = get_input<NumericObject>("bmin")->get<vec3f>();
    auto bmax = get_input<NumericObject>("bmax")->get<vec3f>();
    if (auto p = std::dynamic_pointer_cast<VDBFloatGrid>(grid); p) {
        touch_aabb_region(p->m_grid, bmin, bmax);
    } else if (auto p = std::dynamic_pointer_cast<VDBFloat3Grid>(grid); p) {
        touch_aabb_region(p->m_grid, bmin, bmax);
    }

    set_output("grid", get_input("grid"));
  }
};

ZENO_DEFNODE(VDBTouchAABBRegion)(
     { /* inputs: */ {
     "grid",
     {"vec3f", "bmin", "-1,-1,-1"},
     {"vec3f", "bmax", "1,1,1"},
     }, /* outputs: */ {
       "grid",
     }, /* params: */ {
     }, /* category: */ {
     "openvdb",
     }});



}
