#include <zeno/zeno.h>
#include <zeno/VDBGrid.h>
#include <zeno/types/NumericObject.h>
#include <openvdb/openvdb.h>
#include <openvdb/tree/LeafManager.h>
#include <openvdb/points/PointCount.h>
#include <openvdb/points/PointAdvect.h>
#include <openvdb/tools/Interpolation.h>
#include <openvdb/tools/ChangeBackground.h>

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
    auto grid = safe_uniqueptr_cast<VDBGrid>(clone_input("grid"));
    if (auto p = safe_dynamic_cast<VDBFloatGrid>(grid.get()); p) {
        auto velman = openvdb::tree::LeafManager
            <std::decay_t<decltype(p->m_grid->tree())>>(p->m_grid->tree());
        velman.foreach(fill_voxels_op( get_input2_float("fillValue")));
    } else if (auto p = safe_dynamic_cast<VDBFloat3Grid>(grid.get()); p) {
        auto velman = openvdb::tree::LeafManager
            <std::decay_t<decltype(p->m_grid->tree())>>(p->m_grid->tree());
        velman.foreach(fill_voxels_op(vec_to_other<openvdb::Vec3f>(toVec3f(get_input2_vec3f("fillValue")))));
    }

    set_output("grid", std::move(grid));
  }
};

ZENO_DEFNODE(VDBFillActiveVoxels)(
     { /* inputs: */ {
     {gParamType_VDBGrid, "grid", "", zeno::Socket_ReadOnly},
     {gParamType_Float, "fillValue", "0.0"},
     }, /* outputs: */ {
         {gParamType_VDBGrid, "grid"},
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
    auto grid = safe_uniqueptr_cast<VDBGrid>(clone_input("grid"));
    auto value = get_input<NumericObject>("fillValue")->value;
    if (auto p = safe_dynamic_cast<VDBFloatGrid>(grid.get()); p) {
        auto velman = openvdb::tree::LeafManager
            <std::decay_t<decltype(p->m_grid->tree())>>(p->m_grid->tree());
        velman.foreach(fill_voxels_op(std::get<float>(value)));
    } else if (auto p = safe_dynamic_cast<VDBFloat3Grid>(grid.get()); p) {
        auto velman = openvdb::tree::LeafManager
            <std::decay_t<decltype(p->m_grid->tree())>>(p->m_grid->tree());
        velman.foreach(fill_voxels_op(vec_to_other<openvdb::Vec3f>(std::get<zeno::vec3f>(value))));
    }

    set_output("grid", std::move(grid));
  }
};

ZENO_DEFNODE(VDBMultiplyOperation)(
     { /* inputs: */ {
     "grid",
     {gParamType_Float, "fillValue", "0.0"},
     }, /* outputs: */ {
       "grid",
     }, /* params: */ {
     }, /* category: */ {
     "openvdb",
     }});
#endif


template <class GridPtr>
void touch_aabb_region(GridPtr const &grid, zeno::vec3f const &bmin, zeno::vec3f const &bmax) {
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
    auto grid = safe_uniqueptr_cast<VDBGrid>(clone_input("grid"));
    auto bmin = toVec3f(get_input2_vec3f("bmin"));
    auto bmax = toVec3f(get_input2_vec3f("bmax"));
    if (auto p = safe_dynamic_cast<VDBFloatGrid>(grid.get()); p) {
        touch_aabb_region(p->m_grid, bmin, bmax);
    } else if (auto p = safe_dynamic_cast<VDBFloat3Grid>(grid.get()); p) {
        touch_aabb_region(p->m_grid, bmin, bmax);
    }

    set_output("grid", std::move(grid));
  }
};

ZENO_DEFNODE(VDBTouchAABBRegion)(
     { /* inputs: */ {
     {gParamType_VDBGrid, "grid", "", zeno::Socket_ReadOnly},
     {gParamType_Vec3f, "bmin", "-1,-1,-1"},
     {gParamType_Vec3f, "bmax", "1,1,1"},
     }, /* outputs: */ {
         {gParamType_VDBGrid, "grid"},
     }, /* params: */ {
     }, /* category: */ {
     "openvdb",
     }});


struct VDBTopoCopy : INode{
  virtual void apply() override {
    auto grid = safe_uniqueptr_cast<VDBGrid>(clone_input("grid"));
    auto topo = safe_uniqueptr_cast<VDBGrid>(clone_input("topo"));
    if (auto p = safe_dynamic_cast<VDBFloatGrid>(grid.get()); p) {
        if(auto t = safe_dynamic_cast<VDBFloatGrid>(topo.get()); t)
        {
            p->m_grid->setTree(std::make_shared<openvdb::FloatTree>(t->m_grid->tree(),0, openvdb::TopologyCopy()));
            openvdb::tools::dilateActiveValues(
            p->m_grid->tree(), 1,
            openvdb::tools::NearestNeighbors::NN_FACE_EDGE_VERTEX, openvdb::tools::TilePolicy::EXPAND_TILES);
        }
        else if (auto t = safe_dynamic_cast<VDBFloat3Grid>(topo.get()); t)
        {
            p->m_grid->setTree(std::make_shared<openvdb::FloatTree>(t->m_grid->tree(),0, openvdb::TopologyCopy()));
            openvdb::tools::dilateActiveValues(
            p->m_grid->tree(), 1,
            openvdb::tools::NearestNeighbors::NN_FACE_EDGE_VERTEX, openvdb::tools::TilePolicy::EXPAND_TILES);
        }
    } else if (auto p = safe_dynamic_cast<VDBFloat3Grid>(grid.get()); p) {
        if(auto t = safe_dynamic_cast<VDBFloatGrid>(topo.get()); t)
        {
            p->m_grid->setTree(std::make_shared<openvdb::Vec3fTree>(t->m_grid->tree(), openvdb::Vec3f{0}, openvdb::TopologyCopy()));
            openvdb::tools::dilateActiveValues(
            p->m_grid->tree(), 1,
            openvdb::tools::NearestNeighbors::NN_FACE_EDGE_VERTEX, openvdb::tools::TilePolicy::EXPAND_TILES);
        }
        else if (auto t = safe_dynamic_cast<VDBFloat3Grid>(topo.get()); t)
        {
            p->m_grid->setTree(std::make_shared<openvdb::Vec3fTree>(t->m_grid->tree(), openvdb::Vec3f{0}, openvdb::TopologyCopy()));
            openvdb::tools::dilateActiveValues(
            p->m_grid->tree(), 1,
            openvdb::tools::NearestNeighbors::NN_FACE_EDGE_VERTEX, openvdb::tools::TilePolicy::EXPAND_TILES);
        }
    }


    set_output("grid", std::move(grid));
  }
};
ZENO_DEFNODE(VDBTopoCopy)(
     { /* inputs: */ {
     {gParamType_VDBGrid, "grid", "", zeno::Socket_ReadOnly},
     {gParamType_VDBGrid, "topo", "", zeno::Socket_ReadOnly}
     }, /* outputs: */ {
         {gParamType_VDBGrid, "grid"},
     }, /* params: */ {
     }, /* category: */ {
     "openvdb",
     }});

}
