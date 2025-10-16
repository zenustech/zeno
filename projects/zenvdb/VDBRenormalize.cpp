#include <zeno/zeno.h>
#include <zeno/NumericObject.h>
#include <zeno/StringObject.h>
#include <vector>
#include <zeno/VDBGrid.h>
#include <openvdb/points/PointCount.h>
#include <openvdb/tree/LeafManager.h>
#include <openvdb/points/PointAdvect.h>
#include <openvdb/tools/Morphology.h>
#include <openvdb/tools/MeshToVolume.h>
#include <openvdb/tools/LevelSetTracker.h>
#include <openvdb/tools/Filter.h>
#include <openvdb/openvdb.h>
#include <openvdb/tools/LevelSetSphere.h>
#include <openvdb/tools/ChangeBackground.h>

namespace zeno {
    struct  VDBRenormalizeSDF : zeno::INode {
  virtual void apply() override {

    auto inoutSDF = safe_uniqueptr_cast<VDBFloatGrid>(clone_input("inoutSDF"));
    int normIter = get_param_int("iterations");
    int dilateIter = get_param_int("dilateIters");
    auto lstracker = openvdb::tools::LevelSetTracker<openvdb::FloatGrid>(*(inoutSDF->m_grid));
    lstracker.setState({openvdb::math::FIRST_BIAS, openvdb::math::TVD_RK3, 1, 1});
    lstracker.setTrimming(openvdb::tools::lstrack::TrimMode::kNone);

    if (dilateIter > 0)
        lstracker.dilate(dilateIter);
    else if (dilateIter < 0)
        lstracker.erode(dilateIter);
    for(int i=0;i<normIter;i++)
        lstracker.normalize();
    //openvdb::tools::changeBackground(inoutSDF->m_grid->tree(), ((float)normIter)*(inoutSDF->m_grid->transformPtr()->voxelSize()[0]));
    //openvdb::tools::signedFloodFill(inoutSDF->m_grid->tree());

    set_output("inoutSDF", std::move(inoutSDF));
  }
};

static int defVDBRenormalizeSDF = zeno::defNodeClass<VDBRenormalizeSDF>("VDBRenormalizeSDF",
     { /* inputs: */ {
     {gParamType_VDBGrid,"inoutSDF"},
     }, /* outputs: */ {
     {gParamType_VDBGrid,"inoutSDF"}
     }, /* params: */ {
         {"enum 1oUpwind", "method", "1oUpwind"},
         {gParamType_Int, "iterations", "4"},
         {gParamType_Int, "dilateIters", "0"},
     }, /* category: */ {
     "openvdb",
     }});

struct VDBSmooth : zeno::INode {
    virtual void apply() override {
        auto inoutVDBtype = safe_dynamic_cast<VDBGrid>(get_input("inoutVDB"))->getType();
        int width = get_input2_int("width");
        int iterations = get_input2_int("iterations");
        auto type = get_input2_string("type");

        openvdb::FloatGrid::Ptr mask = nullptr;
        if(has_input("MaskGrid")) {
            mask = safe_dynamic_cast<VDBFloatGrid>(get_input("MaskGrid"))->m_grid;
        }

        if (inoutVDBtype == "FloatGrid") {
            auto inoutVDB = safe_uniqueptr_cast<VDBFloatGrid>(clone_input("inoutVDB"));
            auto lsf = openvdb::tools::Filter<openvdb::FloatGrid>(*(inoutVDB->m_grid));
            lsf.setGrainSize(1);
            if(type == "Gaussian")
              lsf.gaussian(width, iterations, mask.get());
            else if(type == "Mean")
              lsf.mean(width, iterations, mask.get());
            else if(type == "Median")
              lsf.median(width, iterations, mask.get());
            //openvdb::tools::ttls_internal::smoothLevelSet(*inoutSDF->m_grid, normIter, halfWidth);
            set_output("inoutVDB", std::move(inoutVDB));
        }
        else if (inoutVDBtype == "Vec3fGrid") {
            auto inoutVDB = safe_uniqueptr_cast<VDBFloat3Grid>(clone_input("inoutVDB"));
            auto lsf = openvdb::tools::Filter<openvdb::Vec3fGrid>(*(inoutVDB->m_grid));
            lsf.setGrainSize(1);
            if(type == "Gaussian")
              lsf.gaussian(width, iterations, mask.get());
            else if(type == "Mean")
              lsf.mean(width, iterations, mask.get());
            else if(type == "Median")
              lsf.median(width, iterations, mask.get());
            set_output("inoutVDB", std::move(inoutVDB));
        }
    }
};

ZENO_DEFNODE(VDBSmooth)(
    { /* inputs: */ {
    {gParamType_VDBGrid,"inoutVDB", "", zeno::Socket_ReadOnly},
    {gParamType_VDBGrid,"MaskGrid", "", zeno::Socket_ReadOnly},
    {"enum Mean Gaussian Median", "type", "Gaussian"},
    {gParamType_Int, "width", "1"},
    {gParamType_Int, "iterations", "1"},
    }, /* outputs: */ {
        {gParamType_VDBGrid,"inoutVDB"},
    }, /* params: */ {
    }, /* category: */ {
    "openvdb",
} });

struct  VDBSmoothSDF : zeno::INode { /* cihou old graph */
  virtual void apply() override {

    auto inoutSDF = safe_uniqueptr_cast<VDBFloatGrid>(clone_input("inoutSDF"));
    int width = get_param_int("width");
    int iterations = get_param_int("iterations");
    auto lsf = openvdb::tools::Filter<openvdb::FloatGrid>(*(inoutSDF->m_grid));
    lsf.setGrainSize(1);
    lsf.gaussian(width, iterations, nullptr);
    //openvdb::tools::ttls_internal::smoothLevelSet(*inoutSDF->m_grid, normIter, halfWidth);
    set_output("inoutSDF", std::move(inoutSDF));
  }
};

static int defVDBSmoothSDF = zeno::defNodeClass<VDBSmoothSDF>("VDBSmoothSDF",
     { /* inputs: */ {
     {gParamType_VDBGrid,"inoutSDF"},
     }, /* outputs: */ {
     {gParamType_VDBGrid,"inoutSDF"},
     }, /* params: */ {
         {gParamType_Int, "width", "1"},
         {gParamType_Int, "iterations", "1"},
         {gParamType_String, "DEPRECATED", "Use VDBSmooth Instead"},
     }, /* category: */ {
     "deprecated",
     }});

struct  VDBDilateTopo : zeno::INode {
  virtual void apply() override {

    auto inoutSDF = safe_uniqueptr_cast<zeno::VDBGrid>(clone_input("inField"));
    auto layers = get_input2_int("layers");

    inoutSDF->dilateTopo(layers);
    set_output("oField", std::move(inoutSDF));
  }
};

static int defVDBDilateTopo = zeno::defNodeClass<VDBDilateTopo>("VDBDilateTopo",
     { /* inputs: */ {
     {gParamType_VDBGrid,"inField", "", zeno::Socket_ReadOnly},
     {gParamType_Int, "layers",}
     }, /* outputs: */ {
         {gParamType_VDBGrid,"oField"}
     }, /* params: */ {
     }, /* category: */ {
     "openvdb",
     }});

struct VDBErodeSDF : zeno::INode {
  virtual void apply() override {
    auto inoutSDF = safe_uniqueptr_cast<VDBFloatGrid>(clone_input("inoutSDF"));
    auto grid = inoutSDF->m_grid;
    auto depth = get_input2_float("depth");
    auto wrangler = [&](auto &leaf, openvdb::Index leafpos) {
        for (auto iter = leaf.beginValueOn(); iter != leaf.endValueOn(); ++iter) {
            iter.modifyValue([&](auto &v) {
                v += depth;
            });
        }
    };
    auto velman = openvdb::tree::LeafManager<std::decay_t<decltype(grid->tree())>>(grid->tree());
    velman.foreach(wrangler);
    set_output("inoutSDF", std::move(inoutSDF));
  }
};

static int defVDBErodeSDF = zeno::defNodeClass<VDBErodeSDF>("VDBErodeSDF",
     { /* inputs: */ {
     {gParamType_VDBGrid,"inoutSDF"}, {gParamType_Float, "depth"},
     }, /* outputs: */ {
       {gParamType_VDBGrid,"inoutSDF"}
     }, /* params: */ {
     }, /* category: */ {
     "openvdb",
     }});

}
