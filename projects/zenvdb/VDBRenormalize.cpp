#include <zeno/zeno.h>
#include <zeno/NumericObject.h>
#include <vector>
#include <zeno/VDBGrid.h>
#include <openvdb/points/PointCount.h>
#include <openvdb/tree/LeafManager.h>
#include <openvdb/points/PointAdvect.h>
#include <openvdb/tools/Morphology.h>
#include <openvdb/tools/MeshToVolume.h>
#include <openvdb/tools/LevelSetTracker.h>
#include <zeno/ParticlesObject.h>
#include <openvdb/openvdb.h>
#include <openvdb/tools/LevelSetSphere.h>
#include <openvdb/tools/ChangeBackground.h>

namespace zeno {
    struct  VDBRenormalizeSDF : zeno::INode {
  virtual void apply() override {

    auto inoutSDF = get_input("inoutSDF")->as<VDBFloatGrid>();
    int normIter = std::get<int>(get_param("iterations"));
    auto lstracker = openvdb::tools::LevelSetTracker<openvdb::FloatGrid>(*(inoutSDF->m_grid));
    lstracker.setState({openvdb::math::FIRST_BIAS, openvdb::math::TVD_RK3, 1, 1});
    lstracker.setTrimming(openvdb::tools::lstrack::TrimMode::kNone);
    for(int i=0;i<normIter;i++)
        lstracker.normalize();
    //openvdb::tools::changeBackground(inoutSDF->m_grid->tree(), ((float)normIter)*(inoutSDF->m_grid->transformPtr()->voxelSize()[0]));
    //openvdb::tools::signedFloodFill(inoutSDF->m_grid->tree());

    
    set_output("inoutSDF", get_input("inoutSDF"));
  }
};

static int defVDBRenormalizeSDF = zeno::defNodeClass<VDBRenormalizeSDF>("VDBRenormalizeSDF",
     { /* inputs: */ {
     "inoutSDF", 
     }, /* outputs: */ {
     "inoutSDF",
     }, /* params: */ {
         {"string", "method", "1oUpwind"},
         {"int", "iterations", "4"},
     }, /* category: */ {
     "openvdb",
     }});

struct  VDBDilateTopo : zeno::INode {
  virtual void apply() override {

    auto inoutSDF = get_input<zeno::VDBGrid>("inField");
    auto layers = get_input("layers")->as<zeno::NumericObject>()->get<int>();
    inoutSDF->dilateTopo(layers);
    set_output("oField", std::move(inoutSDF));
  }
};

static int defVDBDilateTopo = zeno::defNodeClass<VDBDilateTopo>("VDBDilateTopo",
     { /* inputs: */ {
     "inField", "layers", 
     }, /* outputs: */ {
       "oField"
     }, /* params: */ {
     }, /* category: */ {
     "openvdb",
     }});



}
