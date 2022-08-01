#include <cstddef>
#include <zeno/zeno.h>
#include <zeno/VDBGrid.h>
#include <zeno/ZenoInc.h>

namespace zeno {
namespace {

struct INTERN_PreViewVDB : INode {
    virtual void apply() override {
        auto vdb = get_input<VDBGrid>("arg0");
        std::shared_ptr<IObject> ret;
        if (dynamic_cast<VDBFloatGrid*>(vdb.get())) {
            ret = this->getThisGraph()->callTempNode("SDFToPrimitive",
                        {
                            {"SDF", vdb},
                            {"isoValue:", objectFromLiterial(0.0f)},
                            {"adaptivity:", objectFromLiterial(0.0f)},
                            {"allowQuads:", std::make_shared<NumericObject>(0)},
                        }).at("prim");
        } else
        if (dynamic_cast<VDBPointsGrid*>(vdb.get())) {
            ret = this->getThisGraph()->callTempNode("VDBPointsToPrimitive",
                        {
                            {"grid", vdb},
                        }).at("prim");
        }
        if (ret)
            set_output("ret0", std::move(ret));
        else
            set_output("ret0", std::make_shared<DummyObject>());
    }
};

static int defINTERN_PreViewVDB = zeno::defNodeClass<INTERN_PreViewVDB>("INTERN_PreViewVDB",
    { /* inputs: */ {
        "arg0",
    }, /* outputs: */ {
        "ret0",
    }, /* params: */ {
    }, /* category: */ {
    "deprecated", // internal
    }});


struct SDFScatterPoints : INode {
    virtual void apply() override {
        auto sdf = get_input<VDBFloatGrid>("SDF");
        auto dx = getThisGraph()->callTempNode("GetVDBVoxelSize", {
            {"vdbGrid", sdf},
        }).at("dx");
        auto data = getThisGraph()->callTempNode("MakeVDBGrid", {
            {"name:", std::make_shared<StringObject>("")},
            {"structure:", std::make_shared<StringObject>("Centered")},
            {"type:", std::make_shared<StringObject>("points")},
            {"Dx", dx},
        }).at("data");
        auto points = getThisGraph()->callTempNode("ParticleEmitter", {
            {"vx:", std::make_shared<NumericObject>(0)},
            {"vy:", std::make_shared<NumericObject>(0)},
            {"vz:", std::make_shared<NumericObject>(0)},
            {"Particles", data},
            {"ShapeSDF", sdf},
        }).at("Particles");
        set_output("Points", points);
    }
};

static int defSDFScatterPoints = zeno::defNodeClass<SDFScatterPoints>("SDFScatterPoints",
    { /* inputs: */ {
        "SDF",
    }, /* outputs: */ {
        "Points",
    }, /* params: */ {
    }, /* category: */ {
    "openvdb",
    }});

}
}
