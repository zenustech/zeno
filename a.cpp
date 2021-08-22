#include <zeno/zeno.h>
#include <zeno/extra/ISubgraphNode.h>
namespace {
struct FLIPSimTemplate : zeno::ISubgraphNode {
    virtual std::string subgraph_name() override {
        return "FLIPSimTemplate";
    }
};
ZENDEFNODE(FLIPSimTemplate, {
    {{"", "Velocity", ""}, {"", "Collision", ""}, {"", "PostAdvVel", ""}, {"", "Pressure", ""}, {"", "CollisionVel", ""}, {"", "DynEmitter", ""}, {"", "ifEmit", ""}, {"", "Particles", ""}, {"", "VelocityWeights", ""}, {"", "dx", ""}, {"", "ExtractedLiquidSDF", ""}, {"", "Gravity", ""}, {"", "CellFWeight", ""}, {"", "Divergence", ""}, {"", "LiquidSDF", ""}, {"", "dt", ""}, {"", "SRC", ""}},
    {{"", "Particles", ""}, {"", "DST", ""}},
    {},
    {"FLIPSolver"},
});
struct makeCollision : zeno::ISubgraphNode {
    virtual std::string subgraph_name() override {
        return "makeCollision";
    }
};
ZENDEFNODE(makeCollision, {
    {{"", "StaticSDF", ""}, {"", "dynamicPrim", ""}, {"", "dx", ""}, {"", "SRC", ""}},
    {{"", "Collider", ""}, {"", "DST", ""}},
    {},
    {"FLIP"},
});
struct makeTank : zeno::ISubgraphNode {
    virtual std::string subgraph_name() override {
        return "makeTank";
    }
};
ZENDEFNODE(makeTank, {
    {{"", "TankSDF", ""}, {"", "dx", ""}, {"", "SRC", ""}},
    {{"", "Particles", ""}, {"", "DST", ""}},
    {},
    {"FLIP"},
});
struct GetFluidStepDT : zeno::ISubgraphNode {
    virtual std::string subgraph_name() override {
        return "GetFluidStepDT";
    }
};
ZENDEFNODE(GetFluidStepDT, {
    {{"", "Velocity", ""}, {"", "CFLNumber", ""}, {"", "dx", ""}, {"", "maxSubSteps", ""}, {"", "SRC", ""}},
    {{"", "step_dt", ""}, {"", "DST", ""}},
    {},
    {"FLIPSolver"},
});
struct MakeFLIP : zeno::ISubgraphNode {
    virtual std::string subgraph_name() override {
        return "MakeFLIP";
    }
};
ZENDEFNODE(MakeFLIP, {
    {{"", "dx", ""}, {"", "SRC", ""}},
    {{"", "World", ""}, {"", "DST", ""}},
    {},
    {"FLIP"},
});
struct StepFLIP : zeno::ISubgraphNode {
    virtual std::string subgraph_name() override {
        return "StepFLIP";
    }
};
ZENDEFNODE(StepFLIP, {
    {{"", "World", ""}, {"", "Particles", ""}, {"", "gravity", ""}, {"", "Collider", ""}, {"", "dt", ""}, {"", "maxSubstep", ""}, {"", "SRC", ""}},
    {{"", "World", ""}, {"", "Particles", ""}, {"", "LiquidSDF", ""}, {"", "DST", ""}},
    {},
    {"FLIP"},
});
struct SmoothLiquidSDF : zeno::ISubgraphNode {
    virtual std::string subgraph_name() override {
        return "SmoothLiquidSDF";
    }
};
ZENDEFNODE(SmoothLiquidSDF, {
    {{"", "inoutSDF", ""}, {"", "dx", ""}, {"", "SRC", ""}},
    {{"", "inoutSDF", ""}, {"", "DST", ""}},
    {},
    {"subgraph"},
});
struct MultiStepFLIP : zeno::ISubgraphNode {
    virtual std::string subgraph_name() override {
        return "MultiStepFLIP";
    }
};
ZENDEFNODE(MultiStepFLIP, {
    {{"", "World", ""}, {"", "Particles", ""}, {"", "Collider", ""}, {"", "gravity", ""}, {"", "dt", ""}, {"", "min_scale", ""}, {"", "dt_scale", ""}, {"", "SRC", ""}},
    {{"", "World", ""}, {"", "Particles", ""}, {"", "LiquidSDF", ""}, {"", "DST", ""}},
    {},
    {"subgraph"},
});
}
