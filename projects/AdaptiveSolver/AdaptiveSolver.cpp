#include "AdaptiveSolver.h"
#include <openvdb/Types.h>
#include <tbb/parallel_for.h>

namespace zeno{

struct AdaptiveSolver : zeno::INode
{
    AdaptiveIndexGenerator aig;
    
    virtual void apply() override {
        int levelNum = get_input<int>("levelNum");
        auto level0 = get_input<VDBFloatGrid>("level0");
        auto level1 = has_input("level1") ? get_input<VDBFloatGrid>("level1")
            :zeno::IObject::make<VDBFloatGrid>();
        auto level2 = has_input("level2") ? get_input<VDBFloatGrid>("level2")
            :zeno::IObject::make<VDBFloatGrid>();
        auto level3 = has_input("level3") ? get_input<VDBFloatGrid>("level3")
            :zeno::IObject::make<VDBFloatGrid>();
        auto level4 = has_input("level4") ? get_input<VDBFloatGrid>("level4")
            :zeno::IObject::make<VDBFloatGrid>();
        float h = has_input("Dx") ? get_input<float>("Dx")
            :0.08;
        aig.topoLevels[0] = level0->m_grid;
        aig.topoLevels[1] = level1->m_grid;
        aig.topoLevels[2] = level2->m_grid;
        aig.topoLevels[3] = level3->m_grid;
        aig.topoLevels[4] = level4->m_grid;

        set_output("level0", level4);
        set_output("level1", level3);
        set_output("level2", level2);
        set_output("level3", level1);
        set_output("level4", level0);
    }
}

ZENDEFNODE(AdaptiveSolver, {
        {"levelNum", "level0","level1","level2","level3", "level4", "Dx"},
        {"level0","level1","level2","level3", "level4"},
        {},
        {"AdaptiveSolver"},
});
}