#include "AdaptiveGridGen.h"
#include <openvdb/Types.h>
#include <tbb/parallel_for.h>
namespace zeno{
    
void AdaptiveIndexGenerator::generateAdaptiveGrid(
        AdaptiveIndexGenerator& data, 
        int max_levels, 
        double start_h,
        std::shared_ptr<AdaptiveRule> rule
        )
{
    data.hLevels.resize(max_levels);
    data.hLevels[max_levels-1] = start_h;
    for(int i=max_levels-2; i>=0; i--)
    {
        data.hLevels[i] = data.hLevels[i+1]/2.0;
        printf("%f\n", data.hLevels[i]);
    }
    
    //we shall assume level_max is already provided, by
    //particular method
    
    std::vector<openvdb::FloatTree::LeafNodeType *> leaves;
    openvdb::FloatGrid::Ptr coarse_grid;
    openvdb::FloatGrid::Ptr fine_grid;
    double fine_h;
    
    for(int i = max_levels-2; i>=0; i--)
    {
        
        coarse_grid = data.topoLevels[i+1];
        auto transform =
        openvdb::math::Transform::createLinearTransform(data.hLevels[i]);
        transform->postTranslate(openvdb::Vec3d{0.5,0.5,0.5} *
                                          double(data.hLevels[i]));
        fine_grid = openvdb::FloatGrid::create(float(0));
        fine_grid->setTransform(transform);

        coarse_grid->tree().getNodes(leaves);
        fine_h = data.hLevels[i];
        auto fine_waxr{fine_grid->getAccessor()};
        //loop over voxels of coarser level
        //auto subd = [&](const tbb::blocked_range<size_t> &r) {
        auto coarse_axr{coarse_grid->getConstUnsafeAccessor()};
            
            // leaf iter
            for (auto liter = 0; liter<leaves.size(); ++liter) {
                auto &leaf = *leaves[liter];
                for (auto offset = 0; offset < leaf.SIZE; ++offset) {
                    auto voxelwpos =
                        coarse_grid->indexToWorld(leaf.offsetToGlobalCoord(offset));
                    auto voxelipos = openvdb::Vec3i(coarse_grid->worldToIndex(voxelwpos));
                    float value = coarse_axr.getValue(openvdb::Coord(voxelipos));
                    if(value>0.9)
                    {
                        //we need emit
                    for(int i=-1;i<=1;i+=2)
                    {
                        for(int j=-1;j<=1;j+=2)
                        {
                            for(int k=-1;k<=1;k+=2)
                            {
                                auto fine_pos = voxelwpos + openvdb::Vec3d{(float)i,(float)j,(float)k}*0.5*fine_h;
                                auto wpos = openvdb::Vec3i(fine_grid->worldToIndex(fine_pos));
                                fine_waxr.setValue(openvdb::Coord(wpos), 0.5);
                            }
                        }
                    }
                    }
                } // end for all on voxels
            }
        //}
        //tbb::parallel_for(tbb::blocked_range<size_t>(0, leaves.size()),subd);
        rule->markSubd(fine_grid);
        data.topoLevels[i] = fine_grid->deepCopy();
    }
}

struct testAdaptiveGrid : zeno::INode{
    virtual void apply() override {
        double h_coarse = 1.0/16.0;
        auto coarse_grid = openvdb::FloatGrid::create(float(0));
        auto transform = openvdb::math::Transform::createLinearTransform(h_coarse);
        transform->postTranslate(openvdb::Vec3d{0.5,0.5,0.5} *
                                          double(h_coarse));
        coarse_grid->setTransform(transform);
        auto rule = std::make_shared<TestRule>();
        for(int i=-20; i<=20; i++)
        {
            for(int j=-20;j<=20; j++)
            {
                for(int k=-20;k<=20;k++)
                {
                    openvdb::Coord xyz{i,j,k};
                    coarse_grid->getAccessor().setValue(xyz, 0.5);
                }
            }
        }
        rule->markSubd(coarse_grid);
        auto level0 = zeno::IObject::make<VDBFloatGrid>();
        auto level1 = zeno::IObject::make<VDBFloatGrid>();
        auto level2 = zeno::IObject::make<VDBFloatGrid>();
        auto level3 = zeno::IObject::make<VDBFloatGrid>();
        auto level4 = zeno::IObject::make<VDBFloatGrid>();
        AdaptiveIndexGenerator aig;
        aig.topoLevels.resize(5);
        aig.topoLevels[4] = coarse_grid;
        aig.generateAdaptiveGrid(aig, 5, h_coarse, rule);
        level0->m_grid = aig.topoLevels[0];
        level1->m_grid = aig.topoLevels[1];
        level2->m_grid = aig.topoLevels[2];
        level3->m_grid = aig.topoLevels[3];
        level4->m_grid = aig.topoLevels[4];
        set_output("level0", level4);
        set_output("level1", level3);
        set_output("level2", level2);
        set_output("level3", level1);
        set_output("level4", level0);
    }
};

ZENDEFNODE(testAdaptiveGrid, {
        {},
        {"level0","level1","level2","level3", "level4"},
        {},
        {"AdaptiveSolver"},
});
}