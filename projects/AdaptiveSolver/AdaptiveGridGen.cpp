#include "AdaptiveGridGen.h"
#include <openvdb/Types.h>
#include <tbb/parallel_for.h>
#include <zeno/VDBGrid.h>
#include <zeno/ZenoInc.h>
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
        coarse_grid->tree().getNodes(leaves);
        auto transform =
        openvdb::math::Transform::createLinearTransform(data.hLevels[i]);
        transform->postTranslate(openvdb::Vec3d{0.5,0.5,0.5} *
                                          double(data.hLevels[i]));
        // std::vector<openvdb::FloatGrid::Ptr> fine_grids;
        // fine_grids.resize(leaves.size());
        // for(auto &g:fine_grids)
        // {
        //     g = openvdb::FloatGrid::create(float(0));
        //     g->setTransform(transform);
        // }
        fine_grid = openvdb::FloatGrid::create(float(0));
        fine_grid->setTransform(transform);

        
        fine_h = data.hLevels[i];
        //auto fine_waxr{fine_grid->getAccessor()};
        
        //loop over voxels of coarser level
        //auto subd = [&](const tbb::blocked_range<size_t> &r) {
            auto coarse_axr{coarse_grid->getConstUnsafeAccessor()};
            // leaf iter
            for (auto liter = 0; liter<leaves.size(); ++liter) {
            //for (auto liter = r.begin(); liter != r.end(); ++liter) {
                auto fine_waxr{fine_grid->getAccessor()};
                auto &leaf = *leaves[liter];
                for (auto offset = 0; offset < leaf.SIZE; ++offset) {
                    auto voxelwpos =
                        coarse_grid->indexToWorld(leaf.offsetToGlobalCoord(offset));
                    auto voxelipos = openvdb::Vec3i(coarse_grid->worldToIndex(voxelwpos));
                    float value = coarse_axr.getValue(openvdb::Coord(voxelipos));
                    if(value==1.0)
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
        // };
        // tbb::parallel_for(tbb::blocked_range<size_t>(0, leaves.size()),subd);
        // for(int i=1;i<fine_grids.size();i++)
        // {
        //     zeno::resampleVDB<openvdb::tools::PointSampler,openvdb::FloatGrid>(fine_grids[i], fine_grids[0]);
        // }
        //fine_grid = fine_grids[0]->deepCopy();
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

struct LiquidAdaptiveRule : AdaptiveRule{
    //particle list
    std::shared_ptr<zeno::PrimitiveObject> p;
    virtual void markSubd(openvdb::FloatGrid::Ptr &grid) override
    {
        auto write{grid->getAccessor()};
        // tbb::parallel_for(
        //     (size_t)0,
        //     (size_t)p->size(),
        //     (size_t)1,
        //     [&](size_t index)
        for(int index=0;index<p->size();index++)
            {
                auto pos = p->attr<zeno::vec3f>("pos")[index];
                auto ppos = openvdb::Vec3d{pos[0],pos[1],pos[2]};
                auto wpos = openvdb::Vec3i(grid->worldToIndex(ppos));
                write.setValue(openvdb::Coord(wpos), 1.0);
            }
        //);
        extend(grid);
        extend(grid);
    }
};

struct AdaptiveRuleObject : zeno::IObject
{
    std::shared_ptr<AdaptiveRule> rule;
};

struct MakeLiquidAdaptiveRule : zeno::INode{
    virtual void apply() override{
        auto prim = get_input<zeno::PrimitiveObject>("inGeo");

        auto liquidRule = std::make_shared<LiquidAdaptiveRule>();
        liquidRule->p = prim;
        auto outRuleObj = std::make_shared<AdaptiveRuleObject>();
        outRuleObj->rule = liquidRule;
        set_output("AdaptRule",  outRuleObj);
    }
};
ZENDEFNODE(MakeLiquidAdaptiveRule, {
        {"inGeo"},
        {"AdaptRule"},
        {},
        {"AdaptiveSolver"},
});

struct AdaptiveGridObject : zeno::IObject {
    std::shared_ptr<AdaptiveIndexGenerator> data;
};

struct GenAdaptiveTopo : zeno::INode{
    virtual void apply() override{
        auto rule = get_input<AdaptiveRuleObject>("Rule");
        auto bmin = get_input<zeno::NumericObject>("bmin")->get<zinc::vec3f>();
        auto bmax = get_input<zeno::NumericObject>("bmax")->get<zinc::vec3f>();
        auto dx = get_input<zeno::NumericObject>("startDx")->get<float>();
        auto levels = get_input<zeno::NumericObject>("levels")->get<int>();
        zinc::vec3i dim_min = zinc::vec3i(bmin/dx - zinc::vec3i(4));
        zinc::vec3i dim_max = zinc::vec3i(bmax/dx + zinc::vec3i(4));
        auto coarse_grid = openvdb::FloatGrid::create(float(0));
        auto transform = openvdb::math::Transform::createLinearTransform(dx);
        transform->postTranslate(openvdb::Vec3d{0.5,0.5,0.5} *
                                          double(dx));
        coarse_grid->setTransform(transform);
        for(int i=dim_min[0]; i<=dim_max[0]; i++)
        {
            for(int j=dim_min[1];j<=dim_max[1]; j++)
            {
                for(int k=dim_min[2];k<=dim_max[2];k++)
                {
                    openvdb::Coord xyz{i,j,k};
                    coarse_grid->getAccessor().setValue(xyz, 0.5);
                }
            }
        }
        rule->rule->markSubd(coarse_grid);
        auto oAdaptGrid = std::make_shared<AdaptiveGridObject>();
        auto aig = std::make_shared<AdaptiveIndexGenerator>();
        oAdaptGrid->data = aig;
        aig->topoLevels.resize(levels);
        aig->topoLevels[levels-1] = coarse_grid;
        aig->generateAdaptiveGrid(*aig, levels, dx, rule->rule);
        set_output("AdaptTopo", std::move(oAdaptGrid));
    }
};
ZENDEFNODE(GenAdaptiveTopo, {
        {"Rule", "bmin", "bmax", "startDx", "levels"},
        {"AdaptTopo"},
        {},
        {"AdaptiveSolver"},
});

struct ListAdaptiveTopo : zeno::INode{
    virtual void apply() override{
        auto data = get_input<AdaptiveGridObject>("AdaptTopo");
        auto list = std::make_shared<zeno::ListObject>();
        for(int i=0; i<data->data->topoLevels.size();i++)
        {   
            auto vdbObj = std::make_shared<zeno::VDBFloatGrid>();
            vdbObj->m_grid = data->data->topoLevels[i]->deepCopy();
            list->arr.push_back(std::move(vdbObj));
        }
        set_output("ListAdaptTopo", std::move(list));
    }
};
ZENDEFNODE(ListAdaptiveTopo, {
        {"AdaptTopo"},
        {"ListAdaptTopo"},
        {},
        {"AdaptiveSolver"},
});
}