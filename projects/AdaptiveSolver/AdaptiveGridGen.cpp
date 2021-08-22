#include "AdaptiveGridGen.h"
#include <openvdb/Types.h>
#include <tbb/parallel_for.h>
#include <cmath>
#include <omp.h>
namespace zeno{
    
void AdaptiveIndexGenerator::generateAdaptiveGrid(
        AdaptiveIndexGenerator& data, 
        int max_levels, 
        double start_h,
        std::shared_ptr<AdaptiveRule> rule
        )
{
    data.hLevels.resize(max_levels);
    data.hLevels[0] = start_h;
    for(int i=1; i<max_levels; i++)
    {
        data.hLevels[i] = data.hLevels[i-1]/2.0;
        printf("%f\n", data.hLevels[i]);
    }
    
    //we shall assume level_max is already provided, by
    //particular method
    
    std::vector<openvdb::FloatTree::LeafNodeType *> leaves;
    openvdb::FloatGrid::Ptr coarse_grid;
    openvdb::FloatGrid::Ptr fine_grid;
    double fine_h;
    
    for(int level = 1; level<max_levels; level++)
    {
        
        coarse_grid = data.topoLevels[level-1];
        auto transform =
        openvdb::math::Transform::createLinearTransform(data.hLevels[level]);
        transform->postTranslate(openvdb::Vec3d{0.5,0.5,0.5} *
                                          double(data.hLevels[level]));
        fine_grid = openvdb::FloatGrid::create(float(0));
        fine_grid->setTransform(transform);
        
        coarse_grid->tree().getNodes(leaves);
        fine_h = data.hLevels[level];
        printf("fine_h is %f\n", fine_h);
        printf("coarse size is %d\n", leaves.size());
        auto fine_waxr{fine_grid->getAccessor()};
        //loop over voxels of coarser level
        //auto subd = [&](const tbb::blocked_range<size_t> &r) {
        auto coarse_axr{coarse_grid->getConstUnsafeAccessor()};
            
            // leaf iter
            //#pragma omp parallel for
            for (auto liter = 0; liter<leaves.size(); ++liter) {
                auto &leaf = *leaves[liter];
                for (auto offset = 0; offset < leaf.SIZE; ++offset) {
                    auto voxelwpos =
                        coarse_grid->indexToWorld(leaf.offsetToGlobalCoord(offset));
                    auto voxelipos = openvdb::Vec3i(coarse_grid->worldToIndex(voxelwpos));
                    float value = coarse_axr.getValue(openvdb::Coord(voxelipos));
                    
                    if(value <= 2 * fine_h && value >= -2 *fine_h)
                    {

                        //we need emit
                        for(int i=-1;i<=1;i+=2)
                        {
                            for(int j=-1;j<=1;j+=2)
                            {
                                for(int k=-1;k<=1;k+=2)
                                {
                                    auto fine_pos = voxelwpos + openvdb::Vec3f{(float)i,(float)j,(float)k}*0.5*fine_h;
                                    float fine_value = 0;
                                    for(int ii=0;ii<=1;++ii)
                                    for(int jj=0;jj<=1;++jj)
                                    for(int kk=0;kk<=1;++kk)
                                    {
                                        auto coarse_voxelwpos = voxelwpos + 
                                            openvdb::Vec3f{(float)(i*ii),(float)(j*jj),(float)(k*kk)}*2*fine_h;
                                        auto coarse_voxelipos = openvdb::Vec3i(coarse_grid->worldToIndex(coarse_voxelwpos));
                                        float coarse_value = coarse_axr.getValue(openvdb::Coord(coarse_voxelipos));
                                        
                                        // Trilinear interpolation
                                        auto delta = abs(coarse_voxelwpos - fine_pos);
                                        auto x = 0.5 / fine_h * delta;
                                        fine_value += (ii*(float)(x[0])+(1-ii)*(1-(float)(x[0])))*
                                            (jj*(float)(x[0])+(1-jj)*(1-(float)(x[0])))*
                                            (kk*(float)(x[0])+(1-kk)*(1-(float)(x[0])))*coarse_value;
                                    }
                                    auto wpos = openvdb::Vec3i(fine_grid->worldToIndex(fine_pos));
                                    fine_waxr.setValue(openvdb::Coord(wpos), fine_value);
                                    // if(level == 1)
                                    // printf("level %d's value is %f\n", level,fine_value);
                                }
                            }
                        }
                    }
                } // end for all on voxels
            }
        //}
        //tbb::parallel_for(tbb::blocked_range<size_t>(0, leaves.size()),subd);
        //rule->markSubd(fine_grid);
        openvdb::tools::signedFloodFill(fine_grid->tree());
        data.topoLevels[level] = fine_grid->deepCopy();
        //fine_grid->tree().getNodes(leaves);
        
    }

}

struct generateAdaptiveGrid : zeno::INode{
    virtual void apply() override {

        double h_coarse = 0.08;
        if(has_input("Dx"))
        {
            h_coarse = get_input("Dx")->as<NumericObject>()->get<float>();
        }
        // auto coarse_grid = openvdb::FloatGrid::create(float(0));
        auto coarse_grid = get_input("VDBGrid")->as<VDBFloatGrid>();
        
        auto transform = openvdb::math::Transform::createLinearTransform(h_coarse);
        transform->postTranslate(openvdb::Vec3d{0.5,0.5,0.5} *
                                          double(h_coarse));
        coarse_grid->setTransform(transform);

        auto rule = std::make_shared<TestRule>();
        //rule->markSubd(coarse_grid->m_grid);
        AdaptiveIndexGenerator aig;
        int max_level = 5;
        aig.topoLevels.resize(max_level);
        aig.topoLevels[0] = coarse_grid->m_grid;

        aig.generateAdaptiveGrid(aig, max_level, h_coarse, rule);

        auto level0 = zeno::IObject::make<VDBFloatGrid>();
        auto level1 = zeno::IObject::make<VDBFloatGrid>();
        auto level2 = zeno::IObject::make<VDBFloatGrid>();
        auto level3 = zeno::IObject::make<VDBFloatGrid>();
        auto level4 = zeno::IObject::make<VDBFloatGrid>();
        level0->m_grid = aig.topoLevels[0];
        level1->m_grid = aig.topoLevels[1];
        level2->m_grid = aig.topoLevels[2];
        level3->m_grid = aig.topoLevels[3];
        level4->m_grid = aig.topoLevels[4];
        printf("adaptive grid generate done\n");
        set_output("level0", level0);
        set_output("level1", level1);
        set_output("level2", level2);
        set_output("level3", level3);
        set_output("level4", level4);
    }
};

ZENDEFNODE(generateAdaptiveGrid, {
        {"VDBGrid","Dx"},
        {"level0", "level1", "level2", "level3", "level4"},
        {},
        {"AdaptiveSolver"},
});
}