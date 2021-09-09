#include "AdaptiveGridGen.h"
#include <openvdb/Types.h>
#include <tbb/parallel_for.h>
#include <cmath>
#include <zeno/MeshObject.h>
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
    openvdb::FloatGrid::Ptr tag_grid;
    double fine_h;
    
    for(int level = 1; level<max_levels; level++)
    {
        coarse_grid = data.sdf[level-1];
        auto transform =
        openvdb::math::Transform::createLinearTransform(data.hLevels[level]);
        transform->postTranslate(openvdb::Vec3d{0.5,0.5,0.5} *
                                          double(data.hLevels[level]));
        fine_grid = openvdb::FloatGrid::create(float(0));
        tag_grid = openvdb::FloatGrid::create(float(0));
        fine_grid->setTransform(transform);
        tag_grid->setTransform(transform);

        // openvdb::Vec3f initPos = fine_grid->indexToWorld(openvdb::Coord(openvdb::Vec3i(5,0,0)));
        // openvdb::Vec3f coarseInitPos = coarse_grid->indexToWorld(openvdb::Coord(openvdb::Vec3i(1,0,0)));
        // printf("init pos is (%f,%f,%f), coarse init pos is (%f,%f,%f)\n",
        //     initPos[0], initPos[1], initPos[2], coarseInitPos[0], coarseInitPos[1], coarseInitPos[2]);
        
        coarse_grid->tree().getNodes(leaves);
        fine_h = data.hLevels[level];
        printf("fine_h is %f\n", fine_h);
        //printf("coarse size is %d\n", leaves.size());
        auto fine_waxr{fine_grid->getAccessor()};
        auto tag_waxr{tag_grid->getAccessor()};
        //loop over voxels of coarser level
        //auto subd = [&](const tbb::blocked_range<size_t> &r) {
        auto coarse_axr{coarse_grid->getConstUnsafeAccessor()};
        int emitCount = 0;
        // leaf iter
        //#pragma omp parallel for
        for (auto liter = 0; liter<leaves.size(); ++liter) {
            auto &leaf = *leaves[liter];
            for (auto offset = 0; offset < leaf.SIZE; ++offset) 
            {
                
                auto voxelwpos =
                    coarse_grid->indexToWorld(leaf.offsetToGlobalCoord(offset));
                auto voxelipos = openvdb::Vec3i(coarse_grid->worldToIndex(voxelwpos));
                if(coarse_grid->tree().isValueOff(openvdb::Coord(voxelipos)))
                    continue;
                float value = coarse_axr.getValue(openvdb::Coord(voxelipos));
                
                if(value <= 2 * fine_h && value >= -2 * fine_h)
                {
                    emitCount++;
                    //we need emit
                    for(int i=-1;i<=1;i+=2)
                    {
                        for(int j=-1;j<=1;j+=2)
                        {
                            for(int k=-1;k<=1;k+=2)
                            {
                                auto fine_pos = voxelwpos + openvdb::Vec3d{(float)i,(float)j,(float)k}*0.5*fine_h;
                                float fine_value = 0;

                                auto driftipos = openvdb::Vec3i(i,j,k);
                                for(int tt = 0;tt<3;++tt)
                                    if(driftipos[tt] > 0)
                                        driftipos[tt] = 0;
                                openvdb::Vec3d basewpos = voxelwpos + driftipos * 2 * fine_h;
                                auto x = 0.5 / fine_h * abs(basewpos - fine_pos);
                                float weightsum = 0;
                                float covalue[8];
                                for(int ii=0;ii<=1;++ii)
                                for(int jj=0;jj<=1;++jj)
                                for(int kk=0;kk<=1;++kk)
                                {
                                    auto coarse_voxelwpos = basewpos + 
                                        openvdb::Vec3d{(float)(ii),(float)(jj),(float)(kk)}*2*fine_h;
                                    auto coarse_voxelipos = openvdb::Vec3i(coarse_grid->worldToIndex(coarse_voxelwpos));
                                    float coarse_value = coarse_axr.getValue(openvdb::Coord(coarse_voxelipos));
                                    
                                    // Trilinear interpolation
                                    fine_value += (ii*(float)(x[0])+(1-ii)*(1-(float)(x[0])))*
                                        (jj*(float)(x[1])+(1-jj)*(1-(float)(x[1])))*
                                        (kk*(float)(x[2])+(1-kk)*(1-(float)(x[2])))*coarse_value;
                                    covalue[ii*4+jj*2+kk] = coarse_value;
                                }

                                openvdb::Vec3i coarse_ipos = openvdb::Vec3i(coarse_grid->worldToIndex(basewpos));
                                openvdb::Vec3i wpos = openvdb::Vec3i(fine_grid->worldToIndex(fine_pos));
                                
                                // if(fine_value < 0.1 * fine_h && fine_value > -0.1 * fine_h)
                                //     printf("finevalue is %f. index is (%d,%d,%d), pos is (%f,%f,%f)\n",
                                //         fine_value,
                                //         wpos[0], wpos[1], wpos[2], fine_pos[0], fine_pos[1], fine_pos[2]);
                                
                                // printf("value is %f, Indexpos is (%d,%d,%d), world coord is (%f,%f,%f). baseIndexpos is (%d,%d,%d), world coord is (%f,%f,%f).\n", 
                                //             fine_value,wpos[0],wpos[1],wpos[2], 
                                //             fine_pos[0],fine_pos[1],fine_pos[2],
                                //             coarse_ipos[0],coarse_ipos[1],coarse_ipos[2],
                                //             basewpos[0],basewpos[1],basewpos[2]);
                                fine_waxr.setValue(openvdb::Coord(wpos), fine_value);
                            }
                        }
                    }
                } 
            } // end for all on voxels
        }

        printf("fine count is %d\n", emitCount * 8);
        
        //openvdb::tools::signedFloodFill(fine_grid->tree());
        data.sdf[level] = fine_grid->deepCopy();
        data.tag[level] = fine_grid->deepCopy();
        data.sdf[level]->setGridClass(openvdb::GRID_LEVEL_SET);
        printf("fine grid active voxel is %d\n", fine_grid->activeVoxelCount());
        rule->markSubd(data.sdf[level], data.tag[level]);
        //fine_grid->tree().getNodes(leaves);
        leaves.clear();
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

        auto rule = std::make_shared<LiquidAdaptiveRule>();
        AdaptiveIndexGenerator aig;
        int max_level = 5;
        aig.sdf.resize(max_level);
        aig.tag.resize(max_level);
        aig.sdf[0] = coarse_grid->m_grid;
        aig.tag[0] = coarse_grid->m_grid->deepCopy();
        aig.sdf[0]->setGridClass(openvdb::GRID_LEVEL_SET);
        // aig.tag[0] = openvdb::FloatGrid::create(float(0));
        // aig.tag[0]->setTransform(transform);
        rule->markSubd(aig.sdf[0], aig.tag[0]);

        aig.generateAdaptiveGrid(aig, max_level, h_coarse, rule);

        auto level0 = zeno::IObject::make<VDBFloatGrid>();
        auto level1 = zeno::IObject::make<VDBFloatGrid>();
        auto level2 = zeno::IObject::make<VDBFloatGrid>();
        auto level3 = zeno::IObject::make<VDBFloatGrid>();
        auto level4 = zeno::IObject::make<VDBFloatGrid>();
        level0->m_grid = aig.sdf[0];
        level1->m_grid = aig.sdf[1];
        level2->m_grid = aig.sdf[2];
        level3->m_grid = aig.sdf[3];
        level4->m_grid = aig.sdf[4];
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

struct MeshToMultiGridLevelSet : zeno::INode{
    void fillInner(mgData &data){
        auto sdf = data.sdf[0];
        auto dx = data.hLevels[0];
        openvdb::FloatGrid::Ptr tag = zeno::IObject::make<VDBFloatGrid>()->m_grid;
        tag->setTree(std::make_shared<openvdb::FloatTree>(
                sdf->tree(), /*bgval*/ float(1),
                openvdb::TopologyCopy()));
        int activeNum = sdf->activeVoxelCount();
        std::vector<openvdb::FloatTree::LeafNodeType *> leaves;

        auto extendTag = [&](const tbb::blocked_range<size_t> &r){
            auto tag_axr{tag->getAccessor()};
            auto sdf_axr{sdf->getConstAccessor()};
            
            for (auto liter = r.begin(); liter != r.end(); ++liter) {
                auto &leaf = *leaves[liter];
                for (auto offset = 0; offset < leaf.SIZE; ++offset) {
                    auto coord = leaf.offsetToGlobalCoord(offset);
                    if(!sdf_axr.isValueOn(coord) || tag_axr.getValue(coord) == 2 )
                        continue;
                    if( sdf_axr.getValue(coord) >= 0)
                        continue;
                    int count = 0;
                    for(int i = -1;i<=1; i += 2)
                    for(int j = 0;j<3;++j)
                    {
                        auto neighbor = coord;
                        neighbor[j] += i;
                        if(!sdf_axr.isValueOn(neighbor))
                        {
                            tag_axr.setValue(neighbor, 0.0f);
                        }
                        else
                            count++;
                    }
                    if(count == 6)
                        tag_axr.setValue(coord, 2);
                    else
                        tag_axr.setValue(coord, 1);
                }
            }
        };
        auto computeSDF = [&](const tbb::blocked_range<size_t> &r){
            auto tag_axr{tag->getConstAccessor()};
            auto sdf_axr{sdf->getAccessor()};
            for (auto liter = r.begin(); liter != r.end(); ++liter) {
                auto &leaf = *leaves[liter];
                for (auto offset = 0; offset < leaf.SIZE; ++offset) {
                    auto coord = leaf.offsetToGlobalCoord(offset);
                    if(!tag_axr.isValueOn(coord) || tag_axr.getValue(coord) > 0.01f)
                    {
                        continue;
                    }
                    float dis[3] = {100000,100000,100000};
                    int sign[3];
                    
                    for(int i=-1;i<=1;i += 2)
                    for(int select = 0;select < 3;++select)
                    {
                        auto base = openvdb::Vec3i(0,0,0);
                        base[select] = i;
                        auto ipos = coord + base;
                        if(!tag_axr.isValueOn(openvdb::Coord(ipos)) || tag_axr.getValue(openvdb::Coord(ipos)) == 0)
                            continue;
                        float nei_value = sdf_axr.getValue(openvdb::Coord(ipos));
                        for(int t = 0;t < 3;++t)
                            if(abs(nei_value) < dis[t] && nei_value != 0)
                            {
                                for(int tt= 2;tt>=t+1;--tt)
                                {
                                    dis[tt] = dis[tt-1];
                                    sign[tt] = sign[tt-1];
                                }
                                dis[t] = abs(nei_value);
                                sign[t] = nei_value / abs(nei_value);
                                break;
                            }
                    }
                    
                    float d = dis[0] + dx;
                    if(d > dis[1])
                    {
                        d = 0.5 * (dis[0] + dis[1] + sqrt(2 * dx * dx - (dis[1]-dis[0]) * (dis[1]-dis[0])));
                        if(d > dis[2])
                        {
                            float delta = dis[0] + dis[1] + dis[2];
                            delta = delta * delta  - 3 *(dis[0] * dis[0] + 
                                dis[1] * dis[1] + dis[2] * dis[2] - dx * dx);
                            if(delta < 0)
                                delta = 0;
                            d = 0.3333 * (dis[0] + dis[1] + dis[2] + sqrt(delta));
                        }
                    }
                    float value = sign[0] * d;
                    sdf_axr.setValue(coord, value);
                }
            }
        };

        tag->tree().getNodes(leaves);
        for(int i=0;;++i)
        {
            tbb::parallel_for(tbb::blocked_range<size_t>(0, leaves.size()), extendTag);
            leaves.clear();
            tag->tree().getNodes(leaves);
            //printf("leaves size is %d\n", leaves.size());
            tbb::parallel_for(tbb::blocked_range<size_t>(0, leaves.size()), computeSDF);
            
            int newAN = sdf->activeVoxelCount();
            //printf("iter %d  activeNum is %d, newAN is %d\n", i, activeNum, newAN);
            if(activeNum == newAN)
                break;
            else
                activeNum = newAN;
        }
    }

    virtual void apply() override {
        float h = 0.08;
        int max_level = 5;
        if(has_input("maxDx"))
        {
            h = get_input("maxDx")->as<NumericObject>()->get<float>();
        }
        if(has_input("max_level"))
        {
            max_level = get_input("max_level")->as<NumericObject>()->get<int>();
        }
        auto mesh = get_input("mesh")->as<zeno::MeshObject>();
        
        std::vector<openvdb::Vec3s> points;
        std::vector<openvdb::Vec3I> triangles;
        std::vector<openvdb::Vec4I> quads;

        auto data = zeno::IObject::make<mgData>();

        data->resize(max_level);
        data->hLevels[0] = h;
        
        points.resize(mesh->vertices.size());
        triangles.resize(mesh->vertices.size()/3);
        quads.resize(0);
    #pragma omp parallel for
        for(int i=0;i<mesh->vertices.size();i++)
        {
            points[i] = openvdb::Vec3s(mesh->vertices[i].x, mesh->vertices[i].y, mesh->vertices[i].z);
        }
    #pragma omp parallel for
        for(int i=0;i<mesh->vertices.size()/3;i++)
        {
            triangles[i] = openvdb::Vec3I(i*3, i*3+1, i*3+2);
        }
        for(int i=0;i<max_level;++i)
        {
            if(i > 0)
                data->hLevels[i] = data->hLevels[i - 1] / 2.0;
            auto result = zeno::IObject::make<VDBFloatGrid>();
            auto vdbtransform = openvdb::math::Transform::createLinearTransform(data->hLevels[i]);
            if(std::get<std::string>(get_param("type"))==std::string("vertex"))
            {
                vdbtransform->postTranslate(openvdb::Vec3d{ -0.5,-0.5,-0.5 }*double(data->hLevels[i]));
            }
            result->m_grid = openvdb::tools::meshToLevelSet<openvdb::FloatGrid>(*vdbtransform,points, triangles, quads, 4);
            openvdb::tools::signedFloodFill(result->m_grid->tree());
            data->sdf[i] = result->m_grid;
        }
        printf("begin to init\n");
        data->initData();
        fillInner(*data);
        printf("init over\n");

        printf("adaptive grid generate done\n");
        set_output("mgData", data);
  };
};

ZENDEFNODE(MeshToMultiGridLevelSet,
    {
        {"mesh","maxDx","max_level"},
        {"mgData"},
        {{"string", "type", "vertex"},},
        {"AdaptiveSolver"},
    }
);

}