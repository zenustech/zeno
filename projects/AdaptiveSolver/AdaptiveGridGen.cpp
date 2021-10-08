#include "AdaptiveGridGen.h"
#include <openvdb/Types.h>
#include <tbb/parallel_for.h>

#include <zeno/MeshObject.h>
#include <omp.h>
#include "tbb/blocked_range3d.h"
#include <openvdb/tools/LevelSetUtil.h>
#include <openvdb/tools/FastSweeping.h>
namespace zeno{
void agIterStuff::init(std::vector<openvdb::FloatGrid::Ptr> pressField)
{
    int num = pressField.size();
    rhsGrid.resize(num);
    resGrid.resize(num);
    r2Grid.resize(num);
    pGrid.resize(num);
    ApGrid.resize(num);
    for(int i=0;i<pressField.size();++i)
    {
        rhsGrid[i] = pressField[i]->deepCopy();
        resGrid[i] = pressField[i]->deepCopy();
        r2Grid[i] = pressField[i]->deepCopy();
        pGrid[i] = pressField[i]->deepCopy();
        ApGrid[i] = pressField[i]->deepCopy();
    }
}
void agData::resize(int lNum)
{
    tag = openvdb::Int32Grid::create(0);
    levelNum = lNum;
    dx.resize(levelNum);
    volumeField.resize(levelNum);
    temperatureField.resize(levelNum);
    velField[0].resize(levelNum);velField[1].resize(levelNum);velField[2].resize(levelNum);
    pressField.resize(levelNum);
    gradPressField[0].resize(levelNum);gradPressField[1].resize(levelNum);gradPressField[2].resize(levelNum);
    status.resize(levelNum);
    for(int i=0;i<levelNum;++i)
    {
        volumeField[i] = openvdb::FloatGrid::create(0);
        temperatureField[i] = openvdb::FloatGrid::create(0);
        for(int j=0;j<3;++j)
        {    
            velField[j][i] = openvdb::FloatGrid::create(0);
            gradPressField[j][i] = openvdb::FloatGrid::create(0);
        }
        pressField[i] = openvdb::FloatGrid::create(0);
        status[i] = openvdb::Int32Grid::create(0);
    }

}
void agData::initData(openvdb::FloatGrid::Ptr sdf, int lNum, float inputdt)
{
    resize(lNum);
    dt = inputdt;
    dens = 1;
    dx[0] = sdf->voxelSize()[0];
    for(int l = 1;l < levelNum;++l)
        dx[l] = dx[l-1] * 2;
    float tagdx = dx[0] * 0.5;
    auto inputbbox = openvdb::CoordBBox();
    auto is_valid = sdf->tree().evalLeafBoundingBox(inputbbox);
    if (!is_valid) {
        return;
    }
    auto worldinputbbox = openvdb::BBoxd(sdf->indexToWorld(inputbbox.min()),
                                    sdf->indexToWorld(inputbbox.max()));
    worldinputbbox.min() += openvdb::Vec3d(-0.5);
    worldinputbbox.max() += openvdb::Vec3d(0.5);
    printf("generate world input box over!\n");

    volumeField[0] = sdf->deepCopy();
    openvdb::tools::sdfToFogVolume(*volumeField[0]);
    temperatureField[0] = volumeField[0]->deepCopy();

    // set transform
    for(int level = 0;level < levelNum;++level)
    {
        auto transform = openvdb::math::Transform::createLinearTransform(dx[level]);
        transform->postTranslate(openvdb::Vec3d{0.5,0.5,0.5}*dx[level]);
        temperatureField[level]->setTransform(transform);
        volumeField[level]->setTransform(transform);
        pressField[level]->setTransform(transform);
        status[level]->setTransform(transform);
        for(int i=0;i<3;++i)
        {
            auto veltrans = transform->copy();
            auto drift = openvdb::Vec3d(0);
            drift[i] = -0.5;
            veltrans->postTranslate(drift * dx[level]);
            velField[i][level]->setTransform(veltrans);
            gradPressField[i][level]->setTransform(veltrans);
        }
    }

    auto tagtrans = openvdb::math::Transform::createLinearTransform(tagdx);
    //tagtrans->postTranslate(openvdb::Vec3d{0.5,0.5,0.5}*tagdx);
    tag->setTransform(tagtrans);
    // init tag
    {
        auto loopbegin =
            openvdb::tools::local_util::floorVec3(tag->worldToIndex(worldinputbbox.min()));
        auto loopend =
            openvdb::tools::local_util::ceilVec3(tag->worldToIndex(worldinputbbox.max()));

        auto tag_axr = tag->getAccessor();
        auto pispace_range = tbb::blocked_range3d<int, int, int>(
            loopbegin.x(), loopend.x(), loopbegin.y(), loopend.y(), loopbegin.z(),
            loopend.z());  
        for (int i = pispace_range.pages().begin(); i < pispace_range.pages().end(); i += 1) {
            for (int j = pispace_range.rows().begin(); j < pispace_range.rows().end(); j += 1) {
                for (int k = pispace_range.cols().begin(); k < pispace_range.cols().end(); k += 1) {
                    openvdb::Coord coord(i,j,k);
                    tag_axr.setValue(coord, 0);
                    tag_axr.setValueOff(coord);
                }
            }
        }
    }
    
    // init uniform grids
    for(int level = 0;level < levelNum;++level)
    {
        auto loopbegin =
            openvdb::tools::local_util::floorVec3(volumeField[level]->worldToIndex(worldinputbbox.min()));
        auto loopend =
            openvdb::tools::local_util::ceilVec3(volumeField[level]->worldToIndex(worldinputbbox.max()));
        auto pispace_range = tbb::blocked_range3d<int, int, int>(
            loopbegin.x(), loopend.x(), loopbegin.y(), loopend.y(), loopbegin.z(),
            loopend.z());
        auto vol_axr = volumeField[level]->getAccessor();
        auto tem_axr = temperatureField[level]->getAccessor();
        auto status_axr = status[level]->getAccessor();
        auto press_axr = pressField[level]->getAccessor();
        auto tag_axr = tag->getAccessor();
        openvdb::tree::ValueAccessor<openvdb::FloatTree, true> vel_axr[3]=
                    {velField[0][level]->getAccessor(),
                    velField[1][level]->getAccessor(),
                    velField[2][level]->getAccessor()};
        openvdb::tree::ValueAccessor<openvdb::FloatTree, true> gradP_axr[3]=
                    {gradPressField[0][level]->getAccessor(),
                    gradPressField[1][level]->getAccessor(),
                    gradPressField[2][level]->getAccessor()};
        for (int i = pispace_range.pages().begin(); i < pispace_range.pages().end(); i += 1) {
            for (int j = pispace_range.rows().begin(); j < pispace_range.rows().end(); j += 1) {
                for (int k = pispace_range.cols().begin(); k < pispace_range.cols().end(); k += 1) {
                    openvdb::Coord coord(i,j,k);
                    if(!vol_axr.isValueOn(coord))
                    {
                        vol_axr.setValue(coord, 0);
                        tem_axr.setValue(coord, 0);
                        status_axr.setValue(coord, 2);
                    }
                    if(level == 0)
                    {
                        status_axr.setValue(coord, 0);
                        auto wpos = volumeField[level]->indexToWorld(coord);
                        auto tagindex = round(tag->worldToIndex(wpos));
                        
                        tag_axr.setValue(tagindex, 0);
                    }
                    press_axr.setValue(coord, 0);
                    for(int ii=0;ii<3;++ii)
                        for(int ss = 0;ss<=1;++ss)
                        {
                            auto ipos = coord;
                            ipos[ii] += ss;
                            vel_axr[ii].setValue(ipos, 0);
                            gradP_axr[ii].setValue(ipos, 0);
                        }
                }
            }
        }
        for (openvdb::FloatGrid::ValueOffIter iter = volumeField[level]->beginValueOff(); iter.test(); ++iter) {
            iter.setValue(0);
        }
        for (openvdb::FloatGrid::ValueOffIter iter = temperatureField[level]->beginValueOff(); iter.test(); ++iter) {
            iter.setValue(0);
        }
        for (openvdb::FloatGrid::ValueOffIter iter = pressField[level]->beginValueOff(); iter.test(); ++iter) {
            iter.setValue(0);
        }
    }
    
    buffer.init(pressField);
}
void agData::makeCoarse()
{
    // initial tag
    {
        std::vector<openvdb::Int32Tree::LeafNodeType *> leaves;
        tag->tree().getNodes(leaves);
        auto setTagOFF = [&](const tbb::blocked_range<size_t> &r)
        {
            auto tag_axr = tag->getAccessor();
            for (auto liter = r.begin(); liter != r.end(); ++liter) {
                auto &leaf = *leaves[liter];
                for (auto offset = 0; offset < leaf.SIZE; ++offset) {
                    auto coord = leaf.offsetToGlobalCoord(offset);
                    tag_axr.setValueOff(coord);
                    //if(!tag_axr.isValueOn(coord))
                    //    continue;
                    //tag_axr.setValue(coord, -1);
                }
            }
        };
        tbb::parallel_for(tbb::blocked_range<size_t>(0, leaves.size()), setTagOFF);
        leaves.clear();

        std::vector<openvdb::FloatTree::LeafNodeType *> leaves2;
        auto setTagZero = [&](const tbb::blocked_range<size_t> &r)
        {
            auto tag_axr = tag->getAccessor();
            auto vol_axr = volumeField[0]->getConstAccessor();
            for (auto liter = r.begin(); liter != r.end(); ++liter) {
                auto &leaf = *leaves2[liter];
                for (auto offset = 0; offset < leaf.SIZE; ++offset) {
                    auto coord = leaf.offsetToGlobalCoord(offset);
                    if(!vol_axr.isValueOn(coord))
                        continue;
                    auto wpos = volumeField[0]->indexToWorld(coord);
                    auto tindex = round(tag->worldToIndex(wpos));
                    tag_axr.setValue(tindex, 0);
                }
            }
        };
        
        volumeField[0]->tree().getNodes(leaves2);
        tbb::parallel_for(tbb::blocked_range<size_t>(0, leaves2.size()), setTagZero);
    }
    auto buf = status[0]->deepCopy();
    status[0]->clear();
    status[0]->setTree((std::make_shared<openvdb::Int32Tree>(
                    buf->tree(), /*bgval*/ float(0),
                    openvdb::TopologyCopy())));
    buf->clear();
    for(int level = 1;level < levelNum;++level)
    {
        std::vector<openvdb::FloatTree::LeafNodeType *> leaves;
        buf = status[level]->deepCopy();
        status[level]->clear();
        status[level]->setTree((std::make_shared<openvdb::Int32Tree>(
                    buf->tree(), /*bgval*/ float(2),
                    openvdb::TopologyCopy())));
        buf->clear();
        pressField[level]->tree().getNodes(leaves);

        auto new_status = status[level-1]->deepCopy();
        auto make_Coarse = [&](const tbb::blocked_range<size_t> &r)
        {
            auto vol_axr = volumeField[level]->getAccessor();
            auto tem_axr = temperatureField[level]->getAccessor();
            auto status_axr = status[level]->getAccessor();
            auto press_axr = pressField[level]->getAccessor();
            auto tag_axr = tag->getAccessor();

            auto fine_vol_axr = volumeField[level-1]->getAccessor();
            auto fine_tem_axr = temperatureField[level-1]->getAccessor();
            auto fine_press_axr = pressField[level-1]->getAccessor();
            auto fine_status_axr = status[level-1]->getConstAccessor();

            auto new_fine_status_axr = new_status->getAccessor();
            BoxSample vol_sampler(fine_vol_axr, volumeField[level-1]->transform());
            BoxSample tem_sampler(fine_tem_axr, temperatureField[level-1]->transform());
            BoxSample press_sampler(fine_press_axr, pressField[level-1]->transform());

            for (auto liter = r.begin(); liter != r.end(); ++liter) {
                auto &leaf = *leaves[liter];
                for (auto offset = 0; offset < leaf.SIZE; ++offset) {
                    auto coord = leaf.offsetToGlobalCoord(offset);
                    if(!vol_axr.isValueOn(coord))
                        continue;
                    auto wpos = volumeField[level]->indexToWorld(coord);
                    bool canCoarse = true;
                    for(int i=-1;i<=1;i+=2)
                    for(int j=-1;j<=1;j+=2)
                    for(int k=-1;k<=1;k+=2)
                    {
                        auto fwpos = wpos + openvdb::Vec3i(i,j,k) * 0.25 * dx[level];
                        auto findex = round(volumeField[level-1]->worldToIndex(fwpos));
                        if(fine_status_axr.getValue(findex) != 0)
                        {
                            canCoarse = false;
                            break;
                        }
                    }
                    auto vol = vol_sampler.wsSample(wpos);

                    if(vol<=0.01 && canCoarse)
                    {
                        for(int i=-1;i<=1;i+=2)
                        for(int j=-1;j<=1;j+=2)
                        for(int k=-1;k<=1;k+=2)
                        {
                            auto fwpos = wpos + openvdb::Vec3i(i,j,k) * 0.25 * dx[level];
                            
                            auto findex = round(volumeField[level-1]->worldToIndex(fwpos));
                            new_fine_status_axr.setValue(findex, 2);

                            auto tindex = round(tag->worldToIndex(fwpos));
                            tag_axr.setValueOff(tindex);
                        }
                        auto tindex = round(tag->worldToIndex(wpos));
                        tag_axr.setValue(tindex, level);
                        
                        auto press = press_sampler.wsSample(wpos);
                        auto tem = tem_sampler.wsSample(wpos);
                        press_axr.setValue(coord, press);
                        tem_axr.setValue(coord, tem);
                        status_axr.setValue(coord, 0);
                    }
                }
            }
            
        };

        tbb::parallel_for(tbb::blocked_range<size_t>(0, leaves.size()), make_Coarse);
        leaves.clear();
        velField[0][level]->tree().getNodes(leaves);
        auto computeVel = [&](const tbb::blocked_range<size_t> &r)
        {
            auto status_axr = status[level]->getConstAccessor();
            openvdb::tree::ValueAccessor<openvdb::FloatTree, true> vel_axr[3]=
                    {velField[0][level]->getAccessor(),
                    velField[1][level]->getAccessor(),
                    velField[2][level]->getAccessor()};

            openvdb::tree::ValueAccessor<const openvdb::FloatTree, true> fine_vel_axr[3]=
                    {velField[0][level-1]->getConstAccessor(),
                    velField[1][level-1]->getConstAccessor(),
                    velField[2][level-1]->getConstAccessor()};
            ConstBoxSample velSampler[3] = {
                    ConstBoxSample(fine_vel_axr[0], velField[0][level-1]->transform()),
                    ConstBoxSample(fine_vel_axr[1], velField[1][level-1]->transform()),
                    ConstBoxSample(fine_vel_axr[2], velField[2][level-1]->transform())
                };
            for (auto liter = r.begin(); liter != r.end(); ++liter) {
                auto &leaf = *leaves[liter];
                for (auto offset = 0; offset < leaf.SIZE; ++offset) {
                    auto coord = leaf.offsetToGlobalCoord(offset);
                    for(int i=0;i<3;++i)
                    {
                        if(!vel_axr[i].isValueOn(coord))
                            continue;
                        bool needAvg = false;
                        for(int ss = -1;ss<=0;++ss)
                        {
                            auto ipos = coord;
                            ipos[i] += ss;
                            if(status_axr.isValueOn(ipos) && status_axr.getValue(ipos) == 0)
                            {
                                needAvg = true;
                                break;
                            }
                        }
                        auto wpos = velField[i][level]->indexToWorld(coord);
                        auto vel = velSampler[i].wsSample(wpos);
                        vel_axr[i].setValue(coord, vel);
                    }
                }
            }
        };
        
        tbb::parallel_for(tbb::blocked_range<size_t>(0, leaves.size()), computeVel);
        status[level - 1]->clear();
        status[level - 1] = new_status;
    }
    
    for(int level = 0;level < levelNum;++level)
    {
        std::vector<openvdb::Int32Tree::LeafNodeType *> leaves;
        auto markGhost = [&](const tbb::blocked_range<size_t> &r)
        {
            auto status_axr = status[level]->getAccessor();
            for (auto liter = r.begin(); liter != r.end(); ++liter) {
                auto &leaf = *leaves[liter];
                for (auto offset = 0; offset < leaf.SIZE; ++offset) {
                    auto coord = leaf.offsetToGlobalCoord(offset);
                    if(!status_axr.isValueOn(coord) || status_axr.getValue(coord) != 2)
                        continue;
                    bool isGhost = false;
                    for(int ss = 0;ss<3;++ss)
                    for(int i = -1;i<=1;i+=2)
                    {
                        auto ipos = coord;
                        ipos[ss] += i;
                        if(status_axr.isValueOn(ipos) && status_axr.getValue(ipos) == 0)
                        {
                            isGhost = true;
                            break;
                        }
                    }
                    if(isGhost)
                        status_axr.setValue(coord, 1);
                }
            }
        };
        status[level]->tree().getNodes(leaves);
        tbb::parallel_for(tbb::blocked_range<size_t>(0, leaves.size()), markGhost);
    }
    for(int level = 0;level < levelNum;++level)
    {
        pressField[level]->clear();
        pressField[level]->setTree((std::make_shared<openvdb::FloatTree>(
                        status[level]->tree(), /*bgval*/ float(0),
                        openvdb::TopologyCopy())));
    }

}
struct generateAdaptiveGrid : zeno::INode{
    virtual void apply() override {
        auto coarse_grid = get_input("VDBGrid")->as<VDBFloatGrid>();
        float dt = get_input("dt")->as<zeno::NumericObject>()->get<float>();
        int levelNum = get_input("levelNum")->as<zeno::NumericObject>()->get<int>();
            
        auto data = zeno::IObject::make<agData>();
        data->initData(coarse_grid->m_grid, levelNum, dt);
        set_output("agData", data);
    }
};
ZENDEFNODE(generateAdaptiveGrid, {
        {"VDBGrid", "dt", "levelNum"},
        {"agData"},
        {},
        {"AdaptiveSolver"},
});

struct selectLevel : zeno::INode{
    virtual void apply() override {
        auto data = get_input("agData")->as<agData>();
        int level = get_input("level")->as<zeno::NumericObject>()->get<int>();

        auto result = zeno::IObject::make<VDBFloatGrid>();
        result->m_grid = openvdb::tools::fogToSdf(*(data->volumeField[level]), 0);
        set_output("sdf", result);
    }
};
ZENDEFNODE(selectLevel, {
        {"agData", "level"},
        {"sdf"},
        {},
        {"AdaptiveSolver"},
});

}