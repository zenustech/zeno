#include "AdaptiveGridGen.h"
#include <openvdb/Types.h>
#include <tbb/parallel_for.h>
#include <cmath>
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
            velField[j][i] = openvdb::FloatGrid::create(0);
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
        dx[l] = dx[l-1] * 0.5;
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
    // init uniform grids on coarset grid
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
                        status_axr.setValue(coord, 0);
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

    }

    // subdivide the cell
    for(int level = 0;level < levelNum - 1;++level)
    {
        float criteria = level/levelNum + 0.01;
        float driftdx = dx[level + 1] * 0.5;
        {
            auto vol_axr = volumeField[level]->getAccessor();
            auto tem_axr = temperatureField[level]->getAccessor();
            auto status_axr = status[level]->getAccessor();
            auto fine_vol_axr = volumeField[level+1]->getAccessor();
            auto fine_tem_axr = temperatureField[level+1]->getAccessor();
            auto fine_press_axr = pressField[level+1]->getAccessor();
            auto fine_status_axr = status[level+1]->getAccessor();
            openvdb::tree::ValueAccessor<openvdb::FloatTree, true> vel_axr[3]=
                    {velField[0][level+1]->getAccessor(),
                    velField[1][level+1]->getAccessor(),
                    velField[2][level+1]->getAccessor()};
            for (openvdb::FloatTree::LeafCIter iter = volumeField[level]->tree().cbeginLeaf(); iter; ++iter) {
                const openvdb::FloatTree::LeafNodeType& leaf = *iter;
                for (auto offset = 0; offset < leaf.SIZE; ++offset) {
                    auto coord = leaf.offsetToGlobalCoord(offset);
                    if(!vol_axr.isValueOn(coord))
                        continue;
                    auto vol = vol_axr.getValue(coord);
                    if(vol > criteria)
                    {
                        auto tem = tem_axr.getValue(coord);
                        vol_axr.setValue(coord, 0);
                        tem_axr.setValue(coord, 0);
                        //status_axr.setValue(coord, 1);
                        auto wpos = volumeField[level]->indexToWorld(coord);
                        for(int i=-1;i<=1;i+=2)
                        for(int j=-1;j<=1;j+=2)
                        for(int k=-1;k<=1;k+=2)
                        {
                            openvdb::Vec3f drift = wpos + openvdb::Vec3f(i,j,k) * driftdx;
                            openvdb::Vec3d dindex = volumeField[level + 1]->worldToIndex(drift);
                            openvdb::Coord nindx = 
                                openvdb::Coord(round(dindex[0]), round(dindex[1]),round(dindex[2]));
                            fine_vol_axr.setValue(nindx, vol);
                            fine_tem_axr.setValue(nindx, tem);
                            fine_press_axr.setValue(nindx, 0);
                            fine_status_axr.setValue(nindx, 0);
                            for(int ii=0;ii<3;++ii)
                            for(int ss = 0;ss<=1;++ss)
                            {
                                auto ipos = nindx;
                                ipos[ii] += ss;
                                vel_axr[ii].setValue(ipos, 0);
                            }
                        }
                    }
                }
            }
            
        }
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