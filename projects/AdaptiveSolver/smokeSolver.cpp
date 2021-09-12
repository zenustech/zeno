#include "smokeSolver.h"
#include <openvdb/tools/Interpolation.h>
#include "tbb/blocked_range3d.h"
namespace zeno{
    void smokeData::initData(openvdb::FloatGrid::Ptr sdf, float inputdt)
    {
        dt = inputdt;
        dx = sdf->voxelSize()[0];
        temperatureField = zeno::IObject::make<VDBFloatGrid>()->m_grid;
        volumeField = zeno::IObject::make<VDBFloatGrid>()->m_grid;
        pressField = zeno::IObject::make<VDBFloatGrid>()->m_grid;
        velField = zeno::IObject::make<VDBFloat3Grid>()->m_grid;

        // set up transform
        auto transform = sdf->transformPtr()->copy();
        temperatureField->setTransform(transform);
        volumeField->setTransform(transform);
        pressField->setTransform(transform);

        auto inputbbox = openvdb::CoordBBox();
        auto is_valid = sdf->tree().evalLeafBoundingBox(inputbbox);
        if (!is_valid) {
            return;
        }
        worldinputbbox = openvdb::BBoxd(sdf->indexToWorld(inputbbox.min()),
                                       sdf->indexToWorld(inputbbox.max()));
        worldinputbbox.min() += openvdb::Vec3d(-3);
        worldinputbbox.max() -= openvdb::Vec3d(-3);
        
        auto loopbegin =
            openvdb::tools::local_util::floorVec3(sdf->worldToIndex(worldinputbbox.min()));
        auto loopend =
            openvdb::tools::local_util::ceilVec3(sdf->worldToIndex(worldinputbbox.max()));

        auto pispace_range = tbb::blocked_range3d<int, int, int>(
            loopbegin.x(), loopend.x(), loopbegin.y(), loopend.y(), loopbegin.z(),
            loopend.z());
        auto mark_active_point_leaves =
            [&](const tbb::blocked_range3d<int, int, int> &r) {
                auto sdf_axr{sdf->getConstAccessor()};
                auto temperature_axr{temperatureField->getAccessor()};
                auto volume_axr{volumeField->getAccessor()};
                for (int i = r.pages().begin(); i < r.pages().end(); i += 1) {
                for (int j = r.rows().begin(); j < r.rows().end(); j += 1) {
                    for (int k = r.cols().begin(); k < r.cols().end(); k += 1) {
                        openvdb::Coord coord(i,j,k);
                        if(!sdf_axr.isValueOn(coord) || sdf_axr.getValue(coord) >= 0)
                        {
                            temperature_axr.setValue(coord, 0);
                            volume_axr.setValue(coord, 0);
                        }
                        else
                        {
                            temperature_axr.setValue(coord, 1);
                            volume_axr.setValue(coord, 1);
                        }
                    }     // loop k
                }       // loop j
                }         // loop i
            };          // end mark active point elaves
        tbb::parallel_for(pispace_range, mark_active_point_leaves);

        pressField->setTree((std::make_shared<openvdb::FloatTree>(
                    temperatureField->tree(), /*bgval*/ float(0),
                    openvdb::TopologyCopy())));
        velField->setTree((std::make_shared<openvdb::Vec3fTree>(
                    temperatureField->tree(), /*bgval*/ openvdb::Vec3f(0),
                    openvdb::TopologyCopy())));
        auto velTrans = transform->copy();
        velTrans->postTranslate(openvdb::Vec3d{ -0.5,-0.5,-0.5 }*double(dx));
        velField->setTransform(velTrans);
    }

    void smokeData::advection()
    {
        std::vector<openvdb::FloatTree::LeafNodeType *> leaves;
        auto new_temField = temperatureField->deepCopy();
        auto new_volField = volumeField->deepCopy();
        int sign = 1;
        auto semiLangAdvection = [&](const tbb::blocked_range<size_t> &r) {
            auto vel_axr = velField->getConstAccessor();
            auto tem_axr = temperatureField->getConstAccessor();
            auto vol_axr = volumeField->getConstAccessor();
            auto new_tem_axr{new_temField->getAccessor()};
            auto new_vol_axr{new_volField->getAccessor()};
            // leaf iter
            for (auto liter = r.begin(); liter != r.end(); ++liter) {
                auto &leaf = *leaves[liter];
                for (auto offset = 0; offset < leaf.SIZE; ++offset) {
                    openvdb::Coord coord = leaf.offsetToGlobalCoord(offset);
                    if(!vol_axr.isValueOn(coord))
                        continue;
                    auto wpos = temperatureField->indexToWorld(coord);
                    auto vel = openvdb::tools::BoxSampler::sample(
                            vel_axr, wpos);
                    
                    auto midwpos = wpos - sign * vel * 0.5 * dt;
                    auto midvel = openvdb::tools::BoxSampler::sample(
                            vel_axr, midwpos);
                    auto pwpos = wpos - sign * midvel * dt;
                    auto volume = openvdb::tools::BoxSampler::sample(
                            vol_axr, pwpos);
                    auto tem = openvdb::tools::BoxSampler::sample(
                            tem_axr, pwpos);
                    new_tem_axr.setValue(coord, tem);
                    new_vol_axr.setValue(coord, volume);
                }
            }
        };

        volumeField->tree().getNodes(leaves);
        tbb::parallel_for(tbb::blocked_range<size_t>(0, leaves.size()), semiLangAdvection);
        sign = -1;
        auto temBuffer = temperatureField->deepCopy(); temperatureField->clear(); temperatureField = new_temField->deepCopy();
        auto volBuffer = volumeField->deepCopy(); volumeField->clear(); volumeField=new_volField->deepCopy();
        tbb::parallel_for(tbb::blocked_range<size_t>(0, leaves.size()), semiLangAdvection);
        
        auto computeNewField = [&](const tbb::blocked_range<size_t> &r){
            auto tem_axr = temBuffer->getConstAccessor();
            auto vol_axr = volBuffer->getConstAccessor();
            auto vel_axr = velField->getConstAccessor();
            auto new_tem_axr{new_temField->getAccessor()};
            auto new_vol_axr{new_volField->getAccessor()};
            for (auto liter = r.begin(); liter != r.end(); ++liter) {
                auto &leaf = *leaves[liter];
                for (auto offset = 0; offset < leaf.SIZE; ++offset) {
                    openvdb::Coord coord = leaf.offsetToGlobalCoord(offset);
                    if(!vol_axr.isValueOn(coord))
                        continue;
                    new_tem_axr.setValue(coord, 
                        1.5 * tem_axr.getValue(coord) - 0.5 * new_tem_axr.getValue(coord));
                    new_vol_axr.setValue(coord, 
                        1.5 * vol_axr.getValue(coord) - 0.5 * new_vol_axr.getValue(coord));
                
                }
            }
        };
        tbb::parallel_for(tbb::blocked_range<size_t>(0, leaves.size()), computeNewField);
        sign = 1;
        volumeField->clear();volumeField = new_volField->deepCopy();
        temperatureField->clear();temperatureField = new_temField->deepCopy();
        
        tbb::parallel_for(tbb::blocked_range<size_t>(0, leaves.size()), semiLangAdvection);
        volumeField->clear();temperatureField->clear();
        volumeField = new_volField->deepCopy();
        temperatureField = new_temField->deepCopy();
        
        leaves.clear();
        velField->tree().getNodes(leaves);
        auto new_vel = velField->deepCopy();
        auto inte_vel = velField;
        sign = 1;
        auto velAdvection = [&](const tbb::blocked_range<size_t> &r){
            auto vel_axr = velField->getConstAccessor();
            auto inte_axr = inte_vel->getConstAccessor();
            auto new_vel_axr = new_vel->getAccessor();
            // leaf iter
            for (auto liter = r.begin(); liter != r.end(); ++liter) {
                auto &leaf = *leaves[liter];
                for (auto offset = 0; offset < leaf.SIZE; ++offset) {
                    openvdb::Coord coord = leaf.offsetToGlobalCoord(offset);
                    if(!vel_axr.isValueOn(coord))
                        continue;
                    auto wpos = velField->indexToWorld(coord);
                    auto vel = vel_axr.getValue(coord);
                    
                    auto midwpos = wpos - sign *0.5 * dt * vel;
                    auto midvel = openvdb::tools::BoxSampler::sample(
                            inte_axr, midwpos);
                    
                    auto pwpos = wpos - sign * dt * midvel;
                    auto vel = openvdb::tools::BoxSampler::sample(
                            inte_axr, pwpos);
                    new_vel_axr.setValue(coord, vel);
                }
            }
        };
        sign = -1;
        inte_vel = new_vel->deepCopy();
        tbb::parallel_for(tbb::blocked_range<size_t>(0, leaves.size()), velAdvection);
        
        auto computeNewVel = [&](const tbb::blocked_range<size_t> &r){
            auto vel_axr = velField->getConstAccessor();
            auto inte_axr = inte_vel->getConstAccessor();
            auto new_vel_axr = new_vel->getAccessor();
            // leaf iter
            for (auto liter = r.begin(); liter != r.end(); ++liter) {
                auto &leaf = *leaves[liter];
                for (auto offset = 0; offset < leaf.SIZE; ++offset) {
                    openvdb::Coord coord = leaf.offsetToGlobalCoord(offset);
                    if(!vel_axr.isValueOn(coord))
                        continue;
                    new_vel_axr.setValue(coord, 1.5 *vel_axr.getValue(coord) - 0.5 * new_vel_axr.getValue(coord));
                }
            }
        };   
        
        tbb::parallel_for(tbb::blocked_range<size_t>(0, leaves.size()), computeNewVel);
        inte_vel->clear();inte_vel = new_vel->deepCopy();
        sign = 1;
        tbb::parallel_for(tbb::blocked_range<size_t>(0, leaves.size()), velAdvection);
        inte_vel->clear();velField->clear();
        velField = new_vel->deepCopy();
    }

    void smokeData::applyOuterforce()
    {
        float alpha =-0.1, beta = 0.2;
        
        std::vector<openvdb::FloatTree::LeafNodeType *> leaves;
        auto applyOuterForce = [&](const tbb::blocked_range<size_t> &r) {
            auto vel_axr = velField->getAccessor();
            auto vol_axr = volumeField->getAccessor();
            auto tem_axr = temperatureField->getAccessor();
            for (auto liter = r.begin(); liter != r.end(); ++liter) {
                auto &leaf = *leaves[liter];
                for (auto offset = 0; offset < leaf.SIZE; ++offset) {
                    openvdb::Coord coord = leaf.offsetToGlobalCoord(offset);
                    if(!vel_axr.isValueOn(coord))
                        continue;
                    
                    float dens = (alpha * vol_axr.getValue(coord) -beta * tem_axr.getValue(coord));
                    auto vel = vel_axr.getValue(coord);
                    auto deltaV = vel - dens * 9.8 * dt;
                    vel_axr.setValue(coord, deltaV);
                }
            }
        };
        velField->tree().getNodes(leaves);
        tbb::parallel_for(tbb::blocked_range<size_t>(0, leaves.size()), applyOuterForce);
        
    }
    
    void solvePress(){
        
    }
    struct SDFtoSmoke : zeno::INode{
        virtual void apply() override {
            auto sdf = get_input("mesh")->as<zeno::VDBFloatGrid>();
            auto data = zeno::IObject::make<smokeData>();
            float dt = get_input("dt")->as<zeno::NumericObject>()->get<float>();
            data->initData(sdf->m_grid, dt);
            set_output("smokeData", data);
        }
    };
    ZENDEFNODE(SDFtoSmoke,
    {
        {"sdfgrid", "dt"},
        {"smokeData"},
        {},
        {"AdaptiveSolver"},
    }
    );

}