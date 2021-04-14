#include <zen/zen.h>
#include <zen/MeshObject.h>
#include <zen/VDBGrid.h>
#include <omp.h>
#include "FLIP_vdb.h"
namespace zenbase{
struct FLIPCreator : zen::INode {
    virtual void apply() override {
        
        auto dx = std::get<float>(get_param("dx"));
        auto particles                = zen::IObject::make<VDBPointsGrid>();
        auto pressure                 = zen::IObject::make<VDBFloatGrid>();
        auto rhsgrid                  = zen::IObject::make<VDBFloatGrid>();
        auto face_weight              = zen::IObject::make<VDBFloat3Grid>();
        auto pressure_dofid           = zen::IObject::make<VDBIntGrid>();
        auto isolated_cell_dof        = zen::IObject::make<TBBConcurrentIntArray>();
        auto velocity                 = zen::IObject::make<VDBFloat3Grid>();
        auto velocity_update          = zen::IObject::make<VDBFloat3Grid>();
        auto velocity_snapshot        = zen::IObject::make<VDBFloat3Grid>();
        auto velocity_after_p2g       = zen::IObject::make<VDBFloat3Grid>();
        auto solid_velocity           = zen::IObject::make<VDBFloat3Grid>();
        auto velocity_weights         = zen::IObject::make<VDBFloat3Grid>();
        auto liquid_sdf               = zen::IObject::make<VDBFloatGrid>();
        auto liquid_sdf_snapshot      = zen::IObject::make<VDBFloatGrid>();
        auto pushed_out_liquid_sdf    = zen::IObject::make<VDBFloatGrid>();
        auto shrinked_liquid_sdf      = zen::IObject::make<VDBFloatGrid>();
        auto solid_sdf                = zen::IObject::make<VDBFloatGrid>();
        //auto boundary_velocity_volume = zen::IObject::make<VDBFloat3Grid>();

        auto voxel_center_transform = openvdb::math::Transform::createLinearTransform(dx);
	    auto voxel_vertex_transform = openvdb::math::Transform::createLinearTransform(dx);
	    voxel_vertex_transform->postTranslate(openvdb::Vec3d{ -0.5,-0.5,-0.5 }*double(dx));

        particles->m_grid = openvdb::points::PointDataGrid::create();
        particles->m_grid ->setTransform(voxel_center_transform);
        particles->m_grid ->setName("Particles");


        pressure->m_grid = openvdb::FloatGrid::create(float(0));
        pressure->m_grid->setTransform(voxel_center_transform);
        pressure->m_grid->setGridClass(openvdb::GridClass::GRID_FOG_VOLUME);
        pressure->m_grid->setName("Pressure");

        rhsgrid->m_grid = pressure->m_grid->deepCopy();
        rhsgrid->m_grid->setName("Divergence");

        //velocity
        velocity->m_grid = openvdb::Vec3fGrid::create(openvdb::Vec3f{ 0 });
        velocity->m_grid->setTransform(voxel_center_transform);
        velocity->m_grid->setGridClass(openvdb::GridClass::GRID_STAGGERED);
        velocity->m_grid->setName("Velocity");
        velocity_snapshot->m_grid = velocity->m_grid->deepCopy();
        velocity_snapshot->m_grid->setName("Velocity_Snapshot");
        velocity_update->m_grid = velocity->m_grid->deepCopy();
        velocity_update->m_grid->setName("Velocity_Update");
        solid_velocity->m_grid = openvdb::Vec3fGrid::create(openvdb::Vec3f{ 0,0,0 });
        solid_velocity->m_grid->setTransform(voxel_center_transform);
        solid_velocity->m_grid->setGridClass(openvdb::GridClass::GRID_STAGGERED);
        solid_velocity->m_grid->setName("Solid_Velocity");

        velocity_weights->m_grid = openvdb::Vec3fGrid::create(openvdb::Vec3f{ 0,0,0 });
        velocity_weights->m_grid->setTransform(voxel_center_transform);
        velocity_weights->m_grid->setName("Velocity_P2G_Weights");
        face_weight->m_grid = openvdb::Vec3fGrid::create(openvdb::Vec3f{ 0,0,0 });
        face_weight->m_grid->setName("Face_Weights");
        face_weight->m_grid->setTransform(voxel_center_transform);
        face_weight->m_grid->setGridClass(openvdb::GridClass::GRID_STAGGERED);
        liquid_sdf->m_grid = openvdb::FloatGrid::create(0.9f * dx);
        liquid_sdf->m_grid->setGridClass(openvdb::GridClass::GRID_LEVEL_SET);
        liquid_sdf->m_grid->setTransform(voxel_center_transform);
        liquid_sdf->m_grid->setName("Liquid_SDF");
        liquid_sdf_snapshot->m_grid = liquid_sdf->m_grid->deepCopy();
        pushed_out_liquid_sdf->m_grid = liquid_sdf->m_grid->deepCopy();

        solid_sdf->m_grid = openvdb::FloatGrid::create(3.f * dx);
        solid_sdf->m_grid->setTransform(voxel_vertex_transform);
        solid_sdf->m_grid->setGridClass(openvdb::GridClass::GRID_LEVEL_SET);
        solid_sdf->m_grid->setName("Solid_SDF");
        set_output("Particles",               particles               );
        set_output("Pressure",                pressure                );
        set_output("Divergence",              rhsgrid                 );
        set_output("CellFWeight",             face_weight             );
        set_output("PressureDOFID",           pressure_dofid          );
        set_output("IsolatedCellDOF",         isolated_cell_dof       );
        set_output("Velocity",                velocity                );
        set_output("DeltaVelocity",           velocity_update         );
        set_output("VelocitySnapshot",        velocity_snapshot       );
        set_output("PostAdvVelocity",         velocity_after_p2g      );
        set_output("SolidVelocity",           solid_velocity          );
        set_output("VelocityWeights",         velocity_weights        );
        set_output("LiquidSDF",               liquid_sdf              );
        set_output("LiquidSDFSnapshot",       liquid_sdf_snapshot     );
        set_output("ExtractedLiquidSDF",      pushed_out_liquid_sdf   );
        set_output("ErodedLiquidSDF",         shrinked_liquid_sdf     );
        set_output("SolidSDF",                solid_sdf               );
        //set_output("boundary_velocity_volume", boundary_velocity_volume);
    }
};

static int defFLIPCreator = zen::defNodeClass<FLIPCreator>("SetFLIPWorld",
 { 
            /* inputs: */ {
            }, 
            /* outputs: */ {
                    "Particles",               
                    "Pressure",                
                    "Divergence",                 
                    "CellFWeight",             
                    "PressureDOFID",          
                    "IsolatedCellDOF",       
                    "Velocity",                
                    "DeltaVelocity",         
                    "VelocitySnapshot",       
                    "PostAdvVelocity",      
                    "SolidVelocity",          
                    "VelocityWeights",        
                    "LiquidSDF",              
                    "LiquidSDFSnapshot",     
                    "ExtractedLiquidSDF",   
                    "ErodedLiquidSDF",     
                    "SolidSDF",               
                    //"acceleration_fields",     
                    //"domain_solid_sdf",        
                    //"boundary_velocity_volume",
            }, 
            /* params: */ {
                {"float", "dx", "0.08 0"},
            }, 
            /* category: */ {
                "FLIPSolver",
            }
 });

}

