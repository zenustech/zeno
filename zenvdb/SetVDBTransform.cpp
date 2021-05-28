
#include <zen/zen.h>
#include <zen/VDBGrid.h>
#include <openvdb/tools/GridTransformer.h>
#include <openvdb/tools/Composite.h>

//#include "../../Library/MnBase/Meta/Polymorphism.h"
//openvdb::io::File(filename).write({grid});

namespace zenbase {

// struct SetVDBTransform : zen::INode {
//   virtual void apply() override {
//     auto dx = std::get<float>(get_param("dx"));
//     auto grid = get_input("grid")->as<VDBGrid>();
//     auto position = zen::get_float3<openvdb::Vec3f>(get_param("position"));
//     auto rotation = zen::get_float3<openvdb::Vec3f>(get_param("rotation"));
//     auto scale = zen::get_float3<openvdb::Vec3f>(get_param("scale"));

//     auto transform = openvdb::math::Transform::createLinearTransform(dx);
//     transform->postRotate(rotation[0], openvdb::math::X_AXIS);
//     transform->postRotate(rotation[1], openvdb::math::Y_AXIS);
//     transform->postRotate(rotation[2], openvdb::math::Z_AXIS);
//     grid->setTransform(transform);
//   }
// };


// static int defSetVDBTransform = zen::defNodeClass<SetVDBTransform>("SetVDBTransform",
//     { /* inputs: */ {
//     "grid",
//     }, /* outputs: */ {
//     }, /* params: */ {
//     {"float", "dx", "0.08 0"},
//     {"float3", "position", "0 0 0"},
//     {"float3", "rotation", "0 0 0"},
//     {"float3", "scale", "1 1 1"},
//     }, /* category: */ {
//     "openvdb",
//     }});
template <typename GridT>
void resampleVDB(typename GridT::Ptr source, typename GridT::Ptr target)
{
      const openvdb::math::Transform
      &sourceXform = source->transform(),
      &targetXform = target->transform();
      
      openvdb::Mat4R xform =
      sourceXform.baseMap()->getAffineMap()->getMat4() *
      targetXform.baseMap()->getAffineMap()->getMat4().inverse();
      openvdb::tools::GridTransformer transformer(xform);

      transformer.transformGrid<openvdb::tools::BoxSampler, GridT>(
      *source, *target);
      target->tree().prune();

}

struct  ResampleVDBGrid : zen::INode {
  virtual void apply() override {

    std::string targetType = get_input("resampleTo")->as<VDBGrid>()->getType();
    std::string sourceType = get_input("resampleFrom")->as<VDBGrid>()->getType();
    if(targetType == sourceType)
    {
      
        if(sourceType==std::string("FloatGrid"))
        {
          auto target = get_input("resampleTo")->as<VDBFloatGrid>();
          auto source = get_input("resampleFrom")->as<VDBFloatGrid>();
          resampleVDB<openvdb::FloatGrid>(source->m_grid, target->m_grid);
        }
        else if (sourceType==std::string("Vec3fGrid"))
        {
          auto target = get_input("resampleTo")->as<VDBFloat3Grid>();
          auto source = get_input("resampleFrom")->as<VDBFloat3Grid>();
          resampleVDB<openvdb::Vec3fGrid>(source->m_grid, target->m_grid);
        }
    }
  }
};

static int defResampleVDBGrid = zen::defNodeClass<ResampleVDBGrid>("ResampleVDBGrid",
     { /* inputs: */ {
     "resampleTo", "resampleFrom",
     }, /* outputs: */ {
     }, /* params: */ {
     }, /* category: */ {
     "openvdb",
     }});

struct CombineVDB : zen::INode{
  virtual void apply() override {

    std::string targetType = get_input("FieldA")->as<VDBGrid>()->getType();
    std::string sourceType = get_input("FieldB")->as<VDBGrid>()->getType();
    std::unique_ptr<VDBFloatGrid> dataf;
    std::unique_ptr<VDBFloat3Grid> dataf3;
    
    if(targetType == sourceType && targetType==std::string("FloatGrid"))
    {
        auto OpType = std::get<std::string>(get_param("OpType"));
        dataf = zen::IObject::make<VDBFloatGrid>();
        
        auto target = get_input("FieldA")->as<VDBFloatGrid>();
        auto source = get_input("FieldB")->as<VDBFloatGrid>();
        dataf->m_grid = target->m_grid->deepCopy();
        if(OpType==std::string("CSGUnion"))
        {
          openvdb::tools::csgUnion(*(dataf->m_grid), *(source->m_grid));
        }
        if(OpType==std::string("CSGIntersection"))
        {
          openvdb::tools::csgIntersection(*(dataf->m_grid), *(source->m_grid));
        }
        if(OpType==std::string("CSGDifference"))
        {
          openvdb::tools::csgDifference(*(dataf->m_grid), *(source->m_grid));
        }
        set_output("FieldOut", dataf);
    }
    
  }
};
static int defCombineVDB = zen::defNodeClass<CombineVDB>("CombineVDB",
     { /* inputs: */ {
     "FieldA", "FieldB",
     }, /* outputs: */ {
       "FieldOut",
     }, /* params: */ {
       {"float", "MultiplierA", "1"},
       {"float", "MultiplierB", "1"},
       {"string", "OpType", "CSGUnion"},
     }, /* category: */ {
     "openvdb",
     }});

}

