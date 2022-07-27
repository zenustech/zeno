#include <openvdb/openvdb.h>
#include <zeno/zeno.h>
#include <zeno/VDBGrid.h>
#include <openvdb/tools/GridTransformer.h>
#include <openvdb/tools/Composite.h>

//#include "../../Library/MnBase/Meta/Polymorphism.h"
//openvdb::io::File(filename).write({grid});

namespace zeno {

// struct SetVDBTransform : zeno::INode {
//   virtual void apply() override {
//     auto dx = get_param<float>(("dx"));
//     auto grid = get_input("grid")->as<VDBGrid>();
//     auto position = zeno::get_float3<openvdb::Vec3f>(get_param("position"));
//     auto rotation = zeno::get_float3<openvdb::Vec3f>(get_param("rotation"));
//     auto scale = zeno::get_float3<openvdb::Vec3f>(get_param("scale"));

//     auto transform = openvdb::math::Transform::createLinearTransform(dx);
//     transform->postRotate(rotation[0], openvdb::math::X_AXIS);
//     transform->postRotate(rotation[1], openvdb::math::Y_AXIS);
//     transform->postRotate(rotation[2], openvdb::math::Z_AXIS);
//     grid->setTransform(transform);
//   }
// };


// static int defSetVDBTransform = zeno::defNodeClass<SetVDBTransform>("SetVDBTransform",
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

struct  ResampleVDBGrid : zeno::INode {
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
        set_output("resampleTo", get_input("resampleTo"));
    } else {
        printf("ERROR: resample type mismarch!!");
    }
  }
};

static int defResampleVDBGrid = zeno::defNodeClass<ResampleVDBGrid>("ResampleVDBGrid",
     { /* inputs: */ {
     "resampleTo", "resampleFrom",
     }, /* outputs: */ {
     "resampleTo",
     }, /* params: */ {
     }, /* category: */ {
     "openvdb",
     }});

struct CombineVDB : zeno::INode{
  virtual void apply() override {

    std::string targetType = get_input("FieldA")->as<VDBGrid>()->getType();
    std::string sourceType = get_input("FieldB")->as<VDBGrid>()->getType();
    std::shared_ptr<VDBFloatGrid> dataf;
    
    if(targetType == sourceType && targetType==std::string("FloatGrid"))
    {
        auto OpType = get_param<std::string>(("OpType"));
        dataf = zeno::IObject::make<VDBFloatGrid>();
        
        auto target = get_input("FieldA")->as<VDBFloatGrid>();
        auto source = get_input("FieldB")->as<VDBFloatGrid>();
        if (get_param<bool>("writeBack")) {
            auto srcgrid = source->m_grid->deepCopy();
            if(OpType=="CSGUnion") {
              openvdb::tools::csgUnion(*(target->m_grid), *(srcgrid));
            } else if(OpType=="CSGIntersection") {
              openvdb::tools::csgIntersection(*(target->m_grid), *(srcgrid));
            } else if(OpType=="CSGDifference") {
              openvdb::tools::csgDifference(*(target->m_grid), *(srcgrid));
            }
            set_output("FieldOut", get_input("FieldA"));
        } else {
            auto result = std::make_shared<VDBFloatGrid>();
            if(OpType=="CSGUnion") {
              result->m_grid = openvdb::tools::csgUnionCopy(*(target->m_grid), *(source->m_grid));
            } else if(OpType=="CSGIntersection") {
              result->m_grid = openvdb::tools::csgIntersectionCopy(*(target->m_grid), *(source->m_grid));
            } else if(OpType=="CSGDifference") {
              result->m_grid = openvdb::tools::csgDifferenceCopy(*(target->m_grid), *(source->m_grid));
            }
            set_output("FieldOut", result);
        }
    }
    auto OpType = get_param<std::string>(("OpType"));
    if(OpType==std::string("Add"))
    {
      if(targetType == sourceType && targetType==std::string("FloatGrid")){
        auto target = get_input("FieldA")->as<VDBFloatGrid>();
        auto source = get_input("FieldB")->as<VDBFloatGrid>();
        auto srcgrid = source->m_grid->deepCopy();
        openvdb::tools::compSum(*(target->m_grid), *(srcgrid));
        set_output("FieldOut", get_input("FieldA"));
      }
      if(targetType == sourceType && targetType==std::string("Vec3fGrid")){
        auto target = get_input("FieldA")->as<VDBFloat3Grid>();
        auto source = get_input("FieldB")->as<VDBFloat3Grid>();
        auto srcgrid = source->m_grid->deepCopy();
        openvdb::tools::compSum(*(target->m_grid), *(srcgrid));
        set_output("FieldOut", get_input("FieldA"));
      }
    }
    if(OpType==std::string("Mul"))
    {
      if(targetType == sourceType && targetType==std::string("FloatGrid")){
        auto target = get_input("FieldA")->as<VDBFloatGrid>();
        auto source = get_input("FieldB")->as<VDBFloatGrid>();
        auto srcgrid = source->m_grid->deepCopy();
        openvdb::tools::compMul(*(target->m_grid), *(srcgrid));
        set_output("FieldOut", get_input("FieldA"));
      }
      if(targetType == sourceType && targetType==std::string("Vec3fGrid")){
        auto target = get_input("FieldA")->as<VDBFloat3Grid>();
        auto source = get_input("FieldB")->as<VDBFloat3Grid>();
        auto srcgrid = source->m_grid->deepCopy();
        openvdb::tools::compMul(*(target->m_grid), *(srcgrid));
        set_output("FieldOut", get_input("FieldA"));
      }
    }
    if(OpType==std::string("Replace"))
    {
      if(targetType == sourceType && targetType==std::string("FloatGrid")){
        auto target = get_input("FieldA")->as<VDBFloatGrid>();
        auto source = get_input("FieldB")->as<VDBFloatGrid>();
        auto srcgrid = source->m_grid->deepCopy();
        openvdb::tools::compReplace(*(target->m_grid), *(srcgrid));
        set_output("FieldOut", get_input("FieldA"));
      }
      if(targetType == sourceType && targetType==std::string("Vec3fGrid")){
        auto target = get_input("FieldA")->as<VDBFloat3Grid>();
        auto source = get_input("FieldB")->as<VDBFloat3Grid>();
        auto srcgrid = source->m_grid->deepCopy();
        openvdb::tools::compReplace(*(target->m_grid), *(srcgrid));
        set_output("FieldOut", get_input("FieldA"));
      }
    }
    
  }
};
static int defCombineVDB = zeno::defNodeClass<CombineVDB>("CombineVDB",
     { /* inputs: */ {
     "FieldA", "FieldB",
     }, /* outputs: */ {
       "FieldOut",
     }, /* params: */ {
       {"float", "MultiplierA", "1"},
       {"float", "MultiplierB", "1"},
       {"enum CSGUnion CSGIntersection CSGDifference Add Mul Replace A_Sample_B", "OpType", "CSGUnion"},
       {"bool", "writeBack", "0"},
     }, /* category: */ {
     "openvdb",
     }});


struct VDBDeactivate : zeno::INode
{
  virtual void apply() override {
    auto gType = get_input("Field")->as<VDBGrid>()->getType();
    auto mType = get_input("Mask")->as<VDBGrid>()->getType();
    if(gType == mType && gType==std::string("FloatGrid"))
    {
      auto const &grid = get_input<VDBFloatGrid>("Field")->m_grid;
      auto const &mask = get_input<VDBFloatGrid>("Mask")->m_grid;
      auto modifier = [&](auto &leaf, openvdb::Index leafpos) {
        for (auto iter = leaf.beginValueOn(); iter != leaf.endValueOn(); ++iter) {
            auto coord = iter.getCoord();
            if(mask->getAccessor().getValue(coord)==0)
            {
              iter.setValueOn(false);
            }
            else{
              iter.setValueOn(true);
            }

            //sdf.emplace_back(value);
        }
      };
      openvdb::tree::LeafManager<std::decay_t<decltype(grid->tree())>> leafman(grid->tree());
      leafman.foreach(modifier);
      openvdb::tools::prune(grid->tree());
    }
    if(gType == mType && gType==std::string("Vec3fGrid"))
    {
      auto const &grid = get_input<VDBFloat3Grid>("Field")->m_grid;
      auto const &mask = get_input<VDBFloat3Grid>("Mask")->m_grid;
      auto modifier = [&](auto &leaf, openvdb::Index leafpos) {
        for (auto iter = leaf.beginValueOn(); iter != leaf.endValueOn(); ++iter) {
            auto coord = iter.getCoord();
            if(mask->getAccessor().getValue(coord)[0]==0
            || mask->getAccessor().getValue(coord)[1]==0
            || mask->getAccessor().getValue(coord)[2]==0)
            {
              iter.setValueOn(false);
            }
            else{
              iter.setValueOn(true);
            }

            //sdf.emplace_back(value);
        }
      };
      openvdb::tree::LeafManager<std::decay_t<decltype(grid->tree())>> leafman(grid->tree());
      leafman.foreach(modifier);
      openvdb::tools::prune(grid->tree());
    }
  }
};
static int defVDBDeactivate = zeno::defNodeClass<VDBDeactivate>("VDBDeactivate",
     { /* inputs: */ {
     "Field", "Mask",
     }, /* outputs: */ {
     }, /* params: */ {
     }, /* category: */ {
     "openvdb",
     }});


#if 0 // TODO: datan help me
struct CopyVDBTopology : zeno::INode {
  virtual void apply() override {
    auto dst = get_input("copyTo")->as<VDBGrid>();
    auto src = get_input("copyFrom")->as<VDBGrid>();
    dst->copyTopologyFrom(src);
    set_output("copyTo", std::move(dst));
  }
};

static int defCopyVDBTopology = zeno::defNodeClass<CopyVDBTopology>("CopyVDBTopology",
     { /* inputs: */ {
     "copyTo", "copyFrom",
     }, /* outputs: */ {
     "copyTo",
     }, /* params: */ {
     }, /* category: */ {
     "openvdb",
     }});
#endif

}

