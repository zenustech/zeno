#include <openvdb/openvdb.h>
#include <zeno/zeno.h>
#include <zeno/VDBGrid.h>
#include <openvdb/tools/GridTransformer.h>
#include <openvdb/tools/Composite.h>
#include <zeno/utils/interfaceutil.h>

//#include "../../Library/MnBase/Meta/Polymorphism.h"
//openvdb::io::File(filename).write({grid});

namespace zeno {

// struct SetVDBTransform : zeno::INode {
//   virtual void apply() override {
//     auto dx = get_param_float("dx");
//     auto grid = safe_dynamic_cast<VDBGrid>(get_input("grid"));
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
//     {gParamType_Float, "dx", "0.08 0"},
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

    std::string targetType = safe_dynamic_cast<VDBGrid>(get_input("resampleTo"))->getType();
    std::string sourceType = safe_dynamic_cast<VDBGrid>(get_input("resampleFrom"))->getType();
    if(targetType == sourceType)
    {
        if(sourceType== "FloatGrid")
        {
            auto target = safe_dynamic_cast<VDBFloatGrid>(get_input("resampleTo"));
            auto source = safe_dynamic_cast<VDBFloatGrid>(get_input("resampleFrom"));
            resampleVDB<openvdb::FloatGrid>(source->m_grid, target->m_grid);
        }
        else if (sourceType== "Vec3fGrid")
        {
            auto target = safe_dynamic_cast<VDBFloat3Grid>(get_input("resampleTo"));
            auto source = safe_dynamic_cast<VDBFloat3Grid>(get_input("resampleFrom"));
            resampleVDB<openvdb::Vec3fGrid>(source->m_grid, target->m_grid);
        }
        set_output("resampleTo", clone_input("resampleTo"));
    } else {
        printf("ERROR: resample type mismarch!!");
    }
  }
};

static int defResampleVDBGrid = zeno::defNodeClass<ResampleVDBGrid>("ResampleVDBGrid",
     { /* inputs: */ {
         {gParamType_VDBGrid,"resampleTo", "", zeno::Socket_ReadOnly},
         {gParamType_VDBGrid,"resampleFrom", "", zeno::Socket_ReadOnly},
     }, /* outputs: */ {
         {gParamType_VDBGrid,"resampleTo"},
     }, /* params: */ {
     }, /* category: */ {
     "openvdb",
     }});

struct CombineVDB : zeno::INode{
  virtual void apply() override {

    std::string targetType = safe_dynamic_cast<VDBGrid>(get_input("FieldA"))->getType();
    std::string sourceType = safe_dynamic_cast<VDBGrid>(get_input("FieldB"))->getType();

    if(targetType == sourceType && targetType== "FloatGrid")
    {
        auto OpType = zsString2Std(get_param_string("OpType"));
        auto target = safe_uniqueptr_cast<VDBFloatGrid>(clone_input("FieldA"));
        auto source = safe_uniqueptr_cast<VDBFloatGrid>(clone_input("FieldB"));
        if (get_param_bool("writeBack")) {
            auto srcgrid = source->m_grid->deepCopy();
            if(OpType=="CSGUnion") {
              openvdb::tools::csgUnion(*(target->m_grid), *(srcgrid));
            } else if(OpType=="CSGIntersection") {
              openvdb::tools::csgIntersection(*(target->m_grid), *(srcgrid));
            } else if(OpType=="CSGDifference") {
              openvdb::tools::csgDifference(*(target->m_grid), *(srcgrid));
            }
            set_output("FieldOut", std::move(target));
        } else {
            auto result = std::make_unique<VDBFloatGrid>();
            if(OpType=="CSGUnion") {
              result->m_grid = openvdb::tools::csgUnionCopy(*(target->m_grid), *(source->m_grid));
            } else if(OpType=="CSGIntersection") {
              result->m_grid = openvdb::tools::csgIntersectionCopy(*(target->m_grid), *(source->m_grid));
            } else if(OpType=="CSGDifference") {
              result->m_grid = openvdb::tools::csgDifferenceCopy(*(target->m_grid), *(source->m_grid));
            }
            set_output("FieldOut", std::move(result));
        }
    }
    auto OpType = zsString2Std(get_param_string("OpType"));
    if(OpType== "Add")
    {
      if(targetType == sourceType && targetType== "FloatGrid"){
        auto target = safe_uniqueptr_cast<VDBFloatGrid>(clone_input("FieldA"));
        auto source = safe_uniqueptr_cast<VDBFloatGrid>(clone_input("FieldB"));
        auto srcgrid = source->m_grid->deepCopy();
        openvdb::tools::compSum(*(target->m_grid), *(srcgrid));
        set_output("FieldOut", std::move(target));
      }
      if(targetType == sourceType && targetType== "Vec3fGrid"){
        auto target = safe_uniqueptr_cast<VDBFloat3Grid>(clone_input("FieldA"));
        auto source = safe_uniqueptr_cast<VDBFloat3Grid>(clone_input("FieldB"));
        auto srcgrid = source->m_grid->deepCopy();
        openvdb::tools::compSum(*(target->m_grid), *(srcgrid));
        set_output("FieldOut", std::move(target));
      }
    }
    if(OpType== "Mul")
    {
      if(targetType == sourceType && targetType== "FloatGrid"){
        auto target = safe_uniqueptr_cast<VDBFloatGrid>(clone_input("FieldA"));
        auto source = safe_uniqueptr_cast<VDBFloatGrid>(clone_input("FieldB"));
        auto srcgrid = source->m_grid->deepCopy();
        openvdb::tools::compMul(*(target->m_grid), *(srcgrid));
        set_output("FieldOut", std::move(target));
      }
      if(targetType == sourceType && targetType== "Vec3fGrid"){
        auto target = safe_uniqueptr_cast<VDBFloat3Grid>(clone_input("FieldA"));
        auto source = safe_uniqueptr_cast<VDBFloat3Grid>(clone_input("FieldB"));
        auto srcgrid = source->m_grid->deepCopy();
        openvdb::tools::compMul(*(target->m_grid), *(srcgrid));
        set_output("FieldOut", std::move(target));
      }
    }
    if(OpType== "Replace")
    {
      if(targetType == sourceType && targetType== "FloatGrid"){
        auto target = safe_uniqueptr_cast<VDBFloatGrid>(clone_input("FieldA"));
        auto source = safe_uniqueptr_cast<VDBFloatGrid>(clone_input("FieldB"));
        auto srcgrid = source->m_grid->deepCopy();
        openvdb::tools::compReplace(*(target->m_grid), *(srcgrid));
        set_output("FieldOut", std::move(target));
      }
      if(targetType == sourceType && targetType== "Vec3fGrid"){
        auto target = safe_uniqueptr_cast<VDBFloat3Grid>(clone_input("FieldA"));
        auto source = safe_uniqueptr_cast<VDBFloat3Grid>(clone_input("FieldB"));
        auto srcgrid = source->m_grid->deepCopy();
        openvdb::tools::compReplace(*(target->m_grid), *(srcgrid));
        set_output("FieldOut", std::move(target));
      }
    }
    
  }
};
static int defCombineVDB = zeno::defNodeClass<CombineVDB>("CombineVDB",
     { /* inputs: */ {
         {gParamType_VDBGrid,"FieldA", "", zeno::Socket_ReadOnly},
         {gParamType_VDBGrid,"FieldB", "", zeno::Socket_ReadOnly},
     }, /* outputs: */ {
         {gParamType_VDBGrid,"FieldOut"},
     }, /* params: */ {
       {gParamType_Float, "MultiplierA", "1"},
       {gParamType_Float, "MultiplierB", "1"},
       {"enum CSGUnion CSGIntersection CSGDifference Add Mul Replace A_Sample_B", "OpType", "CSGUnion"},
       {gParamType_Bool, "writeBack", "0"},
     }, /* category: */ {
     "openvdb",
     }});


struct VDBDeactivate : zeno::INode
{
  virtual void apply() override {
    auto gType = safe_dynamic_cast<VDBGrid>(get_input("Field"))->getType();
    auto mType = safe_dynamic_cast<VDBGrid>(get_input("Mask"))->getType();
    if(gType == mType && gType== "FloatGrid")
    {
      auto const &grid = safe_dynamic_cast<VDBFloatGrid>(get_input("Field"))->m_grid;
      auto const &mask = safe_dynamic_cast<VDBFloatGrid>(get_input("Mask"))->m_grid;
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
    if(gType == mType && gType== "Vec3fGrid")
    {
      auto const &grid = safe_dynamic_cast<VDBFloat3Grid>(get_input("Field"))->m_grid;
      auto const &mask = safe_dynamic_cast<VDBFloat3Grid>(get_input("Mask"))->m_grid;
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
         {gParamType_VDBGrid,"Field", "", zeno::Socket_ReadOnly},
         {gParamType_VDBGrid,"Mask", "", zeno::Socket_ReadOnly},
     }, /* outputs: */ {
     }, /* params: */ {
     }, /* category: */ {
     "openvdb",
     }});


#if 0 // TODO: datan help me
struct CopyVDBTopology : zeno::INode {
  virtual void apply() override {
    auto dst = safe_dynamic_cast<VDBGrid>(get_input("copyTo"));
    auto src = safe_dynamic_cast<VDBGrid>(get_input("copyFrom"));
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

