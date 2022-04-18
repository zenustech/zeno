#include <zeno/VDBGrid.h>
#include <zeno/zeno.h>
#include <zeno/StringObject.h>
#include <zeno/NumericObject.h>
#include <zeno/ZenoInc.h>

namespace zeno {

struct MakeVDBGrid : zeno::INode {
  virtual void apply() override {
    //auto dx = get_param<float>("dx"));
    float dx=0.08f;
    if(has_input("Dx"))
    {
      dx = get_input("Dx")->as<NumericObject>()->get<float>();
    }
    auto type = get_param<std::string>("type"));
    auto structure = get_param<std::string>("structure"));
    auto name = get_param<std::string>("name"));
    std::shared_ptr<VDBGrid> data;
    if (type == "float") {
      auto tmp = zeno::IObject::make<VDBFloatGrid>();
      auto transform = openvdb::math::Transform::createLinearTransform(dx);
      if(structure==std::string("vertex"))
        transform->postTranslate(openvdb::Vec3d{ -0.5,-0.5,-0.5 }*double(dx));
      tmp->m_grid->setTransform(transform);
      tmp->m_grid->setName(name);
      data = std::move(tmp);
    } else if (type == "float3") {
      auto tmp = zeno::IObject::make<VDBFloat3Grid>();
      tmp->m_grid->setTransform(openvdb::math::Transform::createLinearTransform(dx));
      tmp->m_grid->setName(name);
      if (structure == "Staggered") {
        tmp->m_grid->setGridClass(openvdb::GridClass::GRID_STAGGERED);
      }
      data = std::move(tmp);
    } else if (type == "int") {
      auto tmp = zeno::IObject::make<VDBIntGrid>();
      tmp->m_grid->setTransform(openvdb::math::Transform::createLinearTransform(dx));
      tmp->m_grid->setName(name);
      data = std::move(tmp);
    } else if (type == "int3") {
      auto tmp = zeno::IObject::make<VDBInt3Grid>();
      tmp->m_grid->setTransform(openvdb::math::Transform::createLinearTransform(dx));
      tmp->m_grid->setName(name);
      data = std::move(tmp);
    } else if (type == "points") {
      auto tmp = zeno::IObject::make<VDBPointsGrid>();
      tmp->m_grid->setTransform(openvdb::math::Transform::createLinearTransform(dx));
      tmp->m_grid->setName(name);
      data = std::move(tmp);
    } else {
      printf("%s\n", type.c_str());
      assert(0 && "bad VDBGrid type");
    }
    set_output("data", data);
  }
};

static int defMakeVDBGrid = zeno::defNodeClass<MakeVDBGrid>(
    "MakeVDBGrid", {/* inputs: */ {{"float","Dx","0.08"},}, /* outputs: */
                    {
                        "data",
                    },
                    /* params: */
                    {
                        //{"float", "dx", "0.08"},
                        {"enum float float3 int int3 points", "type", "float"},
                        {"enum vertex Centered Staggered", "structure", "Centered"},
                        {"string", "name", ""},
                    },
                    /* category: */
                    {
                        "openvdb",
                    }});

struct SetVDBGridName : zeno::INode {
    virtual void apply() override {
        auto grid = get_input<VDBGrid>("grid");
        auto name = get_param<std::string>("name");
        grid->setName(name);
        set_output("grid", std::move(grid));
    }
};

static int defSetVDBGridName = zeno::defNodeClass<SetVDBGridName>(
    "SetVDBGridName", {/* inputs: */ {}, /* outputs: */
                    {
                        "grid",
                    },
                    /* params: */
                    {
                        //{"float", "dx", "0.08"},
                        {"string", "name", "density"},
                    },
                    /* category: */
                    {
                        "openvdb",
                    }});

}
