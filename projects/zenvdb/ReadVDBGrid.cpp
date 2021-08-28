#include <zeno/VDBGrid.h>
#include <zeno/zeno.h>
#include <zeno/StringObject.h>
#include <zeno/NumericObject.h>
#include <zeno/ZenoInc.h>

//#include "../../Library/MnBase/Meta/Polymorphism.h"
// openvdb::io::File(filename).write({grid});

namespace zeno {

// defined in CacheVDBGrid.cpp:
extern std::shared_ptr<VDBGrid> readGenericVDBGrid(const std::string &fn);

static std::shared_ptr<VDBGrid> readvdb(std::string path, std::string type)
{
    if (type == "") {
      return readGenericVDBGrid(path);
    }
    std::shared_ptr<VDBGrid> data;
    if (type == "float") {
      data = zeno::IObject::make<VDBFloatGrid>();
    } else if (type == "float3") {
      data = zeno::IObject::make<VDBFloat3Grid>();
    } else if (type == "int") {
      data = zeno::IObject::make<VDBIntGrid>();
    } else if (type == "int3") {
      data = zeno::IObject::make<VDBInt3Grid>();
    } else if (type == "points") {
      data = zeno::IObject::make<VDBPointsGrid>();
    } else {
      printf("%s\n", type.c_str());
      assert(0 && "bad VDBGrid type");
    }
    data->input(path);
    return data;
}


struct ReadVDBGrid : zeno::INode {
  virtual void apply() override {
    auto path = std::get<std::string>(get_param("path"));
    auto type = std::get<std::string>(get_param("type"));
    auto data = readvdb(path, type);
    set_output("data", data);
  }
};
static int defReadVDBGrid = zeno::defNodeClass<ReadVDBGrid>(
    "ReadVDBGrid", {/* inputs: */ {}, /* outputs: */
                    {
                        "data",
                    },
                    /* params: */
                    {
                        {"string", "type", ""},
                        {"string", "path", ""},
                    },
                    /* category: */
                    {
                        "openvdb",
                    }});

struct ImportVDBGrid : zeno::INode {
  virtual void apply() override {
    auto path = get_input("path")->as<zeno::StringObject>();
    auto type = std::get<std::string>(get_param("type"));
    auto data = readvdb(path->get(), type);
    set_output("data", std::move(data));
  }
};

static int defImportVDBGrid = zeno::defNodeClass<ImportVDBGrid>("ImportVDBGrid",
    { /* inputs: */ {
    "path",
    }, /* outputs: */ {
    "data",
    }, /* params: */ {
    {"string", "type", ""},
    }, /* category: */ {
    "openvdb",
    }});


struct MakeVDBGrid : zeno::INode {
  virtual void apply() override {
    auto dx = std::get<float>(get_param("dx"));
    if(has_input("Dx"))
    {
      dx = get_input("Dx")->as<NumericObject>()->get<float>();
    }
    auto type = std::get<std::string>(get_param("type"));
    auto structure = std::get<std::string>(get_param("structure"));
    auto name = std::get<std::string>(get_param("name"));
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
    "MakeVDBGrid", {/* inputs: */ {"Dx",}, /* outputs: */
                    {
                        "data",
                    },
                    /* params: */
                    {
                        {"float", "dx", "0.08"},
                        {"string", "type", "float"},
                        {"string", "structure", "Centered"},
                        {"string", "name", "Rename!"},
                    },
                    /* category: */
                    {
                        "openvdb",
                    }});

} // namespace zeno
