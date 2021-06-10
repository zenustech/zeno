#include <zen/VDBGrid.h>
#include <zen/zen.h>
#include <zen/StringObject.h>
//#include "../../Library/MnBase/Meta/Polymorphism.h"
// openvdb::io::File(filename).write({grid});

namespace zen {

static std::unique_ptr<VDBGrid> readvdb(std::string path, std::string type)
{
    std::unique_ptr<VDBGrid> data;
    if (type == "float") {
      data = zen::IObject::make<VDBFloatGrid>();
    } else if (type == "float3") {
      data = zen::IObject::make<VDBFloat3Grid>();
    } else if (type == "int") {
      data = zen::IObject::make<VDBIntGrid>();
    } else if (type == "int3") {
      data = zen::IObject::make<VDBInt3Grid>();
    } else if (type == "points") {
      data = zen::IObject::make<VDBPointsGrid>();
    } else {
      printf("%s\n", type.c_str());
      assert(0 && "bad VDBGrid type");
    }
    data->input(path);
    return data;
}


struct ReadVDBGrid : zen::INode {
  virtual void apply() override {
    auto path = std::get<std::string>(get_param("path"));
    auto type = std::get<std::string>(get_param("type"));
    auto data = readvdb(path, type);
    set_output("data", data);
  }
};
static int defReadVDBGrid = zen::defNodeClass<ReadVDBGrid>(
    "ReadVDBGrid", {/* inputs: */ {}, /* outputs: */
                    {
                        "data",
                    },
                    /* params: */
                    {
                        {"string", "type", "float"},
                        {"string", "path", ""},
                    },
                    /* category: */
                    {
                        "openvdb",
                    }});

struct ImportVDBGrid : zen::INode {
  virtual void apply() override {
    auto path = get_input("path")->as<zen::StringObject>();
    auto type = std::get<std::string>(get_param("type"));
    auto data = readvdb(path->get(), type);
    set_output("data", data);
  }
};

static int defImportVDBGrid = zen::defNodeClass<ImportVDBGrid>("ImportVDBGrid",
    { /* inputs: */ {
    "path",
    }, /* outputs: */ {
    "data",
    }, /* params: */ {
    {"string", "type", "float"},
    }, /* category: */ {
    "openvdb",
    }});


struct MakeVDBGrid : zen::INode {
  virtual void apply() override {
    auto dx = std::get<float>(get_param("dx"));
    auto type = std::get<std::string>(get_param("type"));
    auto structure = std::get<std::string>(get_param("structure"));
    auto name = std::get<std::string>(get_param("name"));
    std::unique_ptr<VDBGrid> data;
    if (type == "float") {
      auto tmp = zen::IObject::make<VDBFloatGrid>();
      auto transform = openvdb::math::Transform::createLinearTransform(dx);
      if(structure==std::string("vertex"))
        transform->postTranslate(openvdb::Vec3d{ -0.5,-0.5,-0.5 }*double(dx));
      tmp->m_grid->setTransform(transform);
      tmp->m_grid->setName(name);
      data = std::move(tmp);
    } else if (type == "float3") {
      auto tmp = zen::IObject::make<VDBFloat3Grid>();
      tmp->m_grid->setTransform(openvdb::math::Transform::createLinearTransform(dx));
      tmp->m_grid->setName(name);
      if (structure == "Staggered") {
        tmp->m_grid->setGridClass(openvdb::GridClass::GRID_STAGGERED);
      }
      data = std::move(tmp);
    } else if (type == "int") {
      auto tmp = zen::IObject::make<VDBIntGrid>();
      tmp->m_grid->setTransform(openvdb::math::Transform::createLinearTransform(dx));
      tmp->m_grid->setName(name);
      data = std::move(tmp);
    } else if (type == "int3") {
      auto tmp = zen::IObject::make<VDBInt3Grid>();
      tmp->m_grid->setTransform(openvdb::math::Transform::createLinearTransform(dx));
      tmp->m_grid->setName(name);
      data = std::move(tmp);
    } else if (type == "points") {
      auto tmp = zen::IObject::make<VDBPointsGrid>();
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

static int defMakeVDBGrid = zen::defNodeClass<MakeVDBGrid>(
    "MakeVDBGrid", {/* inputs: */ {}, /* outputs: */
                    {
                        "data",
                    },
                    /* params: */
                    {
                        {"float", "dx", "0.01"},
                        {"string", "type", "float"},
                        {"string", "structure", "Centered"},
                        {"string", "name", "Rename!"},
                    },
                    /* category: */
                    {
                        "openvdb",
                    }});

} // namespace zen
