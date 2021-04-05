#include <zen/zen.h>
#include <zen/VDBGrid.h>
//#include "../../Library/MnBase/Meta/Polymorphism.h"
//openvdb::io::File(filename).write({grid});

namespace zenbase {

struct ReadVDBGrid : zen::INode {
  virtual void apply() override {
    auto path = std::get<std::string>(get_param("path"));
    auto type = std::get<std::string>(get_param("type"));
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
  }
};


static int defReadVDBGrid = zen::defNodeClass<ReadVDBGrid>("ReadVDBGrid",
    { /* inputs: */ {
    }, /* outputs: */ {
    "data",
    }, /* params: */ {
    {"string", "path", ""},
    {"string", "type", "float"},
    }, /* category: */ {
    "openvdb",
    }});

}
