#include <zen/zen.h>
#include <zen/VDBGrid.h>
//#include "../../Library/MnBase/Meta/Polymorphism.h"
//openvdb::io::File(filename).write({grid});

namespace zenbase {

struct WriteVDBGrid : zen::INode {
  virtual void apply() override {
    auto path = std::get<std::string>(get_param("path"));
    auto data = get_input("data")->as<VDBGrid>();
    data->output(path);
  }
};


static int defWriteVDBGrid = zen::defNodeClass<WriteVDBGrid>("WriteVDBGrid",
    { /* inputs: */ {
    "data",
    }, /* outputs: */ {
    }, /* params: */ {
    {"string", "path", ""},
    }, /* category: */ {
    "openvdb",
    }});

}
