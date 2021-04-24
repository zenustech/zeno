#if 0
#include <zen/zen.h>
//#include "../../Library/MnBase/Meta/Polymorphism.h"
//openvdb::io::File(filename).write({grid});

namespace zenbase {

struct WriteBgeo : zen::INode {
  virtual void apply() override {
    auto path = std::get<std::string>(get_param("path"));
    auto data = get_input("data")->as<VDBGrid>();
    data->output(path);
  }
};


static int defWriteBgeo = zen::defNodeClass<WriteBgeo>("WriteBgeo",
    { /* inputs: */ {
    "data",
    }, /* outputs: */ {
    }, /* params: */ {
    {"string", "path", ""},
    }, /* category: */ {
    "openvdb",
    }});

}
#endif