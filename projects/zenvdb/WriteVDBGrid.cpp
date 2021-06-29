#include <zeno/zeno.h>
#include <zeno/VDBGrid.h>
#include <zeno/StringObject.h>
//#include "../../Library/MnBase/Meta/Polymorphism.h"
//openvdb::io::File(filename).write({grid});

namespace zen {

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


struct ExportVDBGrid : zen::INode {
  virtual void apply() override {
    auto path = get_input("path")->as<zen::StringObject>();
    auto data = get_input("data")->as<VDBGrid>();
    data->output(path->get());
  }
};

static int defExportVDBGrid = zen::defNodeClass<ExportVDBGrid>("ExportVDBGrid",
    { /* inputs: */ {
    "data",
    "path",
    }, /* outputs: */ {
    }, /* params: */ {
    }, /* category: */ {
    "openvdb",
    }});

}
