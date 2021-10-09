#include <zeno/zeno.h>
#include <zeno/VDBGrid.h>
#include <zeno/StringObject.h>
#include <zeno/ZenoInc.h>

//#include "../../Library/MnBase/Meta/Polymorphism.h"
//openvdb::io::File(filename).write({grid});

namespace zeno {

struct WriteVDBGrid : zeno::INode {
  virtual void apply() override {
    auto path = get_param<std::string>("path");
    auto data = get_input<VDBGrid>("data");
    data->output(path);
  }
};

static int defWriteVDBGrid = zeno::defNodeClass<WriteVDBGrid>("WriteVDBGrid",
    { /* inputs: */ {
    "data",
    }, /* outputs: */ {
    }, /* params: */ {
    {"writepath", "path", ""},
    }, /* category: */ {
    "openvdb",
    }});


struct ExportVDBGrid : zeno::INode {
  virtual void apply() override {
    auto path = get_input("path")->as<zeno::StringObject>();
    auto data = get_input("data")->as<VDBGrid>();
    data->output(path->get());
  }
};

static int defExportVDBGrid = zeno::defNodeClass<ExportVDBGrid>("ExportVDBGrid",
    { /* inputs: */ {
    "data",
    "path",
    }, /* outputs: */ {
    }, /* params: */ {
    }, /* category: */ {
    "openvdb",
    }});

}
