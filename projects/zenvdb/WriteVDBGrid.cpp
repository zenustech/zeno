#include <zeno/zeno.h>
#include <zeno/VDBGrid.h>
#include <zeno/StringObject.h>
#include <zeno/ZenoInc.h>
#include "zeno/utils/fileio.h"

namespace fs = std::filesystem;

//#include "../../Library/MnBase/Meta/Polymorphism.h"
//openvdb::io::File(filename).write({grid});

namespace zeno {

struct WriteVDBGrid : zeno::INode {
  virtual void apply() override {
    auto path = get_param<std::string>("path");
    path = create_directories_when_write_file(path);
    auto data = get_input<VDBGrid>("data");
    data->output(path);
  }
};

static int defWriteVDBGrid = zeno::defNodeClass<WriteVDBGrid>("WriteVDBGrid",
    { /* inputs: */ {
    {gParamType_VDBGrid, "data", "", zeno::Socket_ReadOnly},
    {gParamType_String, "path", "", NoSocket, WritePathEdit},
    }, /* outputs: */ {
    }, /* params: */ {
    }, /* category: */ {
    "deprecated",
    }});


struct ExportVDBGrid : zeno::INode {
  virtual void apply() override {
    auto path = get_input("path")->as<zeno::StringObject>()->get();
    auto folderPath = fs::path(path).parent_path();

    if (!fs::exists(folderPath)) {
        fs::create_directories(folderPath);
    }
    auto data = get_input("data")->as<VDBGrid>();
    data->output(path);
  }
};

static int defExportVDBGrid = zeno::defNodeClass<ExportVDBGrid>("ExportVDBGrid",
    { /* inputs: */ {
        {"VDBGrid", "data"},
        {gParamType_String, "path"},
    }, /* outputs: */ {
    }, /* params: */ {
    }, /* category: */ {
    "deprecated",
    }});
struct WriteVDB : ExportVDBGrid {
};

static int defWriteVDB = zeno::defNodeClass<WriteVDB>("WriteVDB",
    { /* inputs: */ {
        {"VDBGrid", "data"},
    {gParamType_String, "path", "", zeno::Socket_Primitve, zeno::WritePathEdit},
    }, /* outputs: */ {
    }, /* params: */ {
    }, /* category: */ {
    "openvdb",
    }});

}
