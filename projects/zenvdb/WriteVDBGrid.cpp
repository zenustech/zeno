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
    auto path = zsString2Std(get_param_string("path"));
    path = create_directories_when_write_file(path);
    auto data = safe_dynamic_cast<VDBGrid>(get_input("data"));
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
    auto path = zsString2Std(get_input2_string("path"));
    auto folderPath = fs::path(path).parent_path();

    if (!fs::exists(folderPath)) {
        fs::create_directories(folderPath);
    }
    auto data = safe_dynamic_cast<VDBGrid>(get_input("data"));
    data->output(path);
  }
};

static int defExportVDBGrid = zeno::defNodeClass<ExportVDBGrid>("ExportVDBGrid",
    { /* inputs: */ {
        {gParamType_VDBGrid,"data"},
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
        {gParamType_VDBGrid,"data"},
    {gParamType_String, "path", "", zeno::Socket_Primitve, zeno::WritePathEdit},
    }, /* outputs: */ {
    }, /* params: */ {
    }, /* category: */ {
    "openvdb",
    }});

}
