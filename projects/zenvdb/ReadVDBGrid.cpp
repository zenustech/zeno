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
      std::cout << "vdb read generic data" << std::endl;
      return readGenericVDBGrid(path);
    }
    std::shared_ptr<VDBGrid> data;
    if (type == "float") {
      data = zeno::IObject::make<VDBFloatGrid>();
    } else if (type == "float3") {
      std::cout << "vdb read float3 data" << std::endl;
      data = zeno::IObject::make<VDBFloat3Grid>();
    } else if (type == "int") {
      std::cout << "vdb read int data" << std::endl;
      data = zeno::IObject::make<VDBIntGrid>();
    } else if (type == "int3") {
      std::cout << "vdb read int3 data" << std::endl;
      data = zeno::IObject::make<VDBInt3Grid>();
    } else if (type == "points") {
      std::cout << "vdb read points data" << std::endl;
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
    auto path = get_param<std::string>("path"));
    auto type = get_param<std::string>("type"));
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
                        {"readpath", "path", ""},
                    },
                    /* category: */
                    {
                        "openvdb",
                    }});

struct ImportVDBGrid : zeno::INode {
  virtual void apply() override {
    auto path = get_input("path")->as<zeno::StringObject>();
    auto type = get_param<std::string>("type"));
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

} // namespace zeno
