#include <zeno/zeno.h>
#include <zeno/StringObject.h>
#include <zeno/NumericObject.h>
#include <zeno/ZenoInc.h>
#include <zeno/VDBGrid.h>
#include <zeno/types/StringObject.h>
#include <zeno/types/NumericObject.h>
#include <filesystem>
#include <zeno/utils/type_traits.h>

//#include "../../Library/MnBase/Meta/Polymorphism.h"
// openvdb::io::File(filename).write({grid});

namespace zeno {
namespace {

static std::unique_ptr<VDBGrid> readGenericVDBGrid(const std::string &fn, uint32_t index=0) {
  using GridTypes = std::tuple
    < openvdb::points::PointDataGrid
    , openvdb::FloatGrid
    , openvdb::Vec3fGrid
    , openvdb::Int32Grid
    , openvdb::Vec3IGrid
    >;
  openvdb::io::File file(fn);
  file.open();
  openvdb::GridPtrVecPtr my_grids = file.getGrids();
  file.close();

	if (index >= my_grids->size()) index=0;

  std::unique_ptr<VDBGrid> grid;
  for (openvdb::GridPtrVec::iterator iter = my_grids->begin();
       iter != my_grids->end(); ++iter) {

	std::size_t idx = std::distance(my_grids->begin(), iter);
	if (idx != index) continue;

    openvdb::GridBase::Ptr it = *iter;
    if (zeno::static_for<0, std::tuple_size_v<GridTypes>>([&] (auto i) {
        using GridT = std::tuple_element_t<i, GridTypes>;
        if ((*iter)->isType<GridT>()) {
          
          auto pGrid = std::make_unique<VDBGridWrapper<GridT>>();
          pGrid->m_grid = openvdb::gridPtrCast<GridT>(*iter);
          grid = std::move(pGrid);
          return true;
        }
        return false;
      })) return grid;
  }
  throw zeno::Exception("failed to readGenericVDBGrid: " + fn);
}


#if 0
struct CacheVDBGrid : zeno::INode {
    int m_framecounter = 0;

    virtual void preApply(CalcContext* pContext) override {
        if (get_param_bool("mute")) {
            requireInput("inGrid", pContext);
            set_output("outGrid", get_input("inGrid"));
            return;
        }
        auto dir = zsString2Std(get_param_string("dir"));
        auto prefix = zsString2Std(get_param_string("prefix"));
        bool ignore = get_param_bool("ignore");
        if (!std::filesystem::is_directory(dir)) {
            std::filesystem::create_directory(dir);
        }
        int fno = m_framecounter++;
        if (has_input("frameNum")) {
            requireInput("frameNum", pContext);
            fno = get_input<zeno::NumericObject>("frameNum")->get<int>();
        }
        char buf[512];
        sprintf(buf, "%s%06d.vdb", prefix.c_str(), fno);
        auto path = (std::filesystem::path(dir) / buf).generic_string();
        if (ignore || !std::filesystem::exists(path)) {
            requireInput("inGrid", pContext);
            auto grid = safe_dynamic_cast<VDBGrid>(get_input("inGrid"));
            printf("dumping cache to [%s]\n", path.c_str());
            grid->output(path);
            set_output("outGrid", std::move(grid));
        } else {
            printf("using cache from [%s]\n", path.c_str());
            auto grid = readGenericVDBGrid(path);
            set_output("outGrid", std::move(grid));
        }
    }

    virtual void apply() override {
    }
};

ZENDEFNODE(CacheVDBGrid,
    { /* inputs: */ {
    {gParamType_VDBGrid,"inGrid", "", zeno::Socket_ReadOnly},
    {gParamType_Int, "frameNum"},
    }, /* outputs: */ {
    {gParamType_VDBGrid,"outGrid"},
    }, /* params: */ {
    {gParamType_String, "dir", "/tmp/cache"},
    {gParamType_String, "prefix", ""},
    {gParamType_Bool, "ignore", "0"},
    {gParamType_Bool, "mute", "0"},
    }, /* category: */ {
    "deprecated",
    }});
#endif

static std::unique_ptr<VDBGrid> readvdb(std::string path, std::string type, uint32_t index=0)
{
    if (type == "") {
      std::cout << "vdb read generic data" << std::endl;
      return readGenericVDBGrid(path, index);
    }
    std::unique_ptr<VDBGrid> data;
    if (type == "float") {
      data = std::make_unique<VDBFloatGrid>();
    } else if (type == "float3") {
      std::cout << "vdb read float3 data" << std::endl;
      data = std::make_unique<VDBFloat3Grid>();
    } else if (type == "int") {
      std::cout << "vdb read int data" << std::endl;
      data = std::make_unique<VDBIntGrid>();
    } else if (type == "int3") {
      std::cout << "vdb read int3 data" << std::endl;
      data = std::make_unique<VDBInt3Grid>();
    } else if (type == "points") {
      std::cout << "vdb read points data" << std::endl;
      data = std::make_unique<VDBPointsGrid>();
    } else {
      printf("%s\n", type.c_str());
      assert(0 && "bad VDBGrid type");
    }
    data->input(path);
    return data;
}


struct ReadVDBGrid : zeno::INode {
  virtual void apply() override {
    auto path = zsString2Std(get_param_string("path"));
    auto type = zsString2Std(get_param_string("type"));
    auto data = readvdb(path, type);
    set_output("data", std::move(data));
  }
};
static int defReadVDBGrid = zeno::defNodeClass<ReadVDBGrid>(
    "ReadVDBGrid", {/* inputs: */ {
                        {gParamType_String, "type", "", NoSocket},
                        {gParamType_String, "path", "", NoSocket, ReadPathEdit},}, /* outputs: */
                    {
                        {gParamType_VDBGrid,"data"},
                    },
                    /* params: */
                    {
                    },
                    /* category: */
                    {
                        "deprecated",
                    }});

struct ImportVDBGrid : zeno::INode {
  virtual void apply() override {
    auto path = zsString2Std(get_input2_string("path"));
    // auto type = zsString2Std(get_param_string("type"));
	auto index = get_input2_int("index");
    auto data = readvdb(path, "", index);
    set_output("data", std::move(data));
  }
};

static int defImportVDBGrid = zeno::defNodeClass<ImportVDBGrid>("ImportVDBGrid",
    { /* inputs: */ {
    {gParamType_String, "path", "", NoSocket, ReadPathEdit},
    }, /* outputs: */ {
    {gParamType_VDBGrid,"data"},
    }, /* params: */ {
    {gParamType_String, "type", ""},
    }, /* category: */ {
    "deprecated",
    }});

struct ReadVDB : ImportVDBGrid {
};
static int defReadVDB = zeno::defNodeClass<ReadVDB>("ReadVDB",
    { /* inputs: */ {
    {gParamType_String, "path", "", zeno::Socket_Primitve, zeno::ReadPathEdit},
	{"int", "index", "0"}
    }, /* outputs: */ {
    {gParamType_VDBGrid,"data"},
    }, /* params: */ {
    }, /* category: */ {
    "openvdb",
    }});


}
} // namespace zeno

