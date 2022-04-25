#include <zeno/zeno.h>
#include <zeno/VDBGrid.h>
#include <zeno/types/StringObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/utils/filesystem.h>
#include <zeno/utils/type_traits.h>

namespace zeno {

static std::shared_ptr<VDBGrid> readGenericVDBGrid(const std::string &fn) {
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
  std::shared_ptr<VDBGrid> grid;
  for (openvdb::GridPtrVec::iterator iter = my_grids->begin();
       iter != my_grids->end(); ++iter) {
    openvdb::GridBase::Ptr it = *iter;
    if (zeno::static_for<0, std::tuple_size_v<GridTypes>>([&] (auto i) {
        using GridT = std::tuple_element_t<i, GridTypes>;
        if ((*iter)->isType<GridT>()) {
          
          auto pGrid = std::make_shared<VDBGridWrapper<GridT>>();
          pGrid->m_grid = openvdb::gridPtrCast<GridT>(*iter);
          grid = pGrid;
          return true;
        }
        return false;
      })) return grid;
  }
  throw zeno::Exception("failed to readGenericVDBGrid: " + fn);
}


struct CacheVDBGrid : zeno::INode {
    int m_framecounter = 0;

    virtual void preApply() override {
        if (get_param<bool>("mute")) {
            requireInput("inGrid");
            set_output("outGrid", get_input("inGrid"));
            return;
        }
        auto dir = get_param<std::string>("dir");
        auto prefix = get_param<std::string>("prefix");
        bool ignore = get_param<bool>("ignore");
        if (!fs::is_directory(dir)) {
            fs::create_directory(dir);
        }
        int fno = m_framecounter++;
        if (has_input("frameNum")) {
            requireInput("frameNum");
            fno = get_input<zeno::NumericObject>("frameNum")->get<int>();
        }
        char buf[512];
        sprintf(buf, "%s%06d.vdb", prefix.c_str(), fno);
        auto path = (fs::path(dir) / buf).generic_string();
        if (ignore || !fs::exists(path)) {
            requireInput("inGrid");
            auto grid = get_input<VDBGrid>("inGrid");
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
    "inGrid", "frameNum",
    }, /* outputs: */ {
    "outGrid",
    }, /* params: */ {
    {"string", "dir", "/tmp/cache"},
    {"string", "prefix", ""},
    {"bool", "ignore", "0"},
    {"bool", "mute", "0"},
    }, /* category: */ {
    "openvdb",
    }});


}
