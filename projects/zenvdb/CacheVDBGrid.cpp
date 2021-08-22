#include <zeno/zeno.h>
#include <zeno/VDBGrid.h>
#include <zeno/types/StringObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/utils/filesystem.h>
#include <zeno/utils/fileio.h>

namespace zeno {


template <class GridTypes = std::tuple
    < openvdb::points::PointDataGrid
    , openvdb::FloatGrid
    , openvdb::Vec3fGrid
    , openvdb::Int32Grid
    , openvdb::Vec3IGrid
    >>
std::shared_ptr<VDBGrid> readGenericGrid(const std::string &fn) {
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
          grid = std::make_shared<VDBGridWrapper<GridT>>();
          grid->m_grid = openvdb::gridPtrCast<GridT>(*iter);
          return true;
        }
        return false;
      })) return grid;
  }
  throw zeno::Exception("failed to readGenericGrid: " + fn);
}


struct CacheVDBGrid : zeno::INode {
    virtual void doApply() override {
        if (has_option("MUTE")) {
            requireInput("inGrid");
            set_output("outGrid", get_input("inGrid"));
            return;
        }
        auto dir = get_param<std::string>("dir");
        auto fno = get_input<zeno::NumericObject>("frameNum")->get<int>();
        char buf[512];
        sprintf(buf, "%06d.vdb", fno);
        auto path = fs::path(dir) / buf;
        zeno::file_put_content();
        if (!fs::exists(path)) {
            requireInput("inGrid");
            auto grid = get_input<VDBGrid>("inGrid");
            grid->output(path);
            set_output("outGrid", std::move(grid));
        } else {
            auto grid = readGenericGrid(path);
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
    }, /* category: */ {
    "openvdb",
    }});


}
