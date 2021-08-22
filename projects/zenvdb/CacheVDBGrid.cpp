#include <zeno/zeno.h>
#include <zeno/VDBGrid.h>
#include <zeno/types/StringObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/utils/filesystem.h>


struct CacheVDBGrid : zeno::INode {
    virtual void doGpply() override {
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
        if (!fs::is_file(path)) {
            requireInput("inGrid");
            auto grid = get_input<VDBGrid>("inGrid");
            grid->output(path.str());
            set_output("outGrid", std::move(grid));
        } else {
            grid->input(path.str());
            set_output("outGrid", std::move(grid));
        }
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
