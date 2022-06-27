#include <zeno/zeno.h>
#include <zeno/types/PrimitiveIO.h>
#include <zeno/types/StringObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/utils/filesystem.h>

namespace zeno {


struct CachePrimitive : zeno::INode {
    int m_framecounter = 0;

    virtual void preApply() override {
        /*if (has_option("MUTE")) {
            requireInput("inPrim");
            set_output("outPrim", get_input("inPrim"));
            return;
        }*/
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
        sprintf(buf, "%s%06d.zpm", prefix.c_str(), fno);
        auto path = (fs::path(dir) / buf).generic_string();
        if (ignore || !fs::exists(path)) {
            requireInput("inPrim");
            auto prim = get_input<PrimitiveObject>("inPrim");
            printf("dumping cache to [%s]\n", path.c_str());
            writezpm(prim.get(), path.c_str());
            set_output("outPrim", std::move(prim));
        } else {
            printf("using cache from [%s]\n", path.c_str());
            auto prim = std::make_shared<PrimitiveObject>();
            readzpm(prim.get(), path.c_str());
            set_output("outPrim", std::move(prim));
        }
    }

    virtual void apply() override {
    }
};

ZENDEFNODE(CachePrimitive,
    { /* inputs: */ {
    "inPrim", "frameNum",
    }, /* outputs: */ {
    "outPrim",
    }, /* params: */ {
    {"string", "dir", "/tmp/cache"},
    {"string", "prefix", ""},
    {"bool", "ignore", "0"},
    }, /* category: */ {
    "deprecated",
    }});


}
