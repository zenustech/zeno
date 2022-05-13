#include <zeno/zeno.h>
#include <zeno/types/CurveObject.h>
#include <sstream>

namespace zeno {

struct MakeCurve : zeno::INode {
    virtual void apply() override {
        auto curvemap = std::make_shared<zeno::CurveObject>();
        auto _points = get_param<std::string>("_POINTS");
        std::stringstream ss(_points);
        int count;
        ss >> count;
        for (int i = 0; i < count; i++) {
            float x = 0;
            float y = 0;
            ss >> x >> y;
            curvemap->points.emplace_back(x, y);
        }
        auto _handlers = get_param<std::string>("_HANDLERS");
        std::stringstream hss(_handlers);
        for (int i = 0; i < count; i++) {
            float x0 = 0;
            float y0 = 0;
            float x1 = 0;
            float y1 = 0;
            hss >> x0 >> y0 >> x1 >> y1;
            curvemap->handlers.emplace_back(
                zeno::vec2f(x0, y0),
                zeno::vec2f(x1, y1)
            );
        }
        curvemap->input_min = get_param<float>("input_min");
        curvemap->input_max = get_param<float>("input_max");
        curvemap->output_min = get_param<float>("output_min");
        curvemap->output_max = get_param<float>("output_max");
        set_output("curvemap", std::move(curvemap));
    }
};

ZENDEFNODE(
    MakeCurve,
    {
        // inputs
        {
        },
        // outpus
        {
            "curvemap",
        },
        // params
        {
            {
                "float",
                "input_min",
                "0",
            },
            {
                "float",
                "input_max",
                "1",
            },
            {
                "float",
                "output_min",
                "0",
            },
            {
                "float",
                "output_max",
                "1",
            },
        },
        // category
        {
            "numeric",
        }
    }
);

}
