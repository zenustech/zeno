#include <zeno/zeno.h>
#include <zeno/PrimitiveObject.h>
#include <zeno/StringObject.h>

namespace zeno {

struct HeatmapObject : zeno::IObject {
    std::vector<zeno::vec3f> colors;

    zeno::vec3f interp(float x) {
        x = zeno::clamp(x, 0, 1) * colors.size();
        int i = (int)zeno::floor(x);
        i = zeno::clamp(i, 0, colors.size() - 2);
        float f = x - i;
        return (1 - f) * colors.at(i) + f * colors.at(i + 1);
    }
};

struct MakeHeatmap : zeno::INode {
    virtual void apply() override {
        auto nres = get_param<int>("nres");
        auto ramps = get_param<std::string>("_RAMPS");
        std::stringstream ss(ramps);
        zeno::vec3f begClr, endClr;
        ss >> begClr[0];
        ss >> begClr[1];
        ss >> begClr[2];
        ss >> endClr[0];
        ss >> endClr[1];
        ss >> endClr[2];

        auto heatmap = std::make_shared<HeatmapObject>();
        for (int i = 0; i < nres; i++) {
            float fac = i * (1.f / nres);
            zeno::vec3f clr = zeno::mix(begClr, endClr, fac);
            heatmap->colors.push_back(clr);
        }
        set_output("heatmap", std::move(heatmap));
    }
};

ZENDEFNODE(MakeHeatmap,
        { /* inputs: */ {
        }, /* outputs: */ {
        "heatmap",
        }, /* params: */ {
        {"int", "nres", "1024"},
        {"string", "_RAMPS", "0 0 0.8 0.8 0.8 1"},
        }, /* category: */ {
        "visualize",
        }});


struct PrimitiveColorByHeatmap : zeno::INode {
    virtual void apply() override {
        auto prim = get_input<zeno::PrimitiveObject>("prim");
        auto heatmap = get_input<HeatmapObject>("heatmap");
        auto srcAttr = get_param<std::string>("srcAttr");
        auto colorAttr = get_param<std::string>("colorAttr");
        auto &clr = prim->add_attr<zeno::vec3f>(colorAttr);
        auto &src = prim->attr<float>(srcAttr);

        for (int i = 0; i < src.size(); i++) {
            clr[i] = heatmap->interp(src[i]);
        }

        set_output("prim", std::move(prim));
    }
};

ZENDEFNODE(PrimitiveColorByHeatmap,
        { /* inputs: */ {
        "prim", "heatmap",
        }, /* outputs: */ {
        "prim",
        }, /* params: */ {
        {"string", "srcAttr", "rho"},
        {"string", "colorAttr", "clr"},
        }, /* category: */ {
        "visualize",
        }});

}
