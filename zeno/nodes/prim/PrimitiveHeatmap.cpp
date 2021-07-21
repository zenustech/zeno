#include <zeno/zeno.h>
#include <zeno/PrimitiveObject.h>
#include <zeno/StringObject.h>

namespace zeno {

struct HeatmapObject : zeno::IObject {
    std::vector<zeno::vec3f> colors;

    zeno::vec3f interp(float x) const {
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
        std::vector<std::pair<float, zeno::vec3f>> colors;
        int count;
        ss >> count;
        for (int i = 0; i < count; i++) {
            float f = 0.f, x = 0.f, y = 0.f, z = 0.f;
            ss >> f >> x >> y >> z;
            //printf("%f %f %f %f\n", f, x, y, z);
            colors.emplace_back(
                    f, zeno::vec3f(x, y, z));
        }

        auto heatmap = std::make_shared<HeatmapObject>();
        for (int i = 0; i < nres; i++) {
            float fac = i * (1.f / nres);
            zeno::vec3f clr;
            for (int j = 0; j < colors.size(); j++) {
                auto [f, rgb] = colors[j];
                if (f >= fac) {
                    if (j != 0) {
                        auto [last_f, last_rgb] = colors[j - 1];
                        auto intfac = (fac - last_f) / (f - last_f);
                        //printf("%f %f %f %f\n", fac, last_f, f, intfac);
                        clr = zeno::mix(last_rgb, rgb, intfac);
                    } else {
                        clr = rgb;
                    }
                    break;
                }
            }
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
        //{"string", "_RAMPS", "0 0 0.8 0.8 0.8 1"},
        }, /* category: */ {
        "visualize",
        }});


struct PrimitiveColorByHeatmap : zeno::INode {
    virtual void apply() override {
        auto prim = get_input<zeno::PrimitiveObject>("prim");
        auto heatmap = get_input<HeatmapObject>("heatmap");
        auto attrName = get_param<std::string>("attrName");
        auto &clr = prim->add_attr<zeno::vec3f>("clr");
        auto &src = prim->attr<float>(attrName);

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
        {"string", "attrName", "rho"},
        }, /* category: */ {
        "visualize",
        }});

}
