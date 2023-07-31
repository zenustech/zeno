#include <zeno/zeno.h>
#include <zeno/types/StringObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/utils/format.h>
#include <zeno/utils/fileio.h>
#include <zeno/extra/GlobalState.h>
#include "zeno/utils/string.h"
#include <glm/glm.hpp>
#include <glm/gtc/color_space.hpp>

namespace zeno {
namespace {
static zeno::vec3f read_vec3f(std::vector<std::string> items) {
    zeno::vec3f vec(0, 0, 0);
    int i = 0;
    for (auto item: items) {
        if (item.size() != 0) {
            vec[i] = std::stof(item);
            i += 1;
        }
    }
    return vec;
}
struct RGBHexColor : zeno::INode {
    virtual void apply() override {
        auto rgb_hex =  get_input2<std::string>("rgb_hex");
        vec3f color;
        if (zeno::starts_with(rgb_hex, "rgb") || zeno::starts_with(rgb_hex, "RGB")) {
            rgb_hex = rgb_hex.substr(4, rgb_hex.size() - 5);
            if (rgb_hex.find('.') == std::string::npos) {
                color = read_vec3f(zeno::split_str(rgb_hex, ',')) / 255.0f;
            }
            else {
                color = read_vec3f(zeno::split_str(rgb_hex, ','));
            }
        }
        else {
            if (zeno::starts_with(rgb_hex, "#")) {
                rgb_hex = rgb_hex.substr(1, rgb_hex.size() - 1);
            }
            if (rgb_hex.size() == 3) {
                rgb_hex = zeno::format(
                    "{}{}{}{}{}{}",
                    rgb_hex[0],
                    rgb_hex[0],
                    rgb_hex[1],
                    rgb_hex[1],
                    rgb_hex[2],
                    rgb_hex[2]
                );
            }
            auto r = float(std::stoi(rgb_hex.substr(0, 2), nullptr, 16)) / 255.f;
            auto g = float(std::stoi(rgb_hex.substr(2, 2), nullptr, 16)) / 255.f;
            auto b = float(std::stoi(rgb_hex.substr(4, 2), nullptr, 16)) / 255.f;
            color = {r, g, b};
        }
        set_output("color", std::make_shared<NumericObject>(color));
    }
};

ZENDEFNODE(RGBHexColor, {
    {{ "string", "rgb_hex" }},
    {
        "color"
    },
    {},
    { "utils" },
});

vec3f hsvToRgb(vec3f hsv) {
    // Reference for this technique: Foley & van Dam
    float h = hsv[0]; float s = hsv[1]; float v = hsv[2];
    if (s < 0.0001f) {
        return vec3f (v, v, v);
    } else {
        h = 6.0f * (h - floor(h));  // expand to [0..6)
        int hi = int(trunc(h));
        float f = h - float(hi);
        float p = v * (1.0f-s);
        float q = v * (1.0f-s*f);
        float t = v * (1.0f-s*(1.0f-f));
        if (hi == 0)
            return vec3f (v, t, p);
        else if (hi == 1)
            return vec3f (q, v, p);
        else if (hi == 2)
            return vec3f (p, v, t);
        else if (hi == 3)
            return vec3f (p, q, v);
        else if (hi == 4)
            return vec3f (t, p, v);
        return vec3f (v, p, q);
    }
}

struct HSV2RGB : zeno::INode {
    virtual void apply() override {
        auto hsv = get_input2<vec3f>("hsv");
        vec3f rgb = hsvToRgb(hsv);
        set_output("rgb", std::make_shared<NumericObject>(rgb));
    }
};

ZENDEFNODE(HSV2RGB, {
    {{ "vec3f", "hsv" }},
    {
        "rgb"
    },
    {},
    { "utils" },
});
}
}