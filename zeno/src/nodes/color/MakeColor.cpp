#include <sstream>

#include <zeno/zeno.h>
#include <zeno/utils/log.h>

namespace zeno {

struct MakeColor : zeno::INode {
    virtual void apply() override { 
        auto colorstr = get_input2<std::string>("color:"); // get param
        vec3f color{ 1, 1, 1 };
        if (colorstr.size() == 7) {
            colorstr = colorstr.substr(1);
            std::stringstream ss(colorstr);
            unsigned int hexColor;
            ss >> std::hex >> hexColor;
            float r = (hexColor >> 16 & 0xFF) / 255.0f;
            float g = (hexColor >> 8 & 0xFF) / 255.0f;
            float b = (hexColor & 0xFF) / 255.0f;
            color = { r, g, b };
        }
        set_output2<vec3f>("color", std::move(color));
    }
};

ZENO_DEFNODE(MakeColor)({
    {
    },
    {
        {"vec3f", "color"},
    },
    {
        {"purecolor", "color", "#FFFFFF"},
    },
    {"color"},
});

} // namespace zeno
