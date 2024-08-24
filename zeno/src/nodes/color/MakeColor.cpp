#include <zeno/zeno.h>
#include <zeno/utils/log.h>

namespace zeno {

struct MakeColor : zeno::INode {
    virtual void apply() override { 
        auto color = get_input2<zeno::vec3f>("color");
        set_output2<vec3f>("color", std::move(color));
    }
};

ZENO_DEFNODE(MakeColor)({
    {
        {gParamType_Vec3f, "color", "1, 1, 1"},
    },
    {
        {gParamType_Vec3f, "color"},
    },
    {   
    },
    {"color"},
});

} // namespace zeno
